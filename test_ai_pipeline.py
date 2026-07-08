#!/usr/bin/env python3
"""Standalone AI pipeline tester.

Runs the project's full perception stack on a video — full-frame face
detection (RetinaFace), face recognition against a registered gallery
(ArcFace R100), and gaze estimation (ResNet-50 ONNX) — and prints one log
line per detected face per frame. No drawing is performed; the goal is to
validate the AI behaviour in isolation, without the Qt UI, the Redis
pipeline, the seat/zone logic, or YOLO body detection.

Gallery folder layout::

    gallery/
        Alice.jpg          # one face per file, filename = student name
        Bob_001.png        # underscore + code is stripped from the label
        Charlie/           # OR a directory per person
            front.jpg
            side.jpg

Each registration image is fed through RetinaFace to obtain the
landmark-aligned 112×112 face used during recognition, then through
ArcFace to produce the L2-normalised 512-dim embedding. At runtime, each
detected face's embedding is compared (cosine-tanh similarity) against the
gallery; the top match above ``--threshold`` wins.

Example::

    python test_ai_pipeline.py \\
        --video classroom.mp4 \\
        --gallery /home/mq/.classroom_manager/faces \\
        --log gaze.log

The script intentionally bypasses the YOLO person detector — it runs
RetinaFace on the full frame so you can verify face/gaze quality without
worrying about body-bbox crops.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow running from anywhere — the project's package layout assumes the
# repo root is on sys.path so workers/* imports resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workers.face_models import (  # noqa: E402
    RetinaFaceDetector,
    ArcFaceExtractor,
)
from workers.libs.face_preprocess import preprocess as _face_align  # noqa: E402
from workers.gaze_estimator import (  # noqa: E402
    GazeEstimator,
    DistractionTracker,
)

# Reference face detector — same library the gaze-estimation demo uses.
# When --use-uniface is passed, the pipeline switches to this detector and
# crops faces with no padding, exactly like onnx_inference.py.
try:
    from uniface import RetinaFace as UnifaceRetinaFace  # noqa: E402
    _HAS_UNIFACE = True
except ImportError:
    UnifaceRetinaFace = None
    _HAS_UNIFACE = False


def detect_faces_uniface(detector, frame):
    """Adapt uniface output to the same dict shape as detect_all_faces."""
    raw = detector.detect(frame)
    # uniface may return either a list of Face objects (with .bbox) or a
    # tuple (boxes, landmarks). Normalise both.
    if isinstance(raw, tuple) and len(raw) == 2:
        boxes, _landms = raw
        faces = boxes
    else:
        faces = raw
    out = []
    h, w = frame.shape[:2]
    for f in faces:
        bbox = getattr(f, "bbox", f)
        b = np.asarray(bbox, dtype=float).flatten()
        x1, y1, x2, y2 = b[:4]
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(w, int(x2)); y2 = min(h, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = frame[y1:y2, x1:x2]
        out.append({
            "loc": np.array([x1, y1, x2, y2], dtype=np.float32),
            "landm": np.array([], dtype=np.float32),
            "conf": float(b[4]) if len(b) > 4 else 1.0,
            "aligned_face": crop,  # used only for ArcFace; gaze never uses it
        })
    return out


def detect_all_faces(
    rf: RetinaFaceDetector,
    image: np.ndarray,
    conf_thres: float = 0.5,
) -> list[dict]:
    """RetinaFace forward pass that keeps EVERY face, without the project's
    "biggest face" / 2%-area-gate rules.

    The shipped RetinaFaceDetector is tuned for person-crop inference inside
    the classroom pipeline — it returns at most one face per crop and drops
    faces smaller than 2% of the input frame area. On a full classroom frame
    that filters out every realistic face, so we re-do the postprocess here.
    """
    h, w = image.shape[:2]
    # Aspect-ratio-preserving letterbox — same as uniface and the live
    # RetinaFaceDetector._infer. A naive resize squashes 1920x1080 into
    # 640x640 and shifts bbox centres by tens of pixels.
    canvas, factor = rf._letterbox_resize(image, tuple(rf.image_size))
    inp = rf._preprocess_batch([canvas])
    locs, confs, landms = rf.sess.run(None, {rf.sess.get_inputs()[0].name: inp})

    priors = rf.priors[None]
    boxes = rf._decode_boxes_batch(locs, priors, rf.variance)[0]
    scores = confs[0, :, 1]
    lms = rf._decode_landmarks_batch(landms, priors, rf.variance)[0]

    # Decoded coords are normalised to the network input (0..1). Scale up to
    # the letterbox canvas (input_size in pixels), then undo the resize factor
    # to land on original-frame pixels.
    net_w, net_h = rf.image_size
    boxes[:, [0, 2]] *= net_w
    boxes[:, [1, 3]] *= net_h
    lms[:, 0::2] *= net_w
    lms[:, 1::2] *= net_h
    boxes /= factor
    lms /= factor

    inds = np.where(scores > conf_thres)[0]
    if inds.size == 0:
        return []
    boxes, scores, lms = boxes[inds], scores[inds], lms[inds]
    order = scores.argsort()[::-1]
    boxes, scores, lms = boxes[order], scores[order], lms[order]

    dets = np.hstack((boxes, scores[:, None])).astype(np.float32, copy=False)
    keep = rf._nms(dets, rf.iou_thres)
    dets = dets[keep]
    lms = lms[keep]

    out: list[dict] = []
    for det, lm in zip(dets, lms):
        x1 = float(np.clip(det[0], 0, w))
        y1 = float(np.clip(det[1], 0, h))
        x2 = float(np.clip(det[2], 0, w))
        y2 = float(np.clip(det[3], 0, h))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        lm = np.asarray(lm, dtype=np.float32)
        # Convert RetinaFace's flat 10 numbers (x1,y1,x2,y2,...) into the
        # (2, 5) [[xs], [ys]] layout face_preprocess expects.
        landmarks = np.array([
            lm[0], lm[2], lm[4], lm[6], lm[8],
            lm[1], lm[3], lm[5], lm[7], lm[9],
        ], dtype=np.float32).reshape(2, 5).T
        aligned = _face_align(image, bbox.copy(), landmarks, image_size=rf.align_size)
        aligned = cv2.resize(aligned, tuple(rf.align_size), interpolation=cv2.INTER_AREA)
        out.append({
            "loc": bbox,
            "landm": lm,
            "conf": float(det[4]),
            "aligned_face": aligned,
        })
    return out


# ── Gallery loading ──────────────────────────────────────────────────────


def _label_from_path(path: Path) -> str:
    """Convert ``Alice_001.jpg`` → ``Alice``; directory name wins if used."""
    stem = path.stem
    return stem.split("_", 1)[0] if "_" in stem else stem


def _iter_gallery_images(root: Path):
    """Yield (label, image_path) pairs from either a flat folder or per-person subdirs."""
    if not root.exists():
        raise FileNotFoundError(f"Gallery directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Gallery is not a directory: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            label = entry.name
            for img in sorted(entry.iterdir()):
                if img.suffix.lower() in exts:
                    yield label, img
        elif entry.suffix.lower() in exts:
            yield _label_from_path(entry), entry


def build_gallery(
    root: Path,
    retinaface: RetinaFaceDetector,
    arcface: ArcFaceExtractor,
) -> list[dict]:
    """Run RetinaFace + ArcFace on every gallery image and return its embeddings."""
    items: list[tuple[str, np.ndarray]] = []
    skipped: list[tuple[Path, str]] = []
    for label, img_path in _iter_gallery_images(root):
        image = cv2.imread(str(img_path))
        if image is None:
            skipped.append((img_path, "unreadable image"))
            continue
        dets = detect_all_faces(retinaface, image, conf_thres=0.3)
        if not dets:
            skipped.append((img_path, "no face found"))
            continue
        # pick the biggest face per registration image
        dets.sort(
            key=lambda d: (d["loc"][2] - d["loc"][0]) * (d["loc"][3] - d["loc"][1]),
            reverse=True,
        )
        aligned = dets[0].get("aligned_face")
        if aligned is None:
            skipped.append((img_path, "no aligned face"))
            continue
        embedding = arcface.extract_single(aligned)
        if embedding is None:
            skipped.append((img_path, "ArcFace failure"))
            continue
        items.append((label, embedding.astype(np.float32)))

    print(f"[gallery] loaded {len(items)} face(s) from {root}", flush=True)
    for path, reason in skipped:
        print(f"[gallery] SKIP {path.name}: {reason}", flush=True)

    # Average multiple images of the same person → one robust template.
    by_label: dict[str, list[np.ndarray]] = {}
    for label, emb in items:
        by_label.setdefault(label, []).append(emb)
    gallery = []
    for label, embs in by_label.items():
        stacked = np.stack(embs).astype(np.float32)
        mean = stacked.mean(axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 0:
            mean = mean / norm
        gallery.append({
            "label": label,
            "embedding": mean.astype(np.float32),
            "sample_count": len(embs),
        })
        print(
            f"[gallery]   • {label:<24} {len(embs)} image(s) → template ready",
            flush=True,
        )
    return gallery


# ── Recognition ──────────────────────────────────────────────────────────


def match_embeddings(
    embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery: list[dict],
    threshold: float,
) -> list[tuple[str, float]]:
    """Tanh-calibrated similarity matching (same formula as the live pipeline)."""
    if len(gallery_embeddings) == 0:
        return [("", -1.0) for _ in embeddings]
    distances = np.linalg.norm(
        embeddings[:, None, :] - gallery_embeddings[None, :, :], axis=2
    )
    similarities = (
        np.tanh((1.23132175 - distances) * 6.602259425) + 1.0
    ) / 2.0
    best_idx = np.argmax(similarities, axis=1)
    out: list[tuple[str, float]] = []
    for row, idx in enumerate(best_idx):
        score = float(similarities[row, idx])
        label = gallery[int(idx)]["label"] if score >= threshold else ""
        out.append((label, score))
    return out


# ── Main loop ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to input video file (or '0' for webcam)")
    p.add_argument(
        "--gallery",
        required=True,
        help="Directory of registered face images (flat or per-person subfolders)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Min similarity (0..1) for a positive recognition match",
    )
    p.add_argument(
        "--face-conf",
        type=float,
        default=0.5,
        help="Min RetinaFace confidence to keep a face (lower = more faces)",
    )
    p.add_argument(
        "--every",
        type=int,
        default=1,
        help="Process every Nth frame to save time (default 1 = every frame)",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many processed frames (0 = unlimited)",
    )
    p.add_argument(
        "--gaze-pad",
        type=float,
        default=0.0,
        help="Padding ratio around the face bbox before gaze model. "
             "0.0 = tight crop (matches gaze-estimation/onnx_inference.py). "
             "Larger pad → more stable gaze (less jitter when bbox wobbles).",
    )
    p.add_argument(
        "--use-uniface",
        action="store_true",
        help="Use uniface.RetinaFace (the detector the gaze-estimation demo "
             "uses) instead of the project's RetinaFaceDetector. With this "
             "flag plus --gaze-pad 0.0 the pipeline is bit-identical to "
             "gaze-estimation/onnx_inference.py.",
    )
    p.add_argument("--yaw-threshold", type=float, default=20.0)
    p.add_argument("--pitch-threshold", type=float, default=18.0)
    p.add_argument("--alert-after", type=float, default=0.8)
    p.add_argument("--clear-after", type=float, default=0.6)
    p.add_argument("--ema-alpha", type=float, default=0.35)
    p.add_argument(
        "--log",
        default="",
        help="Optional CSV path to dump every face's per-frame readings",
    )
    p.add_argument(
        "--output",
        default="",
        help="Optional path to write an annotated MP4 (bbox + label + gaze)",
    )
    return p.parse_args()


# ─── Visualisation: identical to gaze-estimation/utils/helpers.py ────────
# draw_bbox + draw_gaze + draw_bbox_gaze copied verbatim so the output video
# looks exactly like the reference demo (corner-style bbox + red gaze arrow).


def draw_bbox(image, bbox, color=(0, 255, 0), thickness=2, proportion=0.2):
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    width = x_max - x_min
    height = y_max - y_min
    corner_length = int(proportion * min(width, height))
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 1)
    # Top-left
    cv2.line(image, (x_min, y_min), (x_min + corner_length, y_min), color, thickness)
    cv2.line(image, (x_min, y_min), (x_min, y_min + corner_length), color, thickness)
    # Top-right
    cv2.line(image, (x_max, y_min), (x_max - corner_length, y_min), color, thickness)
    cv2.line(image, (x_max, y_min), (x_max, y_min + corner_length), color, thickness)
    # Bottom-left
    cv2.line(image, (x_min, y_max), (x_min, y_max - corner_length), color, thickness)
    cv2.line(image, (x_min, y_max), (x_min + corner_length, y_max), color, thickness)
    # Bottom-right
    cv2.line(image, (x_max, y_max), (x_max, y_max - corner_length), color, thickness)
    cv2.line(image, (x_max, y_max), (x_max - corner_length, y_max), color, thickness)


def draw_gaze(frame, bbox, pitch, yaw, thickness=2, color=(0, 0, 255)):
    """Pitch + yaw IN RADIANS — same units the reference returns."""
    x_min, y_min, x_max, y_max = map(int, bbox[:4])
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    length = x_max - x_min
    dx = int(-length * np.sin(yaw) * np.cos(pitch))
    dy = int(-length * np.sin(pitch))
    cv2.circle(frame, (x_center, y_center), radius=4, color=color, thickness=-1)
    cv2.arrowedLine(
        frame, (x_center, y_center), (x_center + dx, y_center + dy),
        color=color, thickness=thickness,
        line_type=cv2.LINE_AA, tipLength=0.25,
    )


def draw_bbox_gaze(frame, bbox, pitch, yaw):
    draw_bbox(frame, bbox)
    draw_gaze(frame, bbox, pitch, yaw)


def main():
    args = parse_args()

    print("[init] loading models…", flush=True)
    if args.use_uniface:
        if not _HAS_UNIFACE:
            print("[fatal] --use-uniface requested but uniface is not installed.\n"
                  "        Install with: pip install uniface", flush=True)
            sys.exit(1)
        uniface_det = UnifaceRetinaFace()
        retinaface = None
        face_backend = "uniface (reference detector)"
    else:
        retinaface = RetinaFaceDetector()
        uniface_det = None
        face_backend = "project RetinaFaceDetector"
    arcface = ArcFaceExtractor()
    gaze = GazeEstimator()
    print(
        f"[gaze]  input_size={gaze.input_size}  pad_ratio={args.gaze_pad}  "
        f"backend={face_backend}  providers={gaze.session.get_providers()}",
        flush=True,
    )

    def run_face_detect(img):
        if uniface_det is not None:
            return detect_faces_uniface(uniface_det, img)
        return detect_all_faces(retinaface, img, conf_thres=args.face_conf)

    # The gallery always uses the project's RetinaFaceDetector because ArcFace
    # needs a landmark-aligned 112x112 face crop, which uniface doesn't expose.
    gallery_detector = retinaface if retinaface is not None else RetinaFaceDetector()
    gallery = build_gallery(Path(args.gallery), gallery_detector, arcface)
    if not gallery:
        print("[gallery] WARNING: empty — recognition disabled", flush=True)
    gallery_embeddings = (
        np.stack([g["embedding"] for g in gallery]).astype(np.float32)
        if gallery else np.empty((0, 512), dtype=np.float32)
    )

    tracker = DistractionTracker(
        yaw_threshold_deg=args.yaw_threshold,
        pitch_threshold_deg=args.pitch_threshold,
        alert_after_sec=args.alert_after,
        clear_after_sec=args.clear_after,
        ema_alpha=args.ema_alpha,
    )

    src = args.video
    try:
        src = int(args.video)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[fatal] cannot open video: {args.video}", flush=True)
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[video] {args.video}  fps≈{fps:.1f}  size={src_w}x{src_h}",
        flush=True,
    )

    writer = None
    if args.output:
        # Effective FPS depends on --every. Output keeps the same wall-clock
        # duration as the input by dividing the source fps accordingly.
        out_fps = max(1.0, fps / max(1, args.every))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, out_fps, (src_w, src_h))
        if not writer.isOpened():
            print(f"[fatal] cannot open output video: {args.output}", flush=True)
            sys.exit(1)
        print(f"[output] writing annotated video to {args.output} @ {out_fps:.1f} fps", flush=True)

    csv_writer = None
    csv_file = None
    if args.log:
        csv_file = open(args.log, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "t_sec", "face_idx",
            "x1", "y1", "x2", "y2",
            "label", "score",
            "yaw_raw_deg", "pitch_raw_deg",
            "yaw_smooth_deg", "pitch_smooth_deg",
            "focused", "alert", "distracted_for_sec",
        ])

    frame_idx = -1
    processed = 0
    started_at = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % args.every != 0:
                continue
            t_sec = frame_idx / fps

            annotated = frame.copy() if writer is not None else frame

            # 1) Face detection on the FULL frame. Either path returns the
            #    same dict shape so the rest of the loop is detector-agnostic.
            dets = run_face_detect(frame)
            if not dets:
                processed += 1
                if processed % 30 == 0:
                    print(f"[frame {frame_idx:>5}] no face", flush=True)
                if writer is not None:
                    # No face → write the raw frame, exactly like the reference.
                    writer.write(annotated)
                continue

            # 2) ArcFace embeddings for all faces in one batch
            aligned_faces = [
                d.get("aligned_face") for d in dets
                if d.get("aligned_face") is not None
            ]
            embeddings = arcface.extract(aligned_faces)
            matches: list[tuple[str, float]] = [("", -1.0)] * len(aligned_faces)
            if embeddings is not None and gallery_embeddings.shape[0] > 0:
                matches = match_embeddings(
                    embeddings.astype(np.float32),
                    gallery_embeddings, gallery, args.threshold,
                )

            # 3) Gaze for every face crop (with padding) in one batched call
            gaze_crops: list[np.ndarray] = []
            face_meta: list[dict] = []
            h, w = frame.shape[:2]
            for det in dets:
                x1, y1, x2, y2 = map(int, det["loc"][:4])
                fw, fh = max(1, x2 - x1), max(1, y2 - y1)
                pad = int(args.gaze_pad * max(fw, fh))
                gx1 = max(0, x1 - pad)
                gy1 = max(0, y1 - pad)
                gx2 = min(w, x2 + pad)
                gy2 = min(h, y2 + pad)
                crop = frame[gy1:gy2, gx1:gx2]
                gaze_crops.append(crop)
                face_meta.append({"loc": (x1, y1, x2, y2)})
            gaze_results = gaze.estimate_batch(gaze_crops) if gaze_crops else []

            # 4) Combine + log. GazeEstimator now returns radians (reference
            # behaviour) — convert to degrees once here.
            now = time.monotonic()
            for i, det in enumerate(dets):
                x1, y1, x2, y2 = face_meta[i]["loc"]
                label, score = (matches[i] if i < len(matches) else ("", -1.0))
                yaw_rad, pitch_rad = (
                    gaze_results[i] if i < len(gaze_results) else (0.0, 0.0)
                )
                yaw = float(np.degrees(yaw_rad))
                pitch = float(np.degrees(pitch_rad))
                # Key on the recognised identity when we have one — otherwise
                # use a quantised bbox-centre bucket so each "unknown" person
                # still gets a stable short-term track for the EMA / hysteresis.
                key = (
                    f"id:{label}" if label else
                    f"bbox:{((x1+x2)//2)//32}:{((y1+y2)//2)//32}"
                )
                flags = tracker.update(key, float(yaw), float(pitch), now)
                marker = "🚨" if flags["gaze_alert"] else (
                    "·" if flags["gaze_focused"] else "!"
                )
                name = label or "UNKNOWN"
                score_str = f"{score:.2f}" if score >= 0 else " n/a"
                ys = flags["gaze_yaw_smooth"]
                ps = flags["gaze_pitch_smooth"]
                print(
                    f"[f{frame_idx:>5} t={t_sec:6.2f}s] {marker} face#{i}  "
                    f"id={name:<14} sim={score_str}  "
                    f"yaw_raw={yaw:+6.1f}° (sm={ys:+6.1f}°)  "
                    f"pitch_raw={pitch:+6.1f}° (sm={ps:+6.1f}°)  "
                    f"focused={str(flags['gaze_focused']):5} "
                    f"alert={str(flags['gaze_alert']):5} "
                    f"d_for={flags['gaze_distracted_for']:.2f}s",
                    flush=True,
                )
                if csv_writer is not None:
                    csv_writer.writerow([
                        frame_idx, f"{t_sec:.3f}", i,
                        x1, y1, x2, y2,
                        name, f"{score:.4f}",
                        f"{yaw:.2f}", f"{pitch:.2f}",
                        f"{ys:.2f}", f"{ps:.2f}",
                        int(flags["gaze_focused"]),
                        int(flags["gaze_alert"]),
                        f"{flags['gaze_distracted_for']:.3f}",
                    ])
                if writer is not None:
                    # Reference-style overlay: corner bbox + red gaze arrow.
                    # draw_gaze expects radians, so pass yaw_rad/pitch_rad.
                    draw_bbox_gaze(
                        annotated, (x1, y1, x2, y2), pitch_rad, yaw_rad,
                    )

            if writer is not None:
                writer.write(annotated)
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    finally:
        cap.release()
        if csv_file is not None:
            csv_file.close()
        if writer is not None:
            writer.release()

    dur = time.monotonic() - started_at
    rate = processed / dur if dur > 0 else 0.0
    print(
        f"[done] processed {processed} frame(s) in {dur:.1f}s "
        f"({rate:.1f} fps effective)",
        flush=True,
    )
    if args.log:
        print(f"[done] CSV log: {args.log}", flush=True)
    if args.output:
        print(f"[done] annotated video: {args.output}", flush=True)


if __name__ == "__main__":
    main()
