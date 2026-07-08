"""Visualize YOLO bbox + SORT track ID + ArcFace match trên 1 video clip.

Usage:
  python visualize_clip.py <video_path> [output.mp4]

Output: video annotated với:
  - YOLO person bbox (xanh lá)
  - SORT track ID + bao nhiêu frames đã track (góc trên)
  - ArcFace match: tên + similarity score (vàng)
  - Presence ratio cuối clip (text overlay)
  - FPS counter
"""

from __future__ import annotations
import sys, os, time
from pathlib import Path

# Thêm path để import các module của project
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "sort"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

# ── Load models ───────────────────────────────────────────────────────────────
print("[viz] Loading models...", flush=True)
from workers.camera_worker import AIDetectionWorker

engine = AIDetectionWorker.__new__(AIDetectionWorker)
engine._INFERENCE_LOCK = __import__("threading").Lock()

# Load ArcFace + RetinaFace + Gaze via shared model cache
engine._yolo, engine._retinaface, engine._arcface, engine._gaze = AIDetectionWorker._shared_models()
import torch as _torch
engine._yolo_device = 0 if _torch.cuda.is_available() else "cpu"

# Load Sort
from sort import Sort
tracker = Sort(
    max_age=int(os.getenv("SORT_MAX_AGE", "60")),
    min_hits=int(os.getenv("SORT_MIN_HITS", "3")),
    iou_threshold=float(os.getenv("SORT_IOU_THRESHOLD", "0.3")),
)

# ── Load face gallery từ Redis ────────────────────────────────────────────────
from db import redis_client as db
import json, base64

r = db.get_client()
cam_id   = "989e86b1-9133-49a2-ad37-a215b70c083c"
cls_id   = "10495f0c-9a46-4446-a8c4-ec283b6512b7"
raw_cfg  = r.get(f"qhh:attendance:camera-class:{cam_id}:{cls_id}")
classroom = json.loads(raw_cfg) if raw_cfg else {}

gallery: list[dict] = []
for s in classroom.get("students", []):
    sid  = s["id"]
    raw  = r.hget(f"qhh:user:{sid}", "embedding")
    name = s.get("fullName") or s.get("studentCode") or sid[:8]
    if raw:
        emb_bytes = base64.b64decode(raw)
        emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
        gallery.append({"studentId": sid, "name": name, "embedding": emb})
        print(f"[viz] gallery: {name} ({sid[:8]}...) emb={emb.shape}", flush=True)
    else:
        print(f"[viz] WARNING: {name} has no embedding!", flush=True)

engine._face_gallery = gallery
# Pre-stack embeddings
if gallery:
    engine._gallery_embeddings = np.stack([g["embedding"] for g in gallery])
else:
    engine._gallery_embeddings = np.empty((0, 512), dtype=np.float32)

print(f"[viz] Gallery size: {len(gallery)}", flush=True)

# ── Similarity helper (tanh-calibrated) ──────────────────────────────────────
def _sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    dist = float(np.linalg.norm(a - b))
    return float((np.tanh((1.23132175 - dist) * 6.602259425) + 1) / 2)

THRESHOLD = 0.55

# ── Open video ────────────────────────────────────────────────────────────────
_DEFAULT_VIDEO = "/app/detection/test_sample.mkv"
video_path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_VIDEO
if not video_path or not Path(video_path).exists():
    print(f"Video not found: {video_path}")
    print("Usage: python visualize_clip.py <video_path> [output.mp4]")
    sys.exit(1)

out_path = sys.argv[2] if len(sys.argv) > 2 else "/app/detection/viz_output.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[viz] Video: {w}x{h} {fps}fps {total_frames} frames", flush=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

YOLO_CONF = float(os.getenv("YOLO_PERSON_CONF", "0.15"))

# Tracking state
track_to_name: dict[int, str]   = {}
track_frames:  dict[int, int]   = {}   # track_id → frames tracked
# Presence per student
presence: dict[str, int] = {g["name"]: 0 for g in gallery}
frame_idx = 0

print(f"[viz] Processing {total_frames} frames...", flush=True)
t0 = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    vis = frame.copy()

    # ── YOLO person detect ────────────────────────────────────────────────
    yolo_res = engine._yolo(frame, classes=[0], verbose=False,
                             device=engine._yolo_device, conf=YOLO_CONF)
    person_dets = []
    if yolo_res and len(yolo_res) > 0:
        for box, conf_v in zip(yolo_res[0].boxes.xyxy, yolo_res[0].boxes.conf):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            score = float(conf_v)
            person_dets.append([x1, y1, x2, y2, score])
            # Vẽ YOLO bbox (xanh lá nhạt)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 255, 100), 1)
            cv2.putText(vis, f"yolo {score:.2f}", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

    # ── SORT update ───────────────────────────────────────────────────────
    dets_arr = (np.array(person_dets, dtype=np.float32)
                if person_dets else np.empty((0, 5), dtype=np.float32))
    tracks = tracker.update(dets_arr)  # Nx5: [x1,y1,x2,y2,track_id]

    # Vẽ SORT tracks (cam) — gaze label sẽ được thêm sau khi tính gaze
    track_gaze: dict[int, tuple[float, float, bool]] = {}  # tid → (yaw_deg, pitch_deg, distracted)
    for trk in tracks:
        x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
        track_frames[tid] = track_frames.get(tid, 0) + 1
        name = track_to_name.get(tid, "?")
        color = (0, 165, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"T{tid} {name} ({track_frames[tid]}f)",
                    (x1, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if name != "?":
            presence[name] = presence.get(name, 0) + 1

    # ── RetinaFace + ArcFace + Gaze ──────────────────────────────────────
    YAW_TH   = float(os.getenv("GAZE_YAW_THRESHOLD",   "25"))
    PITCH_TH = float(os.getenv("GAZE_PITCH_THRESHOLD", "20"))

    face_dets = engine._retinaface.detect_all(frame)
    aligned = [d["aligned_face"] for d in face_dets if d.get("aligned_face") is not None]
    if aligned and len(gallery) > 0:
        embeds = engine._arcface.extract(aligned)
        # Gaze batch trên aligned faces
        gaze_results = []
        if engine._gaze is not None:
            try:
                gaze_results = engine._gaze.estimate_batch(aligned)
            except Exception:
                gaze_results = [None] * len(aligned)
        else:
            gaze_results = [None] * len(aligned)

        for i, (det, emb) in enumerate(zip(face_dets, embeds)):
            if emb is None:
                continue
            # Match vs gallery
            best_sim, best_name, best_sid = -1.0, "UNKNOWN", ""
            sims = []
            for g in gallery:
                s = _sim(emb, g["embedding"])
                sims.append(f"{g['name']}={s:.2f}")
                if s > best_sim:
                    best_sim, best_name, best_sid = s, g["name"], g["studentId"]
            if frame_idx % 30 == 0:
                print(f"[viz] frame={frame_idx} face_sims: {' | '.join(sims)}", flush=True)

            loc = det.get("loc")  # [x1, y1, x2, y2]
            if loc is not None:
                fx1, fy1, fx2, fy2 = [int(v) for v in loc]
                fw = fx2 - fx1

                # Gaze arrow — gaze trả radians, convert sang degrees để hiển thị
                import math
                gaze = gaze_results[i] if i < len(gaze_results) else None
                distracted = False
                if gaze is not None:
                    yaw_rad, pitch_rad = gaze          # radians từ estimator
                    yaw_deg   = math.degrees(yaw_rad)
                    pitch_deg = math.degrees(pitch_rad)
                    distracted = abs(yaw_deg) > YAW_TH or abs(pitch_deg) > PITCH_TH
                    cx_g = (fx1 + fx2) // 2
                    cy_g = (fy1 + fy2) // 2
                    length = max(fw * 2, 60)
                    dx = int(-length * math.sin(yaw_rad) * math.cos(pitch_rad))
                    dy = int(-length * math.sin(pitch_rad))
                    arrow_color = (0, 0, 255) if distracted else (0, 255, 0)
                    # Mũi tên không có chấm tròn
                    cv2.arrowedLine(vis, (cx_g, cy_g), (cx_g + dx, cy_g + dy),
                                    arrow_color, 3, cv2.LINE_AA, tipLength=0.3)

                    # Lưu gaze vào track gần nhất để vẽ dưới SORT bbox
                    for trk in tracks:
                        tx1, ty1, tx2, ty2, ttid = int(trk[0]),int(trk[1]),int(trk[2]),int(trk[3]),int(trk[4])
                        if tx1 <= cx_g <= tx2 and ty1 <= cy_g <= ty2:
                            track_gaze[ttid] = (yaw_deg, pitch_deg, distracted)
                            break

                matched = best_sim >= THRESHOLD
                # Face box: xanh (tập trung/match), đỏ (mất tập trung), xám (unknown)
                if matched:
                    fc = (0, 80, 255) if distracted else (0, 220, 0)
                else:
                    fc = (40, 40, 220)
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), fc, 2)
                label = f"{best_name} {best_sim:.2f}" if matched else f"? {best_sim:.2f}"
                cv2.putText(vis, label, (fx1, fy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, fc, 1)

                # Gán track_id → name nếu face nằm trong track bbox
                cx_face = (fx1 + fx2) // 2
                cy_face = (fy1 + fy2) // 2
                for trk in tracks:
                    tx1, ty1, tx2, ty2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
                    if tx1 <= cx_face <= tx2 and ty1 <= cy_face <= ty2:
                        if best_sim >= 0.35 and tid not in track_to_name:
                            track_to_name[tid] = best_name
                            print(f"[viz] frame={frame_idx} track T{tid} → {best_name} (sim={best_sim:.3f})", flush=True)

    # Vẽ gaze label dưới SORT track bbox
    for trk in tracks:
        x1, y1, x2, y2, tid = int(trk[0]), int(trk[1]), int(trk[2]), int(trk[3]), int(trk[4])
        g = track_gaze.get(tid)
        if g is not None:
            yaw_d, pitch_d, dist = g
            gc = (0, 0, 255) if dist else (0, 255, 0)
            cv2.putText(vis, f"yaw={yaw_d:.1f} pitch={pitch_d:.1f}",
                        (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gc, 2)

    # ── HUD: frame counter + presence ─────────────────────────────────────
    cv2.putText(vis, f"frame {frame_idx}/{total_frames}  YOLO_CONF={YOLO_CONF}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y_off = 48
    for name, cnt in presence.items():
        ratio = cnt / frame_idx if frame_idx else 0
        status = "PRESENT" if ratio >= 0.6 else "ABSENT"
        color  = (0, 220, 0) if status == "PRESENT" else (40, 40, 220)
        cv2.putText(vis, f"{name}: {cnt}/{frame_idx} ({ratio:.2f}) {status}",
                    (8, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)
        y_off += 22

    writer.write(vis)

    if frame_idx % 50 == 0:
        elapsed = time.time() - t0
        active = {int(trk[4]): track_frames.get(int(trk[4]),0) for trk in tracks}
        print(f"[viz] frame={frame_idx}/{total_frames} ({elapsed:.1f}s) active_tracks={active}", flush=True)

cap.release()
writer.release()

elapsed = time.time() - t0
print(f"\n[viz] Done in {elapsed:.1f}s → {out_path}", flush=True)
print("\n=== FINAL PRESENCE REPORT ===")
for name, cnt in presence.items():
    ratio = cnt / frame_idx if frame_idx else 0
    status = "PRESENT" if ratio >= 0.6 else "ABSENT"
    print(f"  {name}: {cnt}/{frame_idx} frames ({ratio:.2%}) → {status}")
print(f"  Total tracks: {len(track_to_name)} matched | {len(track_frames)} total")
print(f"  track_to_name: {track_to_name}")
