"""Test plan_distraction.md v2: gaze → board-line ray intersection.

Bắt mặt GIỐNG HỆT luồng AI production của worker:
    YOLO person bbox → RetinaFace full frame (detect_all)
    → gán face vào person bbox chứa tâm face
    → gaze trên crop SÁT bbox RetinaFace từ frame gốc
Tất cả qua chính `AIDetectionWorker._recognize_faces_in_person_crops`
(camera_worker.py) — không tự chế lại bước detect.

Phân loại attention theo plan §8:
    g = (-sin(yaw)cos(pitch), -sin(pitch))   [KHÔNG normalize]
    ‖g‖ < sin(CENTRAL_CONE)  → Focused (central cone, §5)
    gaze_hits_board (§6b)    → Focused / Distracted

Board line CỐ ĐỊNH cho toàn video: mặc định 2 góc đáy frame,
override bằng env BOARD_L="x,y" BOARD_R="x,y".

Usage (trong worker container):
    python3 /app/test_gaze_wedge.py <video_in> [video_out]
"""

from __future__ import annotations
import math
import os
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

# ── Tham số plan ──────────────────────────────────────────────────────
CENTRAL_CONE_RAD = math.radians(float(os.getenv("CENTRAL_CONE_DEG", "10")))
MARGIN_PX = float(os.getenv("BOARD_MARGIN_PX", "80"))
YOLO_CONF = float(os.getenv("YOLO_PERSON_CONF", "0.15"))
# §6c: pitch limit — so pitch với GÓC NÂNG trong hệ camera (pinhole),
# KHÔNG dùng góc TL-O-L trên mặt phẳng ảnh (đã thử: góc ảnh 2D ~111° trong
# khi pitch tối đa ~42° → không bao giờ báo; hai đại lượng khác hệ quy chiếu).
VFOV_RAD = math.radians(float(os.getenv("CAMERA_VFOV_DEG", "55")))
LOOKUP_MARGIN_RAD = math.radians(float(os.getenv("LOOKUP_MARGIN_DEG", "5")))
# Cảnh báo khi 1 track distraction liên tục đủ N frame (Distracted/LookingUp).
# Frame mất mặt KHÔNG reset chuỗi (quay đầu làm mất face là chuyện thường);
# chỉ frame Focused mới reset.
ALERT_FRAMES = int(os.getenv("DISTRACT_ALERT_FRAMES", "100"))


def gaze_vector(yaw: float, pitch: float) -> tuple[float, float]:
    """(yaw, pitch) RADIAN → vector 2D trên ảnh (không normalize, §3)."""
    dx = -math.sin(yaw) * math.cos(pitch)
    dy = -math.sin(pitch)
    return dx, dy


def gaze_hits_board(O, g, L, R, margin_px=MARGIN_PX):
    """§6b: tia gaze từ O giao đường ngang y=Ly, x giao trong [Lx, Rx]±margin."""
    dx, dy = g
    if abs(dy) < 1e-6:
        return False, None
    t = (L[1] - O[1]) / dy
    if t <= 0:
        return False, None  # gaze ngược hướng bảng
    x_hit = O[0] + dx * t
    ok = (L[0] - margin_px) <= x_hit <= (R[0] + margin_px)
    return ok, x_hit


def attention_state(O, yaw, pitch, L, R, t_y, pitch_per_pix):
    """§8: central cone → pitch limit (§6c) → ray hit (§6b).

    §6c: alpha_top = góc nâng (hệ camera, pinhole) từ hàng pixel của mặt
    lên hàng pixel của đường TL-TR: (O.y - t_y) * VFOV/H. Cùng đơn vị và
    gốc quy chiếu với pitch của gaze model → so sánh trực tiếp được.
    pitch > alpha_top + margin → LookingUp. t_y=None → tắt check.

    Trả (state, x_hit, gnorm, alpha_top).
    """
    dx, dy = gaze_vector(yaw, pitch)
    gnorm = math.hypot(dx, dy)
    if gnorm < math.sin(CENTRAL_CONE_RAD):
        return "Focused", None, gnorm, None  # nhìn gần trục camera (§5)
    alpha_top = None
    if t_y is not None:
        alpha_top = (O[1] - t_y) * pitch_per_pix
        if pitch > alpha_top + LOOKUP_MARGIN_RAD:
            return "LookingUp", None, gnorm, alpha_top
    ok, x_hit = gaze_hits_board(O, (dx, dy), L, R)
    return ("Focused" if ok else "Distracted"), x_hit, gnorm, alpha_top


def _build_engine():
    """Engine Qt-free tái dùng AIDetectionWorker — như visualize_clip.py,
    kèm đủ attr mà `_recognize_faces_in_person_crops` cần."""
    from workers.camera_worker import AIDetectionWorker, _FpsAggregator

    eng = AIDetectionWorker.__new__(AIDetectionWorker)
    eng._INFERENCE_LOCK = threading.Lock()
    eng._yolo, eng._retinaface, eng._arcface, eng._gaze = (
        AIDetectionWorker._shared_models()
    )
    import torch
    eng._yolo_device = 0 if torch.cuda.is_available() else "cpu"
    eng._fps = _FpsAggregator(3600.0, "wedge")
    eng._fps_yolo = eng._fps.get("yolo")
    eng._fps_retina = eng._fps.get("retina")
    eng._fps_arc = eng._fps.get("arc")
    eng._fps_gaze = eng._fps.get("gaze")
    eng._gallery_embeddings = np.empty((0, 512), dtype=np.float32)
    return eng


def main() -> None:
    video_in = sys.argv[1] if len(sys.argv) > 1 else "/app/detection/video_lop_qhh.mp4"
    video_out = sys.argv[2] if len(sys.argv) > 2 else "/app/detection/gaze_wedge_output.mp4"

    print("[wedge] loading models (production pipeline)...", flush=True)
    eng = _build_engine()

    # SORT tracker — như clip_inference, để giữ track_id qua các frame.
    sys.path.insert(0, str(_ROOT / "sort"))
    from sort import Sort  # type: ignore
    tracker = Sort(
        max_age=int(os.getenv("SORT_MAX_AGE", "60")),
        min_hits=int(os.getenv("SORT_MIN_HITS", "3")),
        iou_threshold=float(os.getenv("SORT_IOU_THRESHOLD", "0.3")),
    )

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_in}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[wedge] {video_in}: {W}x{H} {fps}fps {total} frames", flush=True)

    # Board line: CỐ ĐỊNH TUYỆT ĐỐI cho toàn bộ video — chốt 1 lần duy nhất.
    def _pt_env(name: str, default: tuple[float, float]) -> tuple[float, float]:
        raw = os.getenv(name, "")
        if raw:
            x, y = raw.split(",")
            return float(x), float(y)
        return default

    L = _pt_env("BOARD_L", (0.0, float(H - 1)))           # góc dưới-trái frame
    R = _pt_env("BOARD_R", (float(W - 1), float(H - 1)))  # góc dưới-phải frame
    # §6c: TL/TR đặt CAO hẳn — 10% chiều cao frame, sát mép trên.
    TL = _pt_env("TOP_L", (0.0, float(H) * 0.10))          # cao, cạnh trái
    TR = _pt_env("TOP_R", (float(W - 1), float(H) * 0.10)) # cao, cạnh phải
    T_Y = (TL[1] + TR[1]) / 2.0          # đường ngang — logic dùng cao độ y
    pitch_per_pix = VFOV_RAD / float(H)  # pinhole: rad / pixel dọc
    print(f"[wedge] board line FIXED for whole video: L={L} R={R} "
          f"cone={math.degrees(CENTRAL_CONE_RAD):.0f}deg margin={MARGIN_PX}px", flush=True)
    print(f"[wedge] pitch limit: TL={TL} TR={TR} vfov={math.degrees(VFOV_RAD):.0f}deg "
          f"lookup_margin={math.degrees(LOOKUP_MARGIN_RAD):.0f}deg", flush=True)

    writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    counts = {"Focused": 0, "Distracted": 0, "LookingUp": 0}
    cone_hits = 0
    faces_seen = 0
    frame_idx = 0
    t0 = time.time()

    # ── Distraction alert theo track ─────────────────────────────────
    distract_run: dict[int, int] = {}   # tid → số frame distraction liên tục
    alerted: set[int] = set()           # tid đã báo trong chuỗi hiện tại
    alerts: list[dict] = []             # log các lần báo
    alert_dir = Path("/app/detection/distraction_alerts") / Path(video_in).stem
    print(f"[wedge] distraction alert: >={ALERT_FRAMES} consecutive frames "
          f"-> save to {alert_dir}", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        vis = frame

        cv2.line(vis, (int(L[0]), int(L[1]) - 3), (int(R[0]), int(R[1]) - 3),
                 (0, 220, 255), 6)
        # Đường pitch limit (§6c) — tím magenta
        cv2.line(vis, (int(TL[0]), int(TL[1])), (int(TR[0]), int(TR[1])),
                 (255, 0, 255), 4)
        cv2.putText(vis, "PITCH LIMIT", (int(TL[0]) + 10, int(TL[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        # ── 1. YOLO person — giống production ─────────────────────────
        yolo_res = eng._yolo(frame, classes=[0], verbose=False,
                             device=eng._yolo_device, conf=YOLO_CONF)
        person_bboxes: list[tuple[int, int, int, int]] = []
        if yolo_res and len(yolo_res) > 0:
            for box in yolo_res[0].boxes.xyxy:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                person_bboxes.append((x1, y1, x2 - x1, y2 - y1))  # x,y,w,h
                cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # ── 1b. SORT update — giữ track_id ổn định qua các frame ──────
        dets_arr = (np.array([[px, py, px + pw, py + ph, 0.9]
                              for (px, py, pw, ph) in person_bboxes],
                             dtype=np.float32)
                    if person_bboxes else np.empty((0, 5), dtype=np.float32))
        tracks = tracker.update(dets_arr)

        def _iou(a, b):
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
            inter = iw * ih
            ua = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
            return inter / ua if ua > 0 else 0.0

        # person bbox index → track_id (IoU cao nhất, >= 0.3)
        pb_to_tid: dict[int, int] = {}
        for i, (px, py, pw, ph) in enumerate(person_bboxes):
            best_tid, best_iou = None, 0.3
            for tk in tracks:
                v = _iou((px, py, px + pw, py + ph), tk[:4])
                if v > best_iou:
                    best_iou, best_tid = v, int(tk[4])
            if best_tid is not None:
                pb_to_tid[i] = best_tid

        # ── 2. RetinaFace + gaze — ĐÚNG hàm production của worker ─────
        # (full-frame detect_all → gán face vào person bbox → gaze trên
        #  crop sát bbox RetinaFace; trả gaze_yaw_deg/gaze_pitch_deg ĐỘ)
        infos = eng._recognize_faces_in_person_crops(frame, person_bboxes, [])

        pending_alerts: list[tuple[int, tuple[int, int, int, int]]] = []
        for pb_idx, info in enumerate(infos):
            if not info.get("face_found"):
                continue
            fb = info.get("face_box")  # (x, y, w, h)
            yaw_deg = info.get("gaze_yaw_deg")
            pitch_deg = info.get("gaze_pitch_deg")
            if fb is None or yaw_deg is None or pitch_deg is None:
                continue
            faces_seen += 1
            fx, fy, fw, fh = fb
            x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
            O = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)  # center face box (§2)

            # Worker trả ĐỘ — convert về radian MỘT lần tại đây (plan §9).
            yaw = math.radians(float(yaw_deg))
            pitch = math.radians(float(pitch_deg))

            state, x_hit, gnorm, alpha_top = attention_state(
                O, yaw, pitch, L, R, T_Y, pitch_per_pix)
            counts[state] += 1
            in_cone = state == "Focused" and x_hit is None
            if in_cone:
                cone_hits += 1

            # ── Đếm distraction LIÊN TỤC theo track ───────────────────
            # Chưa đủ ALERT_FRAMES liên tục → attention vẫn true.
            # Frame Focused reset chuỗi; frame mất mặt giữ nguyên (không
            # cộng, không reset).
            tid = pb_to_tid.get(pb_idx)
            run = 0
            if tid is not None:
                if state in ("Distracted", "LookingUp"):
                    run = distract_run.get(tid, 0) + 1
                    distract_run[tid] = run
                    if run >= ALERT_FRAMES and tid not in alerted:
                        alerted.add(tid)
                        pending_alerts.append((tid, (x1, y1, x2, y2)))
                else:  # Focused → chuỗi đứt, attention lại true
                    distract_run[tid] = 0
                    alerted.discard(tid)

            if state == "Focused":
                color = (0, 220, 0)
            elif state == "LookingUp":
                color = (255, 0, 255)   # magenta — cùng màu đường pitch limit
            else:
                color = (0, 0, 255)

            # ── Wedge quan sát của người này: O → L và O → R (§4) ──────
            # Tam giác O-L-R tô mờ + 2 cạnh biên, để thấy vùng gaze hợp lệ.
            wedge = np.array([[int(O[0]), int(O[1])],
                              [int(L[0]), int(L[1])],
                              [int(R[0]), int(R[1])]], dtype=np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [wedge], (255, 200, 60))       # xanh cyan nhạt
            cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)    # alpha 18%
            cv2.line(vis, (int(O[0]), int(O[1])), (int(L[0]), int(L[1])),
                     (255, 200, 60), 2, cv2.LINE_AA)             # v_left
            cv2.line(vis, (int(O[0]), int(O[1])), (int(R[0]), int(R[1])),
                     (255, 200, 60), 2, cv2.LINE_AA)             # v_right
            # Cạnh cao độ O→TL (tím) — cùng O→L tạo góc alpha_top (§6c)
            cv2.line(vis, (int(O[0]), int(O[1])), (int(TL[0]), int(TL[1])),
                     (255, 0, 255), 2, cv2.LINE_AA)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

            dx, dy = gaze_vector(yaw, pitch)
            ln = max(fw * 2, 80)
            cv2.arrowedLine(vis, (int(O[0]), int(O[1])),
                            (int(O[0] + dx * ln / max(gnorm, 1e-6)),
                             int(O[1] + dy * ln / max(gnorm, 1e-6))),
                            color, 3, cv2.LINE_AA, tipLength=0.25)

            if x_hit is not None:
                xh = int(np.clip(x_hit, -200, W + 200))
                cv2.circle(vis, (xh, int(L[1]) - 3), 14, color, -1)
                cv2.line(vis, (int(O[0]), int(O[1])), (xh, int(L[1]) - 3),
                         color, 1, cv2.LINE_AA)

            if in_cone:
                tag = "CONE"
            elif state == "LookingUp":
                tag = f"UP pitch>{math.degrees(alpha_top):.0f}"
            else:
                tag = f"hit={int(x_hit)}" if x_hit is not None else "no-hit"
            if alpha_top is not None:
                tag += f" a={math.degrees(alpha_top):.0f}"
            if tid is not None:
                tag = f"T{tid} " + tag
                if run:
                    tag += f" run={run}"
                if tid in alerted:
                    cv2.putText(vis, "DISTRACTION ALERT", (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
            cv2.putText(vis, f"{state} {tag}", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(vis, f"yaw={yaw_deg:.0f} pitch={pitch_deg:.0f} |g|={gnorm:.2f}",
                        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(vis, f"frame {frame_idx}/{total}  persons={len(person_bboxes)}  "
                         f"board=bottom-edge cone={math.degrees(CENTRAL_CONE_RAD):.0f}deg "
                         f"margin={MARGIN_PX:.0f}px",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        # ── Lưu ảnh cảnh báo (frame đã vẽ đầy đủ overlay) ─────────────
        for tid, fb in pending_alerts:
            alert_dir.mkdir(parents=True, exist_ok=True)
            fpath = alert_dir / f"T{tid}_frame{frame_idx:06d}.jpg"
            cv2.imwrite(str(fpath), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            alerts.append({"track": tid, "frame": frame_idx, "path": str(fpath)})
            print(f"[wedge] DISTRACTION ALERT: track T{tid} distracted "
                  f">={ALERT_FRAMES} frames lien tuc -> {fpath}", flush=True)

        writer.write(vis)

        if frame_idx % 100 == 0:
            print(f"[wedge] frame={frame_idx}/{total} faces={faces_seen} "
                  f"focused={counts['Focused']} distracted={counts['Distracted']} "
                  f"lookingup={counts['LookingUp']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    cap.release()
    writer.release()

    n = sum(counts.values())
    print(f"\n[wedge] done in {time.time()-t0:.0f}s → {video_out}", flush=True)
    if n:
        print(f"[wedge] face-frames={n}  Focused={counts['Focused']} "
              f"({counts['Focused']/n:.1%})  Distracted={counts['Distracted']} "
              f"({counts['Distracted']/n:.1%})  LookingUp={counts['LookingUp']} "
              f"({counts['LookingUp']/n:.1%})  central-cone hits={cone_hits}", flush=True)
    else:
        print("[wedge] no faces with gaze detected!", flush=True)

    # ── Attention verdict theo track ──────────────────────────────────
    # attention = true trừ khi track TỪNG distraction >= ALERT_FRAMES liên tục.
    alerted_tids = {a["track"] for a in alerts}
    all_tids = set(distract_run.keys())
    print(f"\n[wedge] === ATTENTION PER TRACK (alert >= {ALERT_FRAMES} "
          f"consecutive frames) ===", flush=True)
    for tid in sorted(all_tids):
        att = tid not in alerted_tids
        print(f"  T{tid}: attention={'true' if att else 'FALSE'}", flush=True)
    for a in alerts:
        print(f"  alert: T{a['track']} frame={a['frame']} -> {a['path']}", flush=True)


if __name__ == "__main__":
    main()
