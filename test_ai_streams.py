"""Test từng luồng model AI + XUẤT VIDEO VISUALIZE, region = full frame.

Chạy: docker exec backup_qhh-worker-1 python3 /app/test_ai_streams.py \
        /app/videos/video_lop_qhh.mp4 <max_frames> <stride> <out.mp4>

Vẽ overlay lên MỌI frame xử lý:
  - YOLO person box (xanh lá) + SORT track id
  - RetinaFace face box (vàng) + ArcFace (dim/norm ok) + Gaze arrow + yaw/pitch
Ghi ra video output để xem trực quan mỗi luồng.
"""
import sys, time, math
import numpy as np
import cv2

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/app/videos/video_lop_qhh.mp4"
MAX_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 120
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUT = sys.argv[4] if len(sys.argv) > 4 else "/app/result_test/video_lop_qhh_viz.mp4"

sys.path.insert(0, "/app")
from workers.camera_worker import AIDetectionWorker

print("== load models ==", flush=True)
yolo, retina, arc, gaze = AIDetectionWorker._shared_models()

_sort_path = "/app/sort"
if _sort_path not in sys.path:
    sys.path.insert(0, _sort_path)
from sort import Sort
tracker = Sort(max_age=60, min_hits=3, iou_threshold=0.3)

cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("ERROR: cannot open video", flush=True); sys.exit(1)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"== video {W}x{H} {FPS:.1f}fps total={TOTAL} → out={OUT} ==", flush=True)

# Output FPS: giữ tốc độ thật = FPS/STRIDE (mỗi STRIDE frame lấy 1)
out_fps = max(1.0, FPS / STRIDE)
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W, H))

YAW_TH, PITCH_TH = 60.0, 9999.0
agg = {"yolo": 0, "retina": 0, "arc": 0, "gaze": 0, "distract": 0}
track_ids_seen = set()
t0 = time.time()
fidx = 0
processed = 0

def put(img, txt, org, color, scale=0.6, th=2):
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)

while processed < MAX_FRAMES:
    ok, frame = cap.read()
    if not ok:
        break
    fidx += 1
    if fidx % STRIDE != 0:
        continue
    processed += 1
    vis = frame.copy()

    # ── [YOLO] person + [SORT] track ──
    try:
        res = yolo.predict(frame, classes=[0], verbose=False, device=0)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0, 4))
        confs = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.empty((0,))
    except Exception as e:
        boxes, confs = np.empty((0, 4)), np.empty((0,))
    n_person = len(boxes)
    agg["yolo"] += n_person
    dets = (np.column_stack([boxes, confs]) if len(confs) == n_person
            else np.column_stack([boxes, np.ones(n_person)])) if n_person else np.empty((0, 5))
    tracks = tracker.update(dets)
    for tk in tracks:
        x1, y1, x2, y2, tid = [int(v) for v in tk]
        track_ids_seen.add(tid)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (16, 200, 16), 3)
        put(vis, f"person #{tid}", (x1, max(y1 - 8, 20)), (16, 200, 16), 0.7, 2)

    # ── [RETINA] faces — detect trong TỪNG person-crop (mặt to tương đối,
    #    bắt tốt hơn full-frame 640 khi lớp quay xa/4K). Map bbox về toạ độ gốc.
    faces = []
    for tk in tracks:
        px1, py1, px2, py2 = [max(0, int(v)) for v in tk[:4]]
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            continue
        dets, _ = retina.detect(crop)  # mặt lớn nhất trong crop
        for d in dets:
            if d.get("aligned_face") is None:
                continue
            fx1, fy1, fx2, fy2 = d["loc"]
            d = dict(d, loc=[fx1 + px1, fy1 + py1, fx2 + px1, fy2 + py1])
            faces.append(d)
    # Fallback: nếu YOLO không ra person nào, thử full-frame.
    if not faces:
        faces = [f for f in retina.detect_all(frame) if f.get("aligned_face") is not None]
    agg["retina"] += len(faces)
    aligned = [f["aligned_face"] for f in faces if f.get("aligned_face") is not None]

    # ── [ARCFACE] embedding ──
    embs = arc.extract(aligned) if aligned else None
    if embs is not None and len(embs):
        agg["arc"] += len(embs)

    # ── [GAZE] yaw/pitch ──
    pairs = gaze.estimate_batch(aligned) if (gaze and aligned) else []
    if pairs:
        agg["gaze"] += len(pairs)

    # ── vẽ face box + gaze arrow ──
    ai = 0
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f["loc"]]
        color = (40, 200, 220)  # vàng: face
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        norm_txt = ""
        if embs is not None and ai < len(embs):
            norm_txt = f" e{embs.shape[1]}"
        if f.get("aligned_face") is not None and ai < len(pairs):
            yaw_r, pitch_r = pairs[ai]
            yaw_d, pitch_d = math.degrees(yaw_r), math.degrees(pitch_r)
            distract = abs(yaw_d) > YAW_TH or abs(pitch_d) > PITCH_TH
            if distract:
                agg["distract"] += 1
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            length = (x2 - x1)
            dx = int(-length * math.sin(yaw_r) * math.cos(pitch_r))
            dy = int(-length * math.sin(pitch_r))
            acolor = (0, 0, 255) if distract else (0, 220, 0)
            cv2.circle(vis, (cx, cy), 4, acolor, -1)
            cv2.arrowedLine(vis, (cx, cy), (cx + dx, cy + dy), acolor, 3, cv2.LINE_AA, tipLength=0.3)
            put(vis, f"yaw{yaw_d:+.0f} pit{pitch_d:+.0f}{norm_txt}",
                (x1, y2 + 22), acolor, 0.55, 2)
            if f.get("aligned_face") is not None and ai < len(aligned):
                pass
        ai += 1

    # HUD
    put(vis, f"frame {fidx}  YOLO:{n_person}  FACE:{len(faces)}  "
             f"ARC:{len(embs) if embs is not None else 0}  GAZE:{len(pairs)}  "
             f"TRACKS:{len(tracks)}", (20, 40), (255, 255, 255), 0.8, 2)
    writer.write(vis)
    if processed % 10 == 0:
        print(f"  ...{processed} frames written", flush=True)

cap.release()
writer.release()
dt = time.time() - t0
print(f"\n== SUMMARY ({processed} frames, {dt:.1f}s) ==", flush=True)
print(f"  [YOLO]    persons total : {agg['yolo']}", flush=True)
print(f"  [RETINA]  faces total   : {agg['retina']}", flush=True)
print(f"  [ARCFACE] embeddings    : {agg['arc']}", flush=True)
print(f"  [GAZE]    estimates     : {agg['gaze']}  (distract={agg['distract']})", flush=True)
print(f"  [SORT]    unique tracks : {len(track_ids_seen)}", flush=True)
print(f"== OUTPUT VIDEO: {OUT} ==", flush=True)
