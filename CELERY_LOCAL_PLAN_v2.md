# Plan v2 — Celery xử lý clip 5s local, ghi kết quả vào Redis QHH SIS

Phiên bản viết lại đúng cho repo `backup_qhh` (Classroom Manager: PySide6 +
`http.server` + Redis + ONNX in-process) và đúng schema Redis bên thứ 3 (xem
`redis-keys-third-party.md`). Bản v1 (`CELERY_LOCAL_PLAN.md`) được viết cho
một dự án khác (insulator + MinIO + Postgres + FastAPI) và không áp dụng được
— v2 này thay thế hoàn toàn v1.

---

## 1. Đầu vào, đầu ra, ranh giới hệ thống

**Đầu vào (do component khác lo, không nằm trong scope):**
- Một process khác cắt clip 5s từ camera RTSP và **ghi file mp4 vào folder
  local** `videos/`.
- Đặt tên file theo convention: `cam{cameraId}_class{classId}_{epochMs}.mp4`
  (vd `cam9b1c…__class3f7e…__1750939200123.mp4`). Đây là kênh duy nhất
  scheduler biết `cameraId` + `classId` của clip.
- Clip ghi xong rename từ `.part` → `.mp4` (atomic) để scheduler không quét
  trúng file đang ghi dở.

**Đầu ra (scope của plan này):**
- Worker chạy pipeline AI hiện có (YOLOv11 person → RetinaFace → ArcFace →
  Gaze) trên toàn bộ frame clip 5s, **aggregate per-student**, ghi 1 bản
  ghi kết quả vào Redis theo schema mới ở §3.
- Mỗi clip **độc lập**: không persist state distraction giữa các clip cùng
  `(cameraId, classId)`. Mỗi student trong clip chỉ có thống kê trong 5 giây
  đó.
- Cấu hình lớp/camera/desk/student được **ĐỌC** từ key chuẩn QHH
  `qhh:attendance:camera-class:{cameraId}:{classId}` đã có sẵn (do
  `qhh-server` ghi, plan này chỉ là consumer).

**Không nằm trong scope:**
- Cắt clip từ stream (component khác).
- Ghi `qhh:camera:*`, `qhh:user:*`, `qhh:attendance:camera-class:*` (do
  `qhh-server` ghi, plan này CHỈ ĐỌC).
- UI desktop (`main.py`) và endpoint streaming live (`web_server.py`
  `/api/ai/snapshot.jpg`, `/api/ai/start`) — vẫn giữ nguyên, không ảnh hưởng.

---

## 2. Stack & dependencies

Thêm vào `requirements.txt`:
```
celery>=5.3
```
(không thêm SQLAlchemy, FastAPI, MinIO, APScheduler, kombu khác — Redis đã
đảm nhiệm cả broker và result backend.)

**Không** thêm `onnxruntime-server` / Docker. ONNX đã chạy in-process trong
`workers/face_models.py` + `workers/gaze_estimator.py` ổn rồi.

**GPU:** 1 máy, 1 GPU, model bundle ~340 MB load 1 lần. Worker chạy:
```
celery -A celery_app worker -Q clip -n clip-worker@%h \
    --concurrency=1 --pool=solo --loglevel=info
```
- `--concurrency=1` + `--pool=solo`: bắt buộc khi dùng CUDA/ONNX-GPU. Prefork
  fork sau khi import torch/onnxruntime-gpu sẽ crash CUDA context.
- Nếu sau này có nhiều GPU: chạy nhiều process worker, mỗi process pin một
  `CUDA_VISIBLE_DEVICES`, cùng queue `clip`.

---

## 3. Redis — key mới do plan này tạo

Giữ nguyên prefix `qhh:` theo `redis-keys-third-party.md`. Tất cả key bên
dưới là **mới**, plan này tự ghi. Đặt dưới namespace `qhh:ai:clip:*` để
không đè key của `qhh-server`.

### 3.1. Dedup — set "đã đẩy vào queue"

| Key | Kiểu | Mô tả |
|-----|------|------|
| `qhh:ai:clip:pushed` | Set | Đường dẫn tuyệt đối clip đã từng được scheduler `apply_async`. SADD trả 1 = claim mới, 0 = đã có → bỏ qua. |

Đây là cơ chế dedup **duy nhất**. File trong `videos/` không bị move.

### 3.2. Trạng thái xử lý từng clip

| Key | Kiểu | Mô tả |
|-----|------|------|
| `qhh:ai:clip:status:{md5(path)}` | Hash | Trạng thái live của 1 clip |

Hash fields:
| Field | Giá trị |
|-------|---------|
| `path` | đường dẫn tuyệt đối clip |
| `cameraId` | GUID camera (parse từ filename) |
| `classId` | GUID lớp (parse từ filename) |
| `status` | `PENDING` \| `PROCESSING` \| `SUCCESS` \| `RETRY` \| `FAILED` |
| `taskId` | Celery task id |
| `attempts` | số lần retry |
| `updatedAt` | ISO 8601 UTC |
| `error` | message nếu fail (chỉ có khi status=RETRY/FAILED) |

TTL: `EXPIRE` 7 ngày sau lần ghi cuối, tránh phình Redis.

### 3.3. Kết quả AI — bản ghi cuối cùng cho mỗi clip

Đây là output chính bên thứ 3 sẽ tiêu thụ. Đặt **2 key song song** để vừa
tra theo clip vừa tra theo `(camera, class)`:

| Key | Kiểu | Mô tả |
|-----|------|------|
| `qhh:ai:clip:result:{md5(path)}` | String JSON | Kết quả tổng hợp 1 clip 5s |
| `qhh:ai:clip:index:{cameraId}:{classId}` | Sorted Set | `score = epochMs từ filename`, `member = md5(path)`. Cho phép `ZREVRANGEBYSCORE` để lấy N clip mới nhất của 1 (camera, class). |

JSON payload `qhh:ai:clip:result:{md5}`:
```json
{
  "clipPath": "/abs/videos/cam{cameraId}_class{classId}_{epochMs}.mp4",
  "cameraId": "guid",
  "classId": "guid",
  "clipStartedAt": "2026-06-26T07:00:05.123Z",
  "clipDurationSec": 5.0,
  "framesProcessed": 150,
  "fps": 30.0,
  "processingMs": 4321,
  "students": [
    {
      "studentId": "guid",
      "studentCode": "HS001",
      "fullName": "Nguyễn Văn A",
      "assignedDeskId": "desk-1",
      "assignedDeskLabel": "Bàn 1",
      "framesPresent": 142,
      "framesInAssignedDesk": 130,
      "presenceRatio": 0.947,
      "inAssignedDeskRatio": 0.867,
      "attendanceStatus": "PRESENT",
      "avgFaceMatchScore": 0.81,
      "avgGazeYawDeg": -3.2,
      "avgGazePitchDeg": 5.1,
      "distractedFrames": 18,
      "distractedRatio": 0.12,
      "distractionAlert": false
    }
  ],
  "unmatchedFaces": 3,
  "modelVersion": {
    "yolo": "yolo11n.onnx",
    "retinaface": "detectFace_model_op16.onnx",
    "arcface": "arcface_r100.onnx",
    "gaze": "resnet50_gaze.onnx"
  },
  "generatedAt": "2026-06-26T07:00:10.456Z"
}
```

Quy ước aggregate:

- `framesPresent` = số frame student được detect & match (cosine ≥
  `FACE_MATCH_THRESHOLD` hiện có), bất kể nằm trong bàn assigned hay không.
- `framesInAssignedDesk` = trong số `framesPresent`, bao nhiêu frame có
  `person_records[i].owner_desk == assignedDeskNum` (lấy thẳng từ output
  `_detect()` đã có trong `workers/camera_worker.py` — xem §11.6 dưới).
- `presenceRatio` = `framesPresent / framesProcessed`.
- `inAssignedDeskRatio` = `framesInAssignedDesk / framesPresent` (= 0 nếu
  `framesPresent == 0`).
- `attendanceStatus` (per-clip, stateless):
  - **`PRESENT`** — `inAssignedDeskRatio ≥ local.assigned_seat_ratio`
    (HS ngồi đúng bàn đa số thời gian).
  - **`WRONG_SEAT`** — `presenceRatio ≥ local.presence_ratio` nhưng
    `inAssignedDeskRatio < local.assigned_seat_ratio` (HS xuất hiện đủ
    nhưng đa số ở bàn khác → coi là vắng do ngồi sai bàn, theo yêu cầu).
  - **`ABSENT`** — `presenceRatio < local.presence_ratio` (không xuất
    hiện đủ trong clip).
- `distractedFrames` = số frame `|yaw| > local.yaw_thresh_deg` hoặc
  `|pitch| > local.pitch_thresh_deg`.
- `distractionAlert` = `distractedRatio ≥ local.distracted_ratio_alert`
  (mặc định 0.5). Per-clip stateless, không hysteresis xuyên clip.
- `unmatchedFaces` = số face detect được nhưng không match student nào
  của lớp — debug "ai đó lạ vào lớp".

**Không có student trong `qhh:attendance:camera-class:*` `regions[]` nào**:
plan v2 không emit row cho HS không gán bàn (consumer dùng
`unmatchedFaces` để biết có người lạ). Nếu sau này cần track HS chưa gán
bàn, thêm `attendanceStatus="UNASSIGNED"` — không nằm trong scope hiện
tại.

### 3.4. Xoá key Redis cũ (theo yêu cầu user)

User yêu cầu: "xóa các key của redis cũ đi". Plan này KHÔNG tự xoá
`qhh:user:*` / `qhh:camera:*` / `qhh:attendance:camera-class:*` (do
`qhh-server` quản lý, chúng ta consumer). "Cũ" ở đây hiểu là **các key do
chính repo `backup_qhh` ghi**:

| Key cũ (do `db/redis_client.py` ghi) | Hành động |
|---|---|
| `student:{id}`, `students` (set) — không có prefix `qhh:` | **Xoá**. Thay bằng đọc trực tiếp `students[]` trong JSON của `qhh:attendance:camera-class:{cameraId}:{classId}` (đã có `id`, `studentCode`, `fullName`, `avatarUrl`). |
| `classroom:{id}`, `desk:{class_id}:{desk_num}` (legacy seat geometry) | **Xoá**. Geometry desk lấy từ `regions[]` (`x,y,w,h` chuẩn hoá 0..1) trong cùng JSON camera-class. |

Việc xoá thực hiện qua một script migration `scripts/drop_legacy_keys.py`
chạy 1 lần: `SCAN MATCH "student:*"` + `SCAN MATCH "classroom:*"` + `SCAN
MATCH "desk:*"` rồi `DEL`. Sau khi xoá, các hàm `db.list_students()`,
`db.list_classrooms()`, `db.list_seats()` được viết lại để đọc từ
`qhh:attendance:camera-class:*`.

> ⚠ **Lưu ý quan trọng:** Việc xoá key cũ ảnh hưởng cả `main.py` desktop UI
> và `web_server.py` CRUD endpoint. Trước khi xoá phải xác nhận: dữ liệu
> nghiệp vụ thực (student/desk/zone) bây giờ **chỉ** đến từ `qhh-server`
> qua key `qhh:attendance:camera-class:*`, không còn nhập tay trong repo
> này nữa. Nếu vẫn còn nhập tay → giữ key cũ làm "override layer", chỉ
> migrate khi chắc chắn.

---

## 4. Layout folder

```
backup_qhh/
├── videos/                          # clip 5s do component khác ghi (atomic .part→.mp4)
│   └── cam{cameraId}_class{classId}_{epochMs}.mp4
├── detection/                       # (tuỳ chọn) snapshot frame có annotation
│   └── {md5(path)}.jpg
├── celery_app.py                    # MỚI — Celery instance
├── local_storage.py                 # MỚI — scan folder + dedup + status Redis
├── tasks.py                         # MỚI — process_clip_task
├── clip_inference.py                # MỚI — wrap _shared_models cho 1 video file
├── scheduler.py                     # MỚI — vòng quét folder (thread hoặc systemd timer)
├── scripts/
│   └── drop_legacy_keys.py          # MỚI — migration 1 lần
├── workers/                         # giữ nguyên, tái dùng _shared_models()
└── ...
```

Không cần `done/`, `error/`, `processing/` — Redis là nguồn chân lý.

---

## 5. Module mới — chi tiết

### 5.1. `celery_app.py`

```python
from celery import Celery
from config_loader import env_or_config

_host = str(env_or_config("REDIS_HOST", "redis", "host", "127.0.0.1"))
_port = int(env_or_config("REDIS_PORT", "redis", "port", 6379))
_db   = int(env_or_config("REDIS_DB",   "redis", "db",   0))
_pw   = env_or_config("REDIS_PASSWORD", "redis", "password", "") or ""

_auth = f":{_pw}@" if _pw else ""
BROKER_URL = f"redis://{_auth}{_host}:{_port}/{_db}"

celery_app = Celery("qhh_ai", broker=BROKER_URL, backend=BROKER_URL,
                    include=["tasks"])
celery_app.conf.update(
    task_default_queue="clip",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    result_expires=7 * 24 * 3600,
)
```

### 5.2. `local_storage.py`

```python
import hashlib, re, time
from datetime import datetime, timezone
from pathlib import Path
from db import redis_client as db

VIDEO_DIR     = Path(env_or_config("VIDEO_DIR", "local", "video_dir", "videos")).resolve()
VIDEO_EXTS    = {".mp4", ".mov", ".mkv"}
MTIME_MIN_AGE = 3.0  # giây — file phải tĩnh >=3s mới coi là ghi xong

PUSHED_SET    = "qhh:ai:clip:pushed"
STATUS_KEY    = "qhh:ai:clip:status:{md5}"
STATUS_TTL    = 7 * 24 * 3600

# filename: cam{cameraId}_class{classId}_{epochMs}.mp4
_NAME_RE = re.compile(
    r"^cam(?P<cam>[0-9a-fA-F-]{36})_class(?P<cls>[0-9a-fA-F-]{36})_(?P<ts>\d+)\.[^.]+$"
)

def _md5(p: str) -> str: return hashlib.md5(p.encode()).hexdigest()
def _now() -> str: return datetime.now(timezone.utc).isoformat()

def parse_clip_name(path: Path):
    m = _NAME_RE.match(path.name)
    if not m: return None
    return {"cameraId": m["cam"], "classId": m["cls"], "epochMs": int(m["ts"])}

def scan_new_clips():
    r = db.get_client()
    out = []
    now = time.time()
    for p in VIDEO_DIR.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS: continue
        try:
            if now - p.stat().st_mtime < MTIME_MIN_AGE: continue
        except FileNotFoundError: continue
        ap = str(p.resolve())
        if r.sismember(PUSHED_SET, ap): continue
        meta = parse_clip_name(p)
        if not meta:
            # tên file sai convention → bỏ qua + log (không claim, không xử lý)
            continue
        out.append((ap, meta))
    out.sort(key=lambda x: x[1]["epochMs"])
    return out

def mark_pushed(path: str) -> bool:
    return db.get_client().sadd(PUSHED_SET, path) == 1

def set_status(path: str, status: str, **fields):
    r = db.get_client()
    key = STATUS_KEY.format(md5=_md5(path))
    data = {"path": path, "status": status, "updatedAt": _now(), **fields}
    r.hset(key, mapping={k: str(v) for k, v in data.items()})
    r.expire(key, STATUS_TTL)

def get_status(path: str) -> dict:
    return db.get_client().hgetall(STATUS_KEY.format(md5=_md5(path)))
```

### 5.3. `clip_inference.py` — adapter pipeline hiện có cho 1 file

Tái dùng `AIDetectionWorker._shared_models()` đã có (cùng cách
`WebDetectionEngine` trong `web_server.py` đang dùng):

```python
import cv2, json, time
from pathlib import Path
from workers.camera_worker import AIDetectionWorker
from db import redis_client as db

CAM_CLASS_KEY = "qhh:attendance:camera-class:{cam}:{cls}"

def load_camera_class(cam_id: str, cls_id: str) -> dict:
    raw = db.get_client().get(CAM_CLASS_KEY.format(cam=cam_id, cls=cls_id))
    if not raw:
        raise RuntimeError(f"missing camera-class config {cam_id}/{cls_id}")
    return json.loads(raw)

def run_clip(video_path: str, cam_id: str, cls_id: str, *, cfg: dict) -> dict:
    yolo, retina, arcface, gaze = AIDetectionWorker._shared_models()
    classroom = load_camera_class(cam_id, cls_id)
    # build face gallery từ classroom["students"] (chỉ HS của lớp này — nhỏ
    # và nhanh; arcface gallery cache đã có sẵn trong workers/face_models.py)
    gallery = _build_gallery(classroom["students"])
    regions = classroom["regions"]

    cap = cv2.VideoCapture(video_path)
    fps_decl = cap.get(cv2.CAP_PROP_FPS) or 30.0
    per_student = {}     # studentId -> aggregator
    unmatched = 0
    frames = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames += 1
        det = _run_one_frame(frame, yolo, retina, arcface, gaze,
                             gallery=gallery, regions=regions, cfg=cfg)
        for s in det.students:
            agg = per_student.setdefault(s.id, _new_agg(s))
            _update_agg(agg, s)
        unmatched += det.unmatched
    cap.release()
    return _finalize(per_student, frames, fps_decl, unmatched,
                     started_at=_parse_epoch(video_path),
                     processing_ms=int((time.time()-t0)*1000),
                     video_path=video_path, cam=cam_id, cls=cls_id, cfg=cfg)
```

`_run_one_frame` lấy trực tiếp logic từ `AIDetectionWorker._run_inference`
hiện có — tách thành hàm thuần (không phụ thuộc QThread/`Signal`). Nếu
refactor lớn, để vòng 1 dùng tạm `WebDetectionEngine` (đã Qt-free).

### 5.4. `tasks.py`

```python
from celery import shared_task
from celery_app import celery_app
import local_storage as ls
import clip_inference as ci
import json
from db import redis_client as db
from pathlib import Path

RESULT_KEY = "qhh:ai:clip:result:{md5}"
INDEX_KEY  = "qhh:ai:clip:index:{cam}:{cls}"
RESULT_TTL = 30 * 24 * 3600

@celery_app.task(bind=True, name="process_clip_task",
                 max_retries=3, default_retry_delay=10, acks_late=True)
def process_clip_task(self, video_path: str, cameraId: str, classId: str):
    ls.set_status(video_path, "PROCESSING",
                  taskId=self.request.id, cameraId=cameraId, classId=classId,
                  attempts=self.request.retries)
    try:
        result = ci.run_clip(video_path, cameraId, classId, cfg=_load_cfg())
        r = db.get_client()
        md5 = ls._md5(video_path)
        r.set(RESULT_KEY.format(md5=md5), json.dumps(result), ex=RESULT_TTL)
        meta = ls.parse_clip_name(Path(video_path))
        r.zadd(INDEX_KEY.format(cam=cameraId, cls=classId),
               {md5: float(meta["epochMs"])})
        ls.set_status(video_path, "SUCCESS",
                      taskId=self.request.id, cameraId=cameraId, classId=classId)
        return {"md5": md5, "students": len(result["students"])}
    except Exception as exc:
        if self.request.retries < self.max_retries:
            ls.set_status(video_path, "RETRY", error=str(exc),
                          attempts=self.request.retries + 1)
            raise self.retry(exc=exc)
        ls.set_status(video_path, "FAILED", error=str(exc),
                      attempts=self.request.retries + 1)
        raise
```

### 5.5. `scheduler.py`

Vòng quét đơn giản, chạy bằng `python -m scheduler` hoặc systemd timer:

```python
import time, traceback
import local_storage as ls
from tasks import process_clip_task

SCAN_INTERVAL = 5.0  # giây

def tick():
    for path, meta in ls.scan_new_clips():
        if not ls.mark_pushed(path):  # SADD atomic, ai claim trước nấy thắng
            continue
        ls.set_status(path, "PENDING", cameraId=meta["cameraId"],
                      classId=meta["classId"])
        process_clip_task.apply_async(
            args=[path, meta["cameraId"], meta["classId"]], queue="clip")

if __name__ == "__main__":
    while True:
        try: tick()
        except Exception: traceback.print_exc()
        time.sleep(SCAN_INTERVAL)
```

Không cần APScheduler — `time.sleep` đủ ở scale này. Single instance là
đủ; muốn HA thì nhiều scheduler chạy song song vẫn an toàn nhờ `SADD`
atomic.

### 5.6. (Tuỳ chọn) Endpoint xem trạng thái — `web_server.py`

Thêm route trong handler hiện có (không thêm framework mới):

```python
# trong handle()
if path == "/api/clip/status":
    qs = parse_qs(parsed.query)
    p = qs.get("path", [""])[0]
    return self._json(200, ls.get_status(p))
if path == "/api/clip/result":
    qs = parse_qs(parsed.query)
    md5 = qs.get("md5", [""])[0]
    raw = db.get_client().get(f"qhh:ai:clip:result:{md5}")
    return self._json(200, json.loads(raw) if raw else {})
if path == "/api/clip/latest":
    qs = parse_qs(parsed.query)
    cam, cls, n = qs["cameraId"][0], qs["classId"][0], int(qs.get("n", ["5"])[0])
    md5s = db.get_client().zrevrange(
        f"qhh:ai:clip:index:{cam}:{cls}", 0, n-1)
    items = [json.loads(db.get_client().get(f"qhh:ai:clip:result:{m}") or "{}")
             for m in md5s]
    return self._json(200, items)
```

---

## 6. Config

`config.json` (thêm block `local`, giữ nguyên `redis`/`middleware`/`web`):
```json
{
  "redis":      { "host": "192.168.6.16", "port": 6378, "db": 0,
                  "password": "qhh_redis_change_me", "prefix": "qhh" },
  "middleware": { "...": "giữ nguyên" },
  "web":        { "host": "0.0.0.0", "port": 8090 },
  "local": {
    "video_dir":         "videos",
    "yaw_thresh_deg":    25,
    "pitch_thresh_deg":  20,
    "distracted_ratio_alert": 0.5
  }
}
```

`config_loader.env_or_config("YAW_THRESH_DEG","local","yaw_thresh_deg",25)`
— giữ pattern hiện có.

---

## 7. Luồng end-to-end

```
[recorder khác]               [scheduler.py]                  [Celery worker]
    │                              │                                │
    ├─ ghi cam{..}_class{..}_..mp4 │                                │
    │  (.part → .mp4 atomic)        │                                │
    │                              │ rglob videos/                  │
    │                              │ mtime ≥3s                      │
    │                              │ parse filename → cam/cls/ts    │
    │                              │ SADD qhh:ai:clip:pushed (1=new)│
    │                              │ HSET status=PENDING            │
    │                              │ apply_async ─────────────────► │
    │                              │                                │ HSET status=PROCESSING
    │                              │                                │ load qhh:attendance:camera-class:{cam}:{cls}
    │                              │                                │ build gallery từ students[]
    │                              │                                │ cv2.VideoCapture(path) — loop frame
    │                              │                                │   YOLO → RetinaFace → ArcFace → Gaze
    │                              │                                │ aggregate per-student (5s, stateless)
    │                              │                                │ SET qhh:ai:clip:result:{md5} (JSON)
    │                              │                                │ ZADD qhh:ai:clip:index:{cam}:{cls} {md5}=epochMs
    │                              │                                │ HSET status=SUCCESS
```

File `.mp4` không bị move/xoá. Dọn file là việc của recorder (vd xoá clip
> N ngày).

---

## 8. Test plan

1. **Smoke local:**
   - Tắt scheduler. Copy 1 file mẫu vào `videos/` với tên đúng convention
     (cameraId + classId thực có trong key `qhh:attendance:camera-class:*`).
   - Chạy worker `celery -A celery_app worker -Q clip -P solo -c 1`.
   - Gọi tay: `process_clip_task.apply_async(args=[path, cam, cls])`.
   - Kiểm tra: `qhh:ai:clip:status:{md5}=SUCCESS`,
     `qhh:ai:clip:result:{md5}` parse được JSON, schema đúng §3.3.

2. **Dedup:**
   - Bật scheduler. Cùng file đã xử lý xong nằm im trong `videos/` →
     `SADD` trả 0 → không apply_async lại.

3. **File đang ghi dở:**
   - Touch file `.part`, không có ext mp4 → bỏ qua. Rename `.part → .mp4`
     khi xong, sleep > 3s → tick kế tiếp pick lên.

4. **Filename sai convention:**
   - `random.mp4` không match regex → log warn, không claim, không xử lý.

5. **Camera-class chưa có trong Redis:**
   - Worker raise → retry 3 lần countdown 10s → `FAILED` + error message.

6. **Multi-clip cùng `(cam, cls)`:**
   - Đẩy 3 clip, thời gian ts tăng dần. `ZREVRANGE qhh:ai:clip:index:...
     0 2` trả về theo thứ tự mới nhất.

7. **Stateless distraction:**
   - Hai clip liên tiếp, student giống nhau, clip 1 nhìn lệch full 5s,
     clip 2 nhìn thẳng. Kết quả clip 2 phải có `distractionAlert=false`
     bất kể clip 1. Khẳng định không leak state.

---

## 9. Checklist triển khai

- [ ] `pip install celery>=5.3` (cài thêm dependency duy nhất).
- [ ] Tạo folder `videos/`. Component recorder ghi clip 5s đúng tên
      `cam{cameraId}_class{classId}_{epochMs}.mp4`, atomic `.part → .mp4`.
- [ ] Thêm block `local` vào `config.json` (§6).
- [ ] Viết `celery_app.py`, `local_storage.py`, `clip_inference.py`,
      `tasks.py`, `scheduler.py` (§5).
- [ ] Tách hàm pure `_run_inference_on_frame(frame, models, gallery, regions,
      cfg)` từ `workers/camera_worker.py` hoặc reuse `WebDetectionEngine`.
- [ ] Chạy `scripts/drop_legacy_keys.py` SAU KHI xác nhận đã đọc được dữ
      liệu từ `qhh:attendance:camera-class:*` đầy đủ (§3.4 — cẩn trọng).
- [ ] Sửa `db/redis_client.py`: `list_students(class_id)` / `list_seats`
      đọc từ `qhh:attendance:camera-class:*` thay vì `student:*` /
      `desk:*` cũ.
- [ ] Thêm 3 endpoint `/api/clip/status|result|latest` vào `web_server.py`.
- [ ] Chạy worker: `celery -A celery_app worker -Q clip -P solo -c 1 -l info`.
- [ ] Chạy scheduler: `python -m scheduler` (hoặc systemd unit).
- [ ] Thả 1 clip test → verify SUCCESS + JSON đúng schema §3.3.

---

## 10. So sánh nhanh v1 (CELERY_LOCAL_PLAN.md) ↔ v2

| Khía cạnh | v1 (giả định insulator project) | v2 (đúng repo này) |
|---|---|---|
| Mục tiêu pipeline | Detect insulator + vẽ bbox | Aggregate per-student attendance + gaze trong clip 5s |
| Storage cấu hình | Postgres `detection_models`/`detection_tasks` | Redis `qhh:attendance:camera-class:*` (đã có) |
| Output | File ảnh `detection/` + Postgres `InsulatorModel` | Redis `qhh:ai:clip:result:{md5}` + index sorted set |
| Web framework | FastAPI mới | Tái dùng `http.server` trong `web_server.py` |
| Scheduler | APScheduler cron | `while True: sleep(5)` |
| Redis prefix | `detection:*` (sai chuẩn QHH) | `qhh:ai:clip:*` (đúng `redis-keys-third-party.md`) |
| ONNX | "Chuyển sang ONNX" (đã sẵn) | Giữ nguyên `_shared_models()` |
| Docker | `kibaes/onnxruntime-server` + compose | Không thêm container |
| Dedup | `detection:pushed` set | `qhh:ai:clip:pushed` set (cùng cơ chế, đổi prefix) |
| State distraction | Không nói | **Stateless per-clip** (yêu cầu user) |
| Model registry | DB `detection_models` + Ultralytics `.onnx` | Hardcode 4 file `.onnx`/`.pt` trong `weights/` (model cố định, không update runtime — yêu cầu user) |

---

## 11. Batch & scaling (xử lý nhiều clip / nhiều camera)

Mục tiêu: 1 GPU ≥12GB, ưu tiên cân bằng latency/throughput. Kết hợp 2 cơ chế:

### 11.1. Fleet — 4 worker process cùng GPU

```bash
# Mỗi process: --pool=solo --concurrency=1 (bắt buộc cho onnxruntime-gpu)
for i in 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=0 \
  celery -A celery_app worker -Q clip -P solo -c 1 \
    -n clip-worker-$i@%h --loglevel=info &
done
```

- Mỗi process load 1 bộ model (~1.8 GB VRAM) → 4 process ≈ 7.2 GB / 12 GB.
- CUDA scheduler tuần tự hoá kernel → throughput thực ~1.5–1.8×, đủ cho
  burst nhiều camera.
- Celery tự load-balance queue `clip` qua 4 worker.

### 11.2. Batch FRAME trong từng clip (stage YOLO + RetinaFace + ArcFace)

Chia clip thành chunk `BATCH_SIZE` (mặc định 8) frame, mỗi stage gọi 1
lần ONNX:

| Stage | API mới | Lợi |
|---|---|---|
| **YOLO person** | `yolo([f1..f8], imgsz=…)` (Ultralytics native) | ~2× |
| **RetinaFace** | `RetinaFaceDetector.detect_all_batch(images)` | ~2× |
| **ArcFace** | `ArcFaceExtractor.embed_batch(aligned_faces)` (1 lần `session.run` với `[N,3,112,112]`) | ~3× (model nặng nhất) |
| **Gaze** | `GazeEstimator.estimate_batch(crops)` — stack `[N,3,448,448]`, 1 `session.run` | ~3–5× (model ResNet-50 nặng) |

> **Gaze dynamic batch ĐÃ verify (2026-06-26):** `weights/resnet50_gaze.onnx`
> đã re-export với dim 0 = `'batch_size'` symbolic. Đã test batch 1/4/8/16
> chạy tốt; cùng 1 ảnh ở batch=1 vs trong batch=4 cho output **bit-identical**
> (`max |diff| = 0.0` cả 2 head yaw/pitch). An toàn tuyệt đối về độ chính xác.

Code change cần thiết (KHÔNG phá API hiện có dùng cho `web_server.py`/`main.py`):
- `workers/face_models.py`:
  - Thêm `RetinaFaceDetector.detect_all_batch(images: list[np.ndarray]) -> list[list[Face]]`
  - Thêm `ArcFaceExtractor.embed_batch(aligned_faces: list[np.ndarray]) -> np.ndarray`
- `workers/gaze_estimator.py`: tách `_preprocess_one(image)->[3,H,W]`,
  viết lại `estimate_batch(crops)` thành batch thật (stack + 1 lần
  `session.run`). `estimate(face)` giữ làm wrapper `estimate_batch([face])[0]`
  để không phá caller real-time (`camera_worker.py`, `web_server.py`).
- `clip_inference.py`: chia clip thành chunk `BATCH_SIZE`, gọi batch API
  theo thứ tự YOLO → RetinaFace → ArcFace → Gaze (gaze vẫn loop trong
  hàm cũ).

### 11.3. Backpressure cho burst

Khi 10 camera đẩy clip đồng thời → queue dài → latency tăng. 2 cơ chế:

- **Drop policy (tuỳ chọn, qua config):** nếu clip nằm queue > `MAX_QUEUE_AGE_SEC`
  (mặc định 60s) hoặc backlog > `MAX_BACKLOG_PER_CAM_CLASS` (mặc định 5)
  cho cùng `(cameraId, classId)` → scheduler skip clip cũ, set
  `status=DROPPED`, log số clip bị drop.
- **Monitor key:** `qhh:ai:clip:queue:depth` (String, TTL 60s) — scheduler
  cập nhật mỗi tick = số task PENDING trong Redis. Dashboard/oncall lấy
  từ đây để cảnh báo.

### 11.4. Config bổ sung trong `config.json`

```json
"local": {
  "video_dir": "videos",
  "yaw_thresh_deg": 25,
  "pitch_thresh_deg": 20,
  "distracted_ratio_alert": 0.5,
  "presence_ratio": 0.6,
  "assigned_seat_ratio": 0.5,
  "detection_mode": "centerpoint",
  "batch_size": 8,
  "worker_count": 4,
  "max_queue_age_sec": 60,
  "max_backlog_per_cam_class": 5
}
```

- `presence_ratio` — ngưỡng `framesPresent / framesProcessed` để không
  bị coi `ABSENT`.
- `assigned_seat_ratio` — ngưỡng `framesInAssignedDesk / framesPresent`
  để được coi `PRESENT` thay vì `WRONG_SEAT`.
- `detection_mode` — chuyển cho `_detect()`: `"centerpoint"` (bbox center
  → polygon ảnh, nhanh, không cần calibration) hoặc `"perspective"`
  (foot → ground-plane cm, chính xác hơn khi đã có homography).

### 11.5. Sizing tham khảo (ước lượng, cần benchmark thật)

| Cấu hình | Throughput/worker | Tổng (4 worker) | Đáp ứng |
|---|---|---|---|
| Plan v2 nguyên bản (no batch) | ~1 clip / 4s | ~1 clip/s | 1–2 camera |
| + Batch YOLO/RetinaFace/ArcFace/Gaze (full batch, all stages dynamic) | ~1 clip / 1.2s | ~3 clip/s | 10+ camera |

Số liệu trên là ước lượng; benchmark thật trên GPU đích để chốt
`BATCH_SIZE` và `worker_count`.

### 11.6. Tái dùng logic "HS có trong area bàn không" — không viết lại

`workers/camera_worker.py` đã có đầy đủ logic point-in-polygon /
bbox-polygon overlap / IoU + tie-break cho khu vực bàn:

- `_zone_corners_and_aabb(zone, w, h)` — line 1501: convert
  `regions[].{x,y,w,h}` (0..1) → 4 corner pixel.
- `_polygon_membership_score(poly, point)` — line 1000: dùng
  `cv2.pointPolygonTest`, trả signed depth normalize theo diện tích.
- `_bbox_polygon_overlap_ratio` / `_bbox_polygon_iou` — line 1020/1041:
  metric cho mode `centerpoint`.
- `_detect()` line 472–617: chạy YOLO → assign mỗi person bbox vào
  `owner_desk` duy nhất (tie-break theo IoU → depth → overlap → area →
  vertical position). Output `person_records[i]["owner_desk"]` là
  `desk_num` chứa HS đó, hoặc `None`.

`clip_inference.py` chỉ cần:

1. Convert `regions[]` từ `qhh:attendance:camera-class:*` JSON sang
   schema `seat` mà `_detect()` mong đợi (cùng cấu trúc
   `{desk_num, zone, slots}`). Mapping `regions[].id` ↔ `desk_num` lưu
   để aggregate.
2. Build `seat_lookup: studentId → (desk_num, slot_num)` từ
   `regions[].studentIds`.
3. Mỗi frame: gọi `_detect()`, lấy `person_records` với face_info +
   owner_desk → cập nhật bucket per-student:
   ```
   bucket[studentId].framesPresent     += 1
   if owner_desk == assigned_desk_num:
       bucket[studentId].framesInAssignedDesk += 1
   ```
4. Cuối clip: tính `inAssignedDeskRatio`, quyết định
   `attendanceStatus` theo §3.3.

→ Không viết lại logic geometry; chỉ adapter input (regions JSON →
seats internal) + aggregator output.

