# Plan: Celery xử lý video từ Local Folder (không dùng MinIO)

Thiết kế lại pipeline detection để **quét video trong một folder local**, dùng **Redis lưu
đường dẫn từng video** (vừa làm hàng đợi, vừa chống đẩy trùng), Celery worker chạy inference
bằng **ONNX Runtime (GPU)**. Bỏ hoàn toàn MinIO.

---

## 1. Mục tiêu & Khác biệt so với hệ thống hiện tại

| Khía cạnh | Hiện tại (MinIO) | Thiết kế mới (Local) |
|-----------|------------------|----------------------|
| Nguồn video | Bucket MinIO (`wayline/`) | Folder local (vd `./videos/`) |
| Liệt kê file | `client.list_objects()` | `os.scandir()` / `Path.glob()` |
| Chống xử lý trùng | MinIO metadata tag | **1 Redis set** chứa path đã đẩy |
| Folder local | — | **Phẳng**, không cần done/error/processing |
| Trạng thái task | DB `detection_tasks` + Redis | **Chỉ Redis** (key trạng thái) |
| Lấy video cho worker | presigned URL | đường dẫn file local trực tiếp |
| Inference runtime | YOLO `.pt` (PyTorch) | **ONNX Runtime GPU** (`.onnx`) |
| Redis | container riêng trong compose | **Đã có sẵn** (external, đang chạy) |

**Nguyên tắc giữ lại:** Celery app, task `process_media_task`, `model_registry`, `TaskStatus`
enum — chỉ thay lớp **storage** và **scheduler dispatch**, và nguồn model sang ONNX.

---

## 2. Quy tắc cốt lõi về Redis (theo yêu cầu)

> **Video nào đã đẩy vào Redis thì không đẩy lại nữa.**
> Folder local **không cần** lưu trạng thái (không move file, không subfolder done/error).
> Mọi trạng thái **chỉ nằm trong key Redis**.

Dùng đúng **2 nhóm key**:

1. **Set "đã đẩy" (dedup)** — `detection:pushed`
   - Mỗi khi scheduler đẩy 1 video vào hàng đợi → `SADD detection:pushed <path>`.
   - Lần quét sau, nếu path đã ở trong set này → **bỏ qua**, không đẩy lại.
   - Đây là cơ chế chống trùng duy nhất. File trong folder cứ để nguyên.

2. **Key trạng thái mỗi video** — `detection:status:<key>` (Redis Hash)
   ```
   path        = /abs/path/video.mp4
   status      = PENDING|PROCESSING|SUCCESS|RETRY|FAILED_RETRY
   task_id     = <celery_task_id>
   updated_at  = <iso>
   error       = <message nếu có>
   ```
   - `<key>` = md5(path) cho gọn.
   - Đây là nơi tra cứu trạng thái live, thay cho metadata tag của MinIO.

> Không dùng các set `pending` / `done` / `error` riêng như trước. Chỉ cần `detection:pushed`
> để chống trùng + `detection:status:*` để theo dõi. Trạng thái SUCCESS/ERROR phản ánh trong
> hash, không cần move file.

---

## 3. Layout folder

```
videos/            # thả video vào đây (phẳng, scheduler quét folder này)
  *.mp4
detection/         # output ảnh đã vẽ bbox (giữ nguyên DETECTION_FOLDER hiện có)
weights/           # model ONNX (.onnx)
```

Không cần `done/`, `error/`, `processing/`. File ở nguyên vị trí; Redis quyết định đã xử lý hay chưa.

---

## 4. Các file cần thêm / sửa

### 4.1. Thêm mới: `local_storage.py` (thay vai trò `storage.py`)

```python
# Trách nhiệm: quét folder + quản lý Redis set "đã đẩy" + key trạng thái
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import redis
from config.constant import config

r = redis.Redis.from_url(config.CELERY_BROKER_URL, decode_responses=True)

PUSHED_SET = "detection:pushed"          # set path đã đẩy vào queue (dedup)

def _key(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def scan_new_videos() -> list[str]:
    """Trả về path video trong VIDEO_DIR CHƯA từng được đẩy vào Redis."""
    paths = []
    for p in Path(config.VIDEO_DIR).rglob("*"):
        if p.suffix.lower() in config.VIDEO_EXTENSIONS:
            ap = str(p.resolve())
            if not r.sismember(PUSHED_SET, ap):
                paths.append(ap)
    return sorted(paths, key=lambda x: Path(x).stat().st_mtime)

def mark_pushed(path: str) -> bool:
    """Đánh dấu đã đẩy. SADD trả 1 nếu mới (claim thành công), 0 nếu đã có."""
    return r.sadd(PUSHED_SET, path) == 1

def set_status(path: str, status: str, task_id: str | None = None, error: str | None = None):
    data = {"path": path, "status": status, "updated_at": _now()}
    if task_id:
        data["task_id"] = task_id
    if error:
        data["error"] = error
    r.hset(f"detection:status:{_key(path)}", mapping=data)

def get_status(path: str) -> dict:
    return r.hgetall(f"detection:status:{_key(path)}")
```

> Thay thế tương ứng `storage.py` cũ:
> `get_processed_need_videos()` → `scan_new_videos()`;
> `update_object_metadata()` → `set_status()`;
> dedup tag → `mark_pushed()` (`detection:pushed`).

### 4.2. Sửa `config/constant.py` + `config.yml`

Bỏ field MinIO bắt buộc, thêm folder local + đường dẫn model ONNX:

```yaml
# config.yml
LOCAL_STORAGE:
  VIDEO_DIR: "videos"          # folder chứa video local
```

```python
# constant.py — thêm field
VIDEO_DIR: str
# from_dict:
local = raw_config.get("LOCAL_STORAGE", {})
VIDEO_DIR=os.path.abspath(local.get("VIDEO_DIR", "videos")),
```

> MINIO_* có thể để optional (đặt `None`) để không phải xoá nhiều chỗ, nhưng không dùng tới.

### 4.3. `model_registry.py` — chuyển sang ONNX Runtime

Hai lựa chọn — chọn 1:

- **(A) Ultralytics nạp `.onnx` trực tiếp** (ít sửa nhất):
  ```python
  from ultralytics import YOLO
  m1 = YOLO(w1)   # w1 = "weights/best_detect_thiet_bi.onnx"  → tự dùng onnxruntime-gpu
  ```
  → Chỉ cần export `.pt → .onnx` và đổi đường dẫn weight trong DB `detection_models`.

- **(B) Gọi onnxruntime-server qua HTTP** (dùng image `kibaes/onnxruntime-server`):
  ```python
  import requests, numpy as np
  def infer_onnx(model_name, input_tensor):
      resp = requests.post(
          f"http://onnxruntime:8001/v1/{model_name}",   # cổng server ONNX
          json={"inputs": input_tensor.tolist()},
      )
      return resp.json()["outputs"]
  ```
  → Worker không cần GPU; inference chạy trong container onnxruntime-server. Cần sửa
  `inference.py` để gọi HTTP thay vì `model.predict()`.

> **Khuyến nghị:** dùng **(A)** trước cho nhanh (giữ nguyên `inference.py`), chỉ cần
> `pip install onnxruntime-gpu` và export model. Chuyển sang **(B)** nếu muốn tách hẳn
> service inference ra container `onnxruntime-server` riêng.

### 4.4. Sửa `tasks.py` — `process_media_task`

Đổi tham số `media` thành **path local**, mọi trạng thái ghi vào Redis (không metadata MinIO):

```python
@celery_app.task(bind=True, queue="detection", max_retries=None)
def process_media_task(self, video_path: str, task_model_id: str, detection_task_id: int):
    db = SessionLocal()
    try:
        requested_model_id = str(task_model_id or config.DEFAULT_MODEL_ID)
        local_storage.set_status(video_path, TaskStatus.PROCESSING.value, self.request.id)
        # update_detection_task_status(PROCESSING) — giữ nếu vẫn dùng bảng detection_tasks

        models = get_models(requested_model_id)
        folder_name = Path(video_path).stem
        data = process_video(video_path, folder_name, models)   # path local trực tiếp

        # ... build error dict + lưu InsulatorModel (giữ nguyên logic ALLOWED_MAP) ...

        local_storage.set_status(video_path, TaskStatus.SUCCESS.value, self.request.id)
        return build_task_result(task_id=self.request.id, status=TaskStatus.SUCCESS,
                                 result={"video_path": video_path, "detections": len(data)})
    except Exception as exc:
        # retry chưa hết: set_status(RETRY) + self.retry(countdown=...)
        # hết retry:      set_status(FAILED_RETRY, error=str(exc))
        ...
    finally:
        db.close()
```

Thay đổi chính so với bản MinIO:
- Bỏ presigned `url` → dùng `video_path` trực tiếp (OpenCV `cv2.VideoCapture(path)`).
- Bỏ toàn bộ `update_object_metadata(... TRUE/ERROR/PROCESSING)` → thay bằng `local_storage.set_status(...)`.
- Bỏ phần `media`/`imageUrl` trỏ MinIO host (link kết quả trỏ static local `/files/...`).

### 4.5. Sửa `main.py` — scheduler dispatch

```python
def detect_insulator():
    try:
        video_paths = local_storage.scan_new_videos()    # chỉ video chưa đẩy
        if not video_paths:
            return
        db = SessionLocal()
        active = get_active_detection_model(db); db.close()
        default_model_id = str(active.id if active else config.DEFAULT_MODEL_ID)

        db = SessionLocal()
        try:
            for path in video_paths:
                if not local_storage.mark_pushed(path):   # SADD dedup, atomic
                    continue
                task = create_detection_task(db, object_name=path,
                                             media_type="video", model_id=default_model_id)
                local_storage.set_status(path, TaskStatus.PENDING.value)
                res = process_media_task.apply_async(
                    args=[path, default_model_id, task.id], queue="detection")
                task.celery_task_id = res.id; db.commit()
        finally:
            db.close()
    except Exception:
        traceback.print_exc()
```

Giữ APScheduler cron `minute="*"`, `max_instances=1`. Bỏ `get_processed_need_images` (chỉ video;
thêm `scan_new_images()` tương tự nếu cần ảnh).

### 4.6. Endpoint tra trạng thái theo path (tùy chọn)

```python
@app.get("/videos/status")
def video_status(path: str):
    return local_storage.get_status(os.path.abspath(path))
```

---

## 5. Docker / Compose

Bỏ service `redis` và `minio` khỏi compose (Redis đã chạy sẵn). Thêm `onnxruntime-server`.
Worker và onnxruntime trỏ tới Redis external qua network.

```yaml
# docker-compose.yml
services:
  onnxruntime:
    image: kibaes/onnxruntime-server:1.27.0-linux-cuda13
    container_name: onnxruntime
    volumes:
      - ./weights:/models          # nơi chứa .onnx
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    networks:
      - mynetwork

  worker_detection:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/app
      - ./weights:/app/weights
      - ./detection:/app/detection
      - ./videos:/app/videos        # folder video local
    environment:
      - MODEL_ID=${MODEL_ID:-1}
    command: celery -A celery_app worker -Q detection --concurrency=1 --loglevel=info -n worker@%h
    # Nếu dùng phương án (A) — worker tự chạy ONNX GPU thì giữ phần GPU dưới:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: on-failure
    networks:
      - mynetwork

networks:
  mynetwork:
    external: true          # network của Redis container đang chạy
    # name: <tên_network_redis_hiện_tại>
```

> - **Redis external:** trỏ `CELERY_BROKER_URL` / `RESULT_BACKEND` trong `config.yml` tới
>   host:port của Redis đang chạy (vd `redis://redis:6379/0` nếu cùng network, hoặc
>   `redis://<host_ip>:6379/0`). Đảm bảo worker và Redis cùng `mynetwork`.
> - **GPU:** nếu dùng phương án (B) (onnxruntime-server lo inference), worker **không cần**
>   GPU → bỏ block `deploy.resources` ở `worker_detection`, chỉ để GPU cho `onnxruntime`.
> - Lấy tên network Redis hiện tại: `docker inspect <redis_container> -f '{{json .NetworkSettings.Networks}}'`.

Chuẩn bị model ONNX:
```bash
# Phương án (A): export .pt -> .onnx (chạy 1 lần)
yolo export model=weights/best_detect_thiet_bi.pt format=onnx device=0
# cập nhật weight_1/2/21 trong bảng detection_models trỏ tới file .onnx
```

Chạy:
```bash
docker compose up -d onnxruntime worker_detection
python main.py                       # scheduler + API (port 8010)
cp my_video.mp4 videos/              # thả video → tự xử lý ở tick kế tiếp
```

---

## 6. Luồng hoạt động end-to-end

```
   videos/*.mp4 ──scan──▶ FastAPI scheduler (mỗi phút)
                          detect_insulator():
                            scan_new_videos()          # lọc path CHƯA ở detection:pushed
                            mark_pushed(path) ─SADD──▶ ┌─────────┐
                            create detection_tasks row │  Redis  │ (external, đang chạy)
                            set_status(PENDING) ──HSET▶│ broker  │
                            apply_async(queue=detection)│ + sets  │
                                                        └────┬────┘
                                                             │ consume
                                                             ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ Celery worker — process_media_task(video_path, model_id, task_id) │
   │   set_status PROCESSING (Redis hash)                              │
   │   get_models(model_id) → ONNX (A: trong worker | B: gọi server)  │
   │   process_video(video_path) → detect + SORT + ALLOWED_MAP        │
   │   lưu InsulatorModel (PostgreSQL)                                 │
   │   set_status SUCCESS                                              │
   │   lỗi → RETRY (countdown) → hết retry: set_status FAILED_RETRY    │
   └──────────────────────────────────────────────────────────────────┘
            (file video KHÔNG bị move — chỉ Redis quyết định đã xử lý)
                                   │ (phương án B)
                                   ▼
                       ┌───────────────────────────┐
                       │ onnxruntime-server (GPU)   │  kibaes/onnxruntime-server
                       │  /v1/<model>  ◀── HTTP     │  :1.27.0-linux-cuda13
                       └───────────────────────────┘
```

---

## 7. Trạng thái & Retry

- `TaskStatus`: `PENDING → PROCESSING → SUCCESS | RETRY → FAILED_RETRY` (giữ enum hiện có).
- **Trạng thái chỉ nằm ở Redis** `detection:status:<key>` (hash) — không lưu ở folder.
- Chống trùng: **chỉ** dựa vào set `detection:pushed`. Đã đẩy = không đẩy lại.
- Bảng `detection_tasks` (PostgreSQL): **tùy chọn giữ lại** để đếm retry bền vững. Nếu muốn
  tối giản theo đúng yêu cầu "trạng thái chỉ ở Redis", có thể đếm retry bằng field trong
  hash (`retries`) và bỏ bảng `detection_tasks` — đánh đổi: mất lịch sử khi Redis flush.
- `GET /tasks/{task_id}` (Celery AsyncResult) + `GET /videos/status?path=...` (Redis hash).

---

## 8. Checklist triển khai

- [ ] Tạo folder `videos/` chứa video.
- [ ] Trỏ `CELERY_BROKER_URL` / `RESULT_BACKEND` tới Redis container đang chạy; nối cùng network.
- [ ] Thêm block `LOCAL_STORAGE.VIDEO_DIR` vào `config.yml` + field trong `constant.py`.
- [ ] Viết `local_storage.py` (`scan_new_videos` / `mark_pushed` / `set_status` / `get_status`).
- [ ] Export model `.pt → .onnx`; cập nhật `detection_models.weight_*` trỏ file `.onnx`.
- [ ] Chọn phương án ONNX: **(A)** Ultralytics nạp `.onnx` trong worker, hoặc **(B)** gọi
      `onnxruntime-server` qua HTTP (sửa `inference.py`).
- [ ] Sửa `tasks.py`: `process_media_task(video_path, ...)`, thay metadata → `local_storage.set_status`.
- [ ] Sửa `main.py`: `detect_insulator()` dùng `scan_new_videos()` + `mark_pushed()`.
- [ ] Thêm `onnxruntime-server` + `worker_detection` vào `docker-compose.yml`; network `external: true`.
- [ ] `pip install redis onnxruntime-gpu` (nếu phương án A).
- [ ] Test: thả 1 video vào `videos/` → tick kế tiếp đẩy vào Redis, worker xử lý, `detection:status:*`
      = SUCCESS, ảnh trong `detection/`, và lần quét sau **không đẩy lại** (đã có trong `detection:pushed`).

---

## 9. Lưu ý quan trọng

- **Atomic dedup:** `SADD detection:pushed` trả `1/0` — claim an toàn dù nhiều scheduler/worker.
- **File đang copy dở:** scheduler có thể quét phải video copy chưa xong. Khắc phục: chỉ nhận
  file có `mtime` cũ hơn N giây, hoặc copy với đuôi tạm `.part` rồi rename khi xong.
- **Redis bị flush:** mất `detection:pushed` → video sẽ bị quét và đẩy lại (xử lý lại từ đầu).
  Vì không move file, đây là rủi ro duy nhất cần lưu ý. Nếu cần bền vững: bật Redis AOF/RDB
  persistence (vốn nên có), hoặc giữ bảng `detection_tasks` làm backup.
- **Dọn Redis:** `detection:pushed` và `detection:status:*` lớn dần. Cân nhắc TTL cho hash
  trạng thái (vd `EXPIRE` 7 ngày) nếu không cần giữ lâu.
- **Đường dẫn tuyệt đối:** luôn `Path.resolve()` để path khớp giữa scheduler ↔ worker, nhất
  là trong Docker (mount `./videos` phải cùng path bên trong container, vd `/app/videos`).
- **ONNX phương án (B):** kiểm tra API/cổng thực tế của `kibaes/onnxruntime-server` (endpoint,
  format input/output) trước khi viết client — chạy `docker run ... --help` hoặc xem doc image.
```

