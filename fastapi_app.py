"""FastAPI gateway:

  POST /api/ai/submit                               — nhận classroom config + dispatch videos → Celery
  GET  /api/ai/result/{cameraId}/{classId}          — poll kết quả (merged per-student)
  GET  /api/ai/result?cameraId=<guid>&classId=<guid>  — legacy live-stream result
  POST /api/students/{studentId}/face-quality-check

AI clip pipeline: caller gửi classroom JSON → ta lưu vào Redis → Celery worker xử lý
từng video trong videos/ → kết quả trả ra theo schema request đầu vào + absence/attention/yaw/pitch.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path as FPath
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config_loader import env_or_config
from web.api_ai_result import build_ai_result_payload
from web.api_face_register import register_user_with_face
from web.video_context import (
    ConfigurationException,
    resolve_video_context_status,
)

# ---------------------------------------------------------------------------
# Internal web_server endpoint (same Docker network)
# ---------------------------------------------------------------------------
WEB_SERVER_BASE = os.getenv("WEB_SERVER_BASE", "http://web:8090").rstrip("/")
_HTTP_TIMEOUT = 5.0  # giây


async def _fetch_ai_state(camera_id: str, class_id: str) -> dict:
    """Gọi web_server lấy AI state hiện tại (non-blocking async)."""
    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            resp = await client.get(f"{WEB_SERVER_BASE}/api/ai/status")
            resp.raise_for_status()
            state = resp.json()
            # Chỉ trả state nếu đúng camera+class đang chạy
            if (
                state.get("camera_id") == camera_id
                and state.get("class_id") == class_id
            ):
                return state
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Face detector – khởi tạo 1 lần, dùng chung cho mọi request
# ---------------------------------------------------------------------------
_detector_lock = threading.Lock()
_face_detector = None


def _get_face_detector():
    global _face_detector
    if _face_detector is None:
        with _detector_lock:
            if _face_detector is None:
                from workers.face_models import RetinaFaceDetector
                _face_detector = RetinaFaceDetector()
    return _face_detector


_arc_extractor = None
_arc_lock = threading.Lock()


def _get_arc_extractor():
    global _arc_extractor
    if _arc_extractor is None:
        with _arc_lock:
            if _arc_extractor is None:
                from workers.face_models import ArcFaceExtractor
                _arc_extractor = ArcFaceExtractor()
    return _arc_extractor


def _truthy_setting(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off", ""}:
            return False
    return bool(value)


def _web_test_record_on() -> bool:
    return _truthy_setting(
        env_or_config("QHH_WEB_RECORD_ON_AI", "web_record", "on_ai", True),
        True,
    )


def _scheduler_requested() -> bool:
    return _truthy_setting(os.getenv("VIDEO_SCHEDULER_ENABLED", "1"), True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Preload detector khi startup để request đầu không chậm
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _get_face_detector)
    await loop.run_in_executor(None, _get_arc_extractor)

    # Background video scheduler là luồng chạy thật theo Redis aiEnabled + TKB.
    # Khi web test còn bật, nút Start AI trên web vẫn là chủ luồng để tránh
    # scheduler xử lý song song cùng các segment test.
    scheduler_task = None
    stop_event = asyncio.Event()
    web_test_on = _web_test_record_on()
    if _scheduler_requested() and not web_test_on:
        import video_scheduler
        scheduler_task = asyncio.create_task(
            video_scheduler.run_loop(stop_event),
            name="video_scheduler",
        )
    elif web_test_on:
        print(
            "[api] video scheduler disabled while web test recording is enabled",
            flush=True,
        )

    try:
        yield
    finally:
        if scheduler_task is not None:
            stop_event.set()
            try:
                await asyncio.wait_for(scheduler_task, timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                scheduler_task.cancel()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QHH AI Public API",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve distraction snapshot images: GET /snapshots/{camId}/{videoStem}/{frame}.jpg
_SNAP_DIR = Path(os.getenv("QHH_DISTRACTION_SNAP_DIR", "/app/detection/distraction"))
_SNAP_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=str(_SNAP_DIR)), name="snapshots")

@app.get("/api/admin/scheduler/status",
         summary="Trạng thái video scheduler",
         include_in_schema=False)
async def scheduler_status():
    import video_scheduler
    return video_scheduler.get_status()


@app.post("/api/admin/scheduler/tick",
          summary="Chạy 1 tick scan + dispatch ngay",
          include_in_schema=False)
async def scheduler_tick():
    import video_scheduler
    info = await asyncio.to_thread(video_scheduler.tick)
    return {"ok": True, **info}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Face register schemas — quality check là phần trong /register-face
# ---------------------------------------------------------------------------
class FaceQualityIssue(BaseModel):
    code: str = Field(..., description="Mã issue, ví dụ BLURRY_FACE")
    message: str = Field(..., description="Mô tả tiếng Việt")


class FaceQualityCheckResponse(BaseModel):
    id: str = Field(..., description="Student GUID")
    studentCode: str = ""
    fullName: str = ""
    avatarUrl: str = Field(default="", description="URL ảnh gốc đã gửi lên")
    passed: bool = Field(..., description="Có vượt qua tất cả check hay không")
    faceCount: int = Field(..., description="Số khuôn mặt phát hiện trong ảnh")
    qualified: bool = Field(..., description="Đủ chất lượng đăng ký face hay không")
    issues: list[FaceQualityIssue] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /api/students/{studentId}/register-face
# ---------------------------------------------------------------------------
class FaceRegisterIssue(BaseModel):
    code: int = Field(..., description="Numeric code (8001/8004/8006/8008/8010)")
    name: str = Field(..., description="Tên mã, vd 'NO_FACE'")
    message: str = Field(..., description="Mô tả tiếng Việt")


class FaceRegisterResponse(BaseModel):
    qualified: bool = Field(
        ...,
        description=(
            "`true`  → ảnh đạt chuẩn, embedding đã lưu Redis. `issues` rỗng.\n"
            "`false` → ảnh vi phạm ít nhất 1 quy tắc, xem `issues[]`."
        ),
    )
    message: str = Field(
        ...,
        description=(
            "Thông điệp tổng (tiếng Việt). Khi qualified=true là "
            "'Đăng ký khuôn mặt thành công'. Khi false là tóm tắt nhanh "
            "các lỗi đầu tiên (chi tiết xem `issues`)."
        ),
    )
    issues: list[FaceRegisterIssue] = Field(
        default_factory=list,
        description=(
            "Danh sách MỌI mã loi phát hiện được. Rỗng khi qualified=true. "
            "Có thể chứa nhiều mã đồng thời, vd ảnh vừa mờ vừa nhiều mặt "
            "→ trả `[BLURRY_FACE, MULTIPLE_FACES]`.\n\n"
            "Bảng mã:\n"
            "  • `8001` NO_FACE         — Không phát hiện khuôn mặt nào.\n"
            "  • `8004` MULTIPLE_FACES  — Có nhiều hơn một khuôn mặt.\n"
            "  • `8006` BLURRY_FACE     — Khuôn mặt bị mờ.\n"
            "  • `8008` INVALID_IMAGE   — File ảnh hỏng / không decode được.\n"
            "  • `8010` REGISTER_FAILED — Quality OK nhưng hậu kỳ lưu lỗi."
        ),
    )


@app.post(
    "/api/students/register-face",
    response_model=FaceRegisterResponse,
    summary="Đăng ký khuôn mặt: client gửi studentCode + ảnh, server tự sinh studentId và lưu",
    description=(
        "Client gửi `studentCode` (vd HS001) + `avatar` + (optional) "
        "`fullName`, `username`, `userType`. Server tự map studentCode → "
        "studentId (UUID, ổn định qua các lần gọi với cùng code), lưu ảnh "
        "+ embedding ArcFace + profile vào `qhh:user:{studentId}`. "
        "Gọi lại cùng studentCode → cập nhật cùng record (không tạo ID mới)."
    ),
)
async def register_face(
    studentCode: Annotated[str, Form(description="Mã học sinh, vd HS001 (bắt buộc)")],
    avatar: Annotated[UploadFile, File(description="File ảnh khuôn mặt")],
    fullName: Annotated[str | None, Form(description="Họ tên đầy đủ (optional)")] = None,
    username: Annotated[str | None, Form(description="Username đăng nhập (optional)")] = None,
    userType: Annotated[str | None, Form(description="student | teacher | other")] = None,
):
    code = studentCode.strip()
    if not code:
        raise HTTPException(status_code=400, detail="studentCode là bắt buộc")
    content = await avatar.read()
    if not content:
        raise HTTPException(status_code=400, detail="Ảnh rỗng")
    if len(content) > 12 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File ảnh vượt quá 12 MB")

    # Map studentCode → studentId stable (tạo mới UUID nếu chưa có).
    from web.api_face_register import get_or_create_student_id
    student_id = await asyncio.to_thread(get_or_create_student_id, code)

    loop = asyncio.get_event_loop()
    raw = await loop.run_in_executor(
        None,
        lambda: register_user_with_face(
            student_id,
            file_bytes=content,
            student_code=code,
            full_name=fullName or "",
            username=username or "",
            user_type=userType or "student",
            face_detector=_get_face_detector(),
            arc_extractor=_get_arc_extractor(),
        ),
    )

    issues = raw.get("issues", []) or []
    qualified = bool(raw.get("qualified", False))
    registered = bool(raw.get("registered", False))

    # Quality OK nhưng hậu kỳ fail → đẩy reason vào issues_messages.
    if qualified and not registered:
        reason = raw.get("reason") or "REGISTER_FAILED"
        if reason.startswith("SAVE_IMAGE_FAILED"):
            reason = "REGISTER_FAILED"
        if reason.startswith("EMBEDDING_FAILED"):
            reason = "REGISTER_FAILED"
        issues = [{"code": reason, "message": "Đăng ký không thành công, vui lòng thử lại."}]
        qualified = False

    # ── Mã loi nghiệp vụ ─────────────────────────────────────────────────
    #
    #   8000  SUCCESS         — Pass quality + embedding đã ghi Redis.
    #   8001  NO_FACE         — RetinaFace không detect được khuôn mặt.
    #   8004  MULTIPLE_FACES  — Detect ≥ 2 khuôn mặt.
    #   8006  BLURRY_FACE     — Laplacian variance vùng face < ngưỡng 25.0.
    #   8008  INVALID_IMAGE   — File ảnh hỏng / không decode được.
    #   8010  REGISTER_FAILED — Lỗi server hậu kỳ (lưu file/ArcFace/Redis).
    #
    # Lưu ý: các case loi CÓ THỂ xảy ra đồng thời (vd vừa mờ vừa nhiều mặt).
    # Trả về list MỌI issue. Chỉ case thành công mới đứng riêng một mình.
    # ─────────────────────────────────────────────────────────────────────
    ISSUE_CODES = {
        "NO_FACE":         (8001, "Không tìm thấy khuôn mặt trong ảnh."),
        "MULTIPLE_FACES":  (8004, "Có nhiều hơn một khuôn mặt trong ảnh."),
        "FACE_TOO_SMALL":  (8003, "Khuôn mặt quá nhỏ so với khung hình."),
        "BLURRY_FACE":     (8006, "Khuôn mặt bị mờ."),
        "INVALID_IMAGE":   (8008, "Không đọc được dữ liệu ảnh đầu vào."),
    }

    if qualified:
        return {
            "qualified": True,
            "message":   "Đăng ký khuôn mặt thành công.",
            "issues":    [],
        }

    # Build full issues list — KHÔNG dedupe nhiều case khác nhau.
    out_issues: list[dict] = []
    seen_names: set[str] = set()
    for it in issues:
        name = str(it.get("code") or "NO_FACE")
        if name not in ISSUE_CODES:
            name = "NO_FACE"
        if name in seen_names:
            continue
        seen_names.add(name)
        num, msg = ISSUE_CODES[name]
        out_issues.append({
            "code":    num,
            "name":    name,
            "message": str(it.get("message") or msg),
        })
    if not out_issues:
        num, msg = ISSUE_CODES["NO_FACE"]
        out_issues.append({"code": num, "name": "NO_FACE", "message": msg})

    top_msg = " | ".join(i["message"] for i in out_issues)
    return {
        "qualified": False,
        "message":   top_msg,
        "issues":    out_issues,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VIDEO_DIR = FPath(os.getenv("VIDEO_DIR", "/app/videos"))
_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}


def _get_db():
    from db import redis_client as _db
    return _db.get_client()


# ---------------------------------------------------------------------------
# Resolve-video helpers
# ---------------------------------------------------------------------------
from datetime import datetime, timezone


def _camera_video_dir(camera_id: str) -> FPath:
    return _VIDEO_DIR / camera_id


def _parse_video_start_time(name: str) -> datetime | None:
    """Dùng chung parser với video_scheduler để đồng nhất hành vi."""
    from video_scheduler import _parse_start_time
    return _parse_start_time(name)


def _list_camera_videos(camera_id: str) -> list[FPath]:
    """Đệ quy folder camera (kể cả YYYYMMDD subdir kiểu FFmpeg_record)."""
    folder = _camera_video_dir(camera_id)
    if not folder.exists():
        return []
    out: list[FPath] = []
    stack = [folder]
    while stack:
        cur = stack.pop()
        for p in cur.iterdir():
            if p.is_dir():
                stack.append(p)
            elif p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
                out.append(p)
    out.sort(key=lambda p: p.name)
    return out


# ---------------------------------------------------------------------------
# POST /api/ai/submit
# ---------------------------------------------------------------------------

class ResolveVideoRequest(BaseModel):
    cameraId: str
    videoStartTime: datetime
    videoPath: str = Field(
        default="",
        description="Optional. Bỏ trống thì tự scan videos/{cameraId}/* theo timestamp.",
    )
    callbackUrl: str = ""


def _resolve_video_error_detail(reason: str) -> str:
    if reason == "AI_DISABLED":
        return "Camera/lớp đang có tiết học nhưng aiEnabled đang tắt trong Redis"
    if reason == "NO_CAMERA_CLASS":
        return "Camera chưa có cấu hình camera-class trong Redis"
    if reason == "NO_CAMERA":
        return "cameraId là bắt buộc"
    return "Không có tiết học đang diễn ra cho camera tại thời điểm này"


@app.post("/api/ai/resolve-video-context",
          summary="Resolve class + period + AI config từ Redis (không dispatch)",
          include_in_schema=False)
async def resolve_video(body: ResolveVideoRequest):
    """Step 1-4 của instruction — chỉ trả context, không dispatch Celery."""
    try:
        ctx, reason = resolve_video_context_status(body.cameraId, body.videoStartTime)
    except ConfigurationException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=_resolve_video_error_detail(reason),
        )
    return ctx


@app.post("/api/ai/process-video",
          summary="Resolve context + dispatch 1 video → Celery",
          include_in_schema=False)
async def process_video(body: ResolveVideoRequest):
    """Flow đầy đủ:
        1. resolve_video_context(cameraId, videoStartTime) → class/period/AI config
        2. Tìm file video (body.videoPath hoặc auto-scan)
        3. Đẩy task vào queue 'clip' cho worker xử lý
    """
    try:
        ctx, reason = resolve_video_context_status(body.cameraId, body.videoStartTime)
    except ConfigurationException as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    if ctx is None:
        raise HTTPException(
            status_code=404,
            detail=_resolve_video_error_detail(reason),
        )

    cam_id = ctx["cameraId"]
    cls_id = ctx["classId"]

    # Resolve video file
    if body.videoPath:
        vpath = FPath(body.videoPath)
    else:
        candidates = _list_camera_videos(cam_id)
        match = None
        for p in candidates:
            ts = _parse_video_start_time(p.name)
            if ts is None:
                continue
            if abs((ts - body.videoStartTime.astimezone(timezone.utc)).total_seconds()) < 60:
                match = p
                break
        vpath = match
    if vpath is None or not vpath.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Không tìm thấy video cho camera {cam_id} tại {body.videoStartTime}",
        )

    # Đăng ký callbackUrl
    r = _get_db()
    if body.callbackUrl:
        r.set(f"qhh:ai:clip:callback:{cam_id}:{cls_id}", body.callbackUrl)

    # Dispatch Celery
    import asyncio
    import clip_queue as cq
    from tasks import process_clip_task

    def _dispatch():
        ap = str(vpath)
        md5 = cq.md5_of(ap)
        status_raw = r.hgetall(f"qhh:ai:clip:status:{md5}")
        status_raw = {(k.decode() if isinstance(k, bytes) else k):
                      (v.decode() if isinstance(v, bytes) else v)
                      for k, v in status_raw.items()}
        if status_raw.get("status") in ("PROCESSING", "SUCCESS"):
            return {"videoPath": ap, "md5": md5,
                    "taskId": status_raw.get("taskId"),
                    "status": status_raw.get("status")}
        if not cq.mark_pushed(ap):
            return {"videoPath": ap, "md5": md5, "taskId": None,
                    "status": "ALREADY_QUEUED"}
        cq.set_status(ap, "PENDING", cameraId=cam_id, classId=cls_id)
        task = process_clip_task.apply_async(args=[ap, cam_id, cls_id], queue="clip")
        cq.set_status(ap, "PENDING", taskId=task.id,
                      cameraId=cam_id, classId=cls_id)
        return {"videoPath": ap, "md5": md5,
                "taskId": task.id, "status": "PENDING"}

    loop = asyncio.get_event_loop()
    dispatch_result = await loop.run_in_executor(None, _dispatch)

    return {
        "context": ctx,
        "dispatch": dispatch_result,
    }


# ---------------------------------------------------------------------------
# POST /api/ai/result/{cameraId}/{classId}
# ---------------------------------------------------------------------------

def _clip_state_from_redis(camera_id: str, class_id: str) -> dict:
    """Đọc kết quả clip_inference mới nhất + map sang shape giống live AI state.

    Pipeline offline (run_clip) ghi vào qhh:ai:clip:result:{md5} và index
    qhh:ai:clip:index:{cam}:{cls} (ZSet sort theo epochMs). Lấy clip mới
    nhất rồi map mỗi student sang format mà build_ai_result_payload đợi.
    """
    r = _get_db()
    index_key = f"qhh:ai:clip:index:{camera_id}:{class_id}"
    latest = r.zrevrange(index_key, 0, 0)
    if not latest:
        return {}
    md5 = latest[0].decode() if isinstance(latest[0], bytes) else latest[0]
    raw = r.get(f"qhh:ai:clip:result:{md5}")
    if not raw:
        return {}
    clip = json.loads(raw)

    presence_th = 0.6
    distracted_th = 0.5
    results = []
    for s in clip.get("students", []):
        frames_total = clip.get("framesProcessed", 0) or 0
        frames_present = s.get("framesPresent", 0) or 0
        presence_ratio = frames_present / frames_total if frames_total else 0.0
        distracted_ratio = s.get("distractedRatio", 0.0) or 0.0
        present = presence_ratio >= presence_th
        gaze_focused = (distracted_ratio < distracted_th) if frames_present else None
        results.append({
            "student_id": s.get("studentId", ""),
            "assigned_student_id": s.get("studentId", ""),
            "present": present,
            "gaze_focused": gaze_focused,
            "gaze_yaw_deg": s.get("avgGazeYawDeg"),
            "gaze_pitch_deg": s.get("avgGazePitchDeg"),
        })

    import time
    return {
        "active": True,
        "results": results,
        "updated_at": time.time(),
        "camera_id": camera_id,
        "class_id": class_id,
    }


@app.post(
    "/api/ai/result/{cameraId}/{classId}",
    summary="Trạng thái lớp học hiện tại (absence / attention / yaw / pitch)",
)
async def clip_result(cameraId: str, classId: str):
    """Trả về trạng thái lớp tại thời điểm gọi.

    Ưu tiên live state từ web_server (nếu AI đang chạy trên camera RTSP),
    fallback sang kết quả clip_inference mới nhất trong Redis (offline video).
    Trả 404 nếu không có cấu hình camera-class JSON tương ứng trong Redis.
    """
    # Kiểm tra cấu hình camera-class tồn tại trong Redis trước khi xử lý.
    r = _get_db()
    cfg_key = f"qhh:attendance:camera-class:{cameraId}:{classId}"
    if not r.exists(cfg_key):
        # Phân biệt rõ lý do để client dễ debug.
        cam_index_exists = r.exists(f"qhh:attendance:camera-class:index:{cameraId}")
        cls_index_exists = r.exists(f"qhh:attendance:class-cameras:index:{classId}")
        if not cam_index_exists and not cls_index_exists:
            detail = (
                f"Không tìm thấy cameraId='{cameraId}' và classId='{classId}' "
                "trong cấu hình hệ thống."
            )
        elif not cam_index_exists:
            detail = f"Không tìm thấy cameraId='{cameraId}' trong cấu hình hệ thống."
        elif not cls_index_exists:
            detail = f"Không tìm thấy classId='{classId}' trong cấu hình hệ thống."
        else:
            detail = (
                f"Camera '{cameraId}' và lớp '{classId}' tồn tại nhưng chưa "
                "được gán cấu hình AI cho cặp này."
            )
        raise HTTPException(status_code=404, detail=detail)

    state = await _fetch_ai_state(cameraId, classId)
    if not state.get("results"):
        state = _clip_state_from_redis(cameraId, classId)
    return build_ai_result_payload(cameraId, classId, state)
