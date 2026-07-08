"""Background scheduler: scan videos/{cameraId}/* + dispatch clip mới qua Celery.

Module độc lập:
    • `main()` để chạy như process riêng:   python -m video_scheduler
    • `run_loop()` async để embed vào FastAPI qua lifespan.

Logic mỗi tick:
    1. Scan thư mục VIDEO_DIR/<cameraId>/* lấy snapshot {key → VideoState}
    2. Bỏ qua file mtime trẻ hơn MIN_AGE (camera đang ghi)
    3. Diff với snapshot trước → file mới = ADD
    4. Với mỗi ADD:
         resolve_video_context_status(cameraId, startTime)  ← qua Redis
         mark_pushed (dedup)
         process_clip_task.apply_async(..., queue='clip')
    5. Sleep INTERVAL giây

Trạng thái:
    • In-memory dict `_prev` cho diff nhanh
    • Backup: SADD qhh:ai:clip:pushed (cross-process dedup, sống sót restart)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import clip_queue as cq
from web.video_context import ConfigurationException, resolve_video_context_status

logger = logging.getLogger("video_scheduler")

VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "/app/videos"))
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
INTERVAL_SEC = float(os.getenv("VIDEO_SCAN_INTERVAL_SEC", "10"))
MIN_AGE_SEC = float(os.getenv("VIDEO_MIN_AGE_SEC", "30"))  # đợi camera ghi xong
DISPATCH_SPACING_SEC = float(os.getenv("VIDEO_DISPATCH_SPACING_SEC", "0.5"))
MAX_PER_TICK = int(os.getenv("VIDEO_MAX_PER_TICK", "32"))

# Múi giờ VN — tên file ghi theo wall-clock VN (vd 20260626_101232 = 10:12:32 VN).
VN_TZ = timezone(timedelta(hours=7))

# Pattern: 8 digits date + '_' + 6 digits time. VD: '20260626_101232_web'
DATETIME_RE = re.compile(r"(\d{8})_(\d{6})")

# Fallback: epoch (s hoặc ms) cuối cùng trong stem.
EPOCH_RE = re.compile(r"(\d{10,13})")


@dataclass(frozen=True)
class VideoState:
    camera_id: str
    path: str
    start_time: datetime
    size: int
    mtime_ns: int


def _parse_start_time(name: str) -> datetime | None:
    """Parse videoStartTime từ tên file. Hỗ trợ 2 format:

    1) ``YYYYMMDD_HHMMSS[_*]`` — wall-clock giờ VN (vd 20260626_101232_web)
    2) Fallback: epoch 10-13 chữ số (vd clip_1782710904000)
    """
    stem = Path(name).stem

    m = DATETIME_RE.search(stem)
    if m:
        date_s, time_s = m.group(1), m.group(2)
        try:
            local = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S")
            return local.replace(tzinfo=VN_TZ).astimezone(timezone.utc)
        except ValueError:
            pass  # ngày/giờ không hợp lệ → thử epoch

    candidates = EPOCH_RE.findall(stem)
    if candidates:
        ts = int(candidates[-1])
        if ts > 10**12:
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        if ts > 10**9:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    return None


def _iter_video_files(cam_dir: Path):
    """Yield mọi file video trong cam_dir, đệ quy qua subfolder (kiểu YYYYMMDD)."""
    for entry in cam_dir.iterdir():
        if entry.is_dir():
            yield from _iter_video_files(entry)
        elif entry.is_file() and entry.suffix.lower() in VIDEO_EXTS:
            yield entry


def _scan() -> dict[str, VideoState]:
    """Trả snapshot tất cả video đủ "chín" (mtime > MIN_AGE).

    Cấu trúc folder hỗ trợ:
      VIDEO_DIR/{cameraId}/*.ext                      (legacy, flat)
      VIDEO_DIR/{cameraId}/{YYYYMMDD}/*.ext           (FFmpeg_record style)
    """
    now_ns = time.time_ns()
    threshold = int(MIN_AGE_SEC * 1e9)
    out: dict[str, VideoState] = {}
    if not VIDEO_DIR.exists():
        return out
    for cam_dir in VIDEO_DIR.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_id = cam_dir.name
        for vfile in _iter_video_files(cam_dir):
            try:
                st = vfile.stat()
            except FileNotFoundError:
                continue
            if now_ns - st.st_mtime_ns < threshold:
                continue
            start = _parse_start_time(vfile.name)
            if start is None:
                logger.debug("skip (no epoch in name): %s", vfile.name)
                continue
            full_path = str(vfile.resolve())
            out[full_path] = VideoState(
                camera_id=cam_id,
                path=full_path,
                start_time=start,
                size=st.st_size,
                mtime_ns=st.st_mtime_ns,
            )
    return out


def _dispatch_one(v: VideoState) -> str:
    """Resolve context + apply_async. Trả status string."""
    try:
        ctx, reason = resolve_video_context_status(v.camera_id, v.start_time)
    except ConfigurationException as exc:
        logger.error("[CONFIG] %s @ %s: %s", v.camera_id, v.start_time, exc)
        return "CONFIG_ERROR"
    if ctx is None:
        if reason == "AI_DISABLED":
            logger.info("ai disabled: cam=%s start=%s", v.camera_id, v.start_time)
        elif reason == "NO_CAMERA_CLASS":
            logger.info("no camera-class: cam=%s start=%s", v.camera_id, v.start_time)
        else:
            logger.info("no active period: cam=%s start=%s", v.camera_id, v.start_time)
        return reason

    if not cq.mark_pushed(v.path):
        return "ALREADY_QUEUED"

    # Lazy import — tasks pull in Celery + ONNX nặng, chỉ cần khi thực sự dispatch.
    from tasks import process_clip_task

    cq.set_status(
        v.path, "PENDING",
        cameraId=ctx["cameraId"], classId=ctx["classId"],
    )
    task = process_clip_task.apply_async(
        args=[v.path, ctx["cameraId"], ctx["classId"]],
        queue="clip",
    )
    cq.set_status(
        v.path, "PENDING",
        taskId=task.id, cameraId=ctx["cameraId"], classId=ctx["classId"],
    )
    logger.info(
        "[DISPATCH] cam=%s class=%s task=%s file=%s",
        ctx["cameraId"], ctx["classId"], task.id, Path(v.path).name,
    )
    return "PENDING"


_prev: dict[str, VideoState] = {}
_status = {
    "last_tick_at": None,
    "last_count_scanned": 0,
    "last_count_new": 0,
    "last_count_dispatched": 0,
    "errors": 0,
    "running": False,
}


def tick() -> dict:
    """Một vòng quét + dispatch. Trả thống kê."""
    global _prev
    curr = _scan()
    new_keys = sorted(set(curr) - set(_prev))[:MAX_PER_TICK]
    dispatched = 0
    statuses: dict[str, int] = {}
    for k in new_keys:
        try:
            s = _dispatch_one(curr[k])
        except Exception:
            logger.exception("dispatch failed for %s", curr[k].path)
            _status["errors"] += 1
            s = "EXCEPTION"
        statuses[s] = statuses.get(s, 0) + 1
        if s == "PENDING":
            dispatched += 1
            if DISPATCH_SPACING_SEC > 0:
                time.sleep(DISPATCH_SPACING_SEC)
    _prev = curr
    info = {
        "scanned": len(curr),
        "new": len(new_keys),
        "dispatched": dispatched,
        "by_status": statuses,
    }
    _status.update({
        "last_tick_at": datetime.now(timezone.utc).isoformat(),
        "last_count_scanned": info["scanned"],
        "last_count_new": info["new"],
        "last_count_dispatched": info["dispatched"],
    })
    return info


def get_status() -> dict:
    return dict(_status, video_dir=str(VIDEO_DIR), interval_sec=INTERVAL_SEC)


async def run_loop(stop_event: asyncio.Event | None = None) -> None:
    """Async loop dành cho FastAPI lifespan. Stop bằng cách set stop_event."""
    _status["running"] = True
    logger.info(
        "video_scheduler started: dir=%s interval=%.1fs min_age=%.1fs",
        VIDEO_DIR, INTERVAL_SEC, MIN_AGE_SEC,
    )
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            try:
                info = await asyncio.to_thread(tick)
                if info["new"]:
                    logger.info("tick: %s", info)
            except Exception:
                logger.exception("tick failed")
                _status["errors"] += 1
            try:
                await asyncio.wait_for(
                    stop_event.wait() if stop_event else asyncio.sleep(INTERVAL_SEC),
                    timeout=INTERVAL_SEC,
                )
            except asyncio.TimeoutError:
                pass
    finally:
        _status["running"] = False
        logger.info("video_scheduler stopped")


def main() -> None:
    """Chế độ standalone:  python -m video_scheduler"""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stop_event = asyncio.Event()

    def _stop(*_a):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    asyncio.run(run_loop(stop_event))


if __name__ == "__main__":
    main()
