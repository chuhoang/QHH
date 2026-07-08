"""Production recorder: quay RTSP các camera aiEnabled thành segment ngắn.

Chạy độc lập với web-test UI (không cần bấm "Start AI"). Ghi segment vào
`recording_root()/{cameraId}/{YYYYMMDD}/%Y%m%d_%H%M%S_web.mkv` — đúng cây thư mục
mà `video_scheduler` quét mỗi 10s để dispatch qua Celery.

Luồng mỗi camera:
    discover → (aiEnabled + đang trong tiết Active) → 1 thread ffmpeg loop record
    segment WEB_RECORD_DURATION_SEC giây, lặp lại cho tới khi camera rời điều kiện.

Standalone:  python -m video_recorder

Tách bạch trách nhiệm:
    • recorder  : RTSP → file segment (module này)
    • scheduler : file segment → Celery dispatch (video_scheduler.py)
    • worker    : chạy AI + bắn snapshot (tasks.py)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from db import redis_client as db
from web.rtsp import build_camera_rtsp_url
from web.video_context import resolve_video_context_status
from workers.video_reader import recording_root

logger = logging.getLogger("video_recorder")

# Độ dài mỗi segment (giây). Cùng default với web-test để hành vi nhất quán.
SEGMENT_SEC = max(1, int(os.getenv("QHH_WEB_RECORD_DURATION_SEC", "10")))
# Chu kỳ refresh danh sách camera cần quay (bắt kịp bật/tắt aiEnabled, đổi tiết).
DISCOVER_INTERVAL_SEC = max(5, int(os.getenv("RECORDER_DISCOVER_INTERVAL_SEC", "30")))
# Gate "chỉ quay trong tiết học". Tắt (=0) → quay mọi camera aiEnabled 24/7.
RECORD_IN_PERIOD_ONLY = os.getenv("RECORDER_IN_PERIOD_ONLY", "1").strip() in {
    "1", "true", "yes", "on",
}

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
RTSP_TRANSPORT = os.getenv("QHH_WEB_RECORD_RTSP_TRANSPORT", "tcp")
STIMEOUT_US = os.getenv(
    "QHH_WEB_RECORD_STIMEOUT_US",
    os.getenv("QHH_WEB_RECORD_RW_TIMEOUT_US", "15000000"),
)
ANALYZE_US = os.getenv("QHH_WEB_RECORD_ANALYZE_US", "10000000")
PROBESIZE = os.getenv("QHH_WEB_RECORD_PROBESIZE", "10000000")
FFMPEG_LOGLEVEL = os.getenv("QHH_WEB_RECORD_FFMPEG_LOGLEVEL", "warning")

VIDEO_EXTS_ROOT = recording_root()


def _safe_camera_id(camera_id: str) -> str:
    """Sanitize để dùng làm tên thư mục (khớp web_server._safe_camera_id)."""
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in str(camera_id))


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# ---------------------------------------------------------------------------
# Discover: camera nào đang cần quay?
# ---------------------------------------------------------------------------
def _discover_targets() -> dict[str, str]:
    """Trả {cameraId: classId} các camera đang cần quay.

    Điều kiện:
      • Có trong qhh:attendance:camera-class:index:{cam}
      • camera-class config aiEnabled = true
      • (nếu RECORD_IN_PERIOD_ONLY) đang trong tiết Active của TKB — dùng lại
        resolve_video_context_status với thời điểm HIỆN TẠI.
    """
    r = db.get_client()
    targets: dict[str, str] = {}
    now = datetime.now(timezone.utc)
    for key in r.scan_iter(match="qhh:attendance:camera-class:index:*", count=500):
        key = key.decode() if isinstance(key, bytes) else key
        cam = key.split(":")[-1]
        if not cam:
            continue
        if RECORD_IN_PERIOD_ONLY:
            # Gate đầy đủ: index + TKB slot Active + aiEnabled (đúng luồng scheduler).
            try:
                ctx, _reason = resolve_video_context_status(cam, now)
            except Exception as exc:  # noqa: BLE001 — data hỏng 1 camera không giết recorder
                logger.warning("resolve failed cam=%s: %s", cam, exc)
                continue
            if ctx is not None:
                targets[cam] = ctx["classId"]
            continue
        # 24/7: chỉ cần 1 class aiEnabled bất kỳ của camera.
        for member in r.smembers(key):
            cls = member.decode() if isinstance(member, bytes) else member
            cfg = db.get_camera_class_config(cam, cls)
            if cfg and _truthy(cfg.get("aiEnabled", False)):
                targets[cam] = cls
                break
    return targets


# ---------------------------------------------------------------------------
# Recorder cho 1 camera
# ---------------------------------------------------------------------------
class CameraRecorder:
    """Thread quay 1 camera: loop ffmpeg segment cho tới khi bị stop."""

    def __init__(self, camera_id: str, class_id: str):
        self.camera_id = str(camera_id)
        self.class_id = str(class_id)
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"qhh-rec-{_safe_camera_id(self.camera_id)[:12]}",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self._stop_ffmpeg()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=SEGMENT_SEC + 5)

    def _stop_ffmpeg(self) -> None:
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    def _run(self) -> None:
        url = build_camera_rtsp_url(self.camera_id, self.class_id)
        if not url:
            logger.error("cam=%s class=%s: không dựng được RTSP URL — bỏ qua",
                         self.camera_id, self.class_id)
            return
        logger.info("start cam=%s class=%s segment=%ss",
                    self.camera_id, self.class_id, SEGMENT_SEC)
        while not self.stop_event.is_set():
            started = time.monotonic()
            self._record_once(url)
            # Quay liên tiếp: chỉ nghỉ nếu segment lỗi quá nhanh (tránh busy-loop).
            elapsed = time.monotonic() - started
            if elapsed < 1.0 and not self.stop_event.is_set():
                self.stop_event.wait(2.0)
        self._stop_ffmpeg()
        logger.info("stop cam=%s", self.camera_id)

    def _record_once(self, url: str) -> None:
        now = datetime.now()  # giờ VN nhờ TZ container = Asia/Ho_Chi_Minh
        directory = VIDEO_EXTS_ROOT / _safe_camera_id(self.camera_id) / now.strftime("%Y%m%d")
        directory.mkdir(parents=True, exist_ok=True)
        stem = now.strftime("%Y%m%d_%H%M%S_web")
        final_path = directory / f"{stem}.mkv"
        temp_path = directory / f".{stem}.mkv.part"
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", FFMPEG_LOGLEVEL, "-nostdin",
            "-rtsp_transport", RTSP_TRANSPORT,
            "-stimeout", STIMEOUT_US,
            "-analyzeduration", ANALYZE_US,
            "-probesize", PROBESIZE,
            "-fflags", "+genpts",
            "-use_wallclock_as_timestamps", "1",
            "-i", url,
            "-t", str(SEGMENT_SEC),
            "-map", "0:v:0", "-an",
            "-c:v", "copy",
            "-reset_timestamps", "1",
            "-f", "matroska", "-y", str(temp_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
            )
            _out, stderr = self._proc.communicate(timeout=SEGMENT_SEC + 30)
            rc = self._proc.returncode
        except subprocess.TimeoutExpired:
            self._stop_ffmpeg()
            stderr, rc = "ffmpeg timeout", 124
        finally:
            self._proc = None

        if self.stop_event.is_set() and rc != 0:
            temp_path.unlink(missing_ok=True)
            return
        if rc == 0 and temp_path.is_file() and temp_path.stat().st_size > 0:
            temp_path.replace(final_path)
            logger.info("ready cam=%s file=%s", self.camera_id, final_path.name)
        else:
            temp_path.unlink(missing_ok=True)
            tail = " | ".join((stderr or "").strip().splitlines()[-3:])
            logger.warning("failed cam=%s rc=%s %s", self.camera_id, rc, tail)


# ---------------------------------------------------------------------------
# Manager: đồng bộ tập recorder theo discover
# ---------------------------------------------------------------------------
class RecorderManager:
    def __init__(self) -> None:
        self._recorders: dict[str, CameraRecorder] = {}

    def reconcile(self) -> None:
        """Khớp tập recorder đang chạy với kết quả discover."""
        targets = _discover_targets()
        # Dừng camera không còn trong target (hết tiết / tắt AI).
        for cam in list(self._recorders):
            if cam not in targets:
                logger.info("drop cam=%s (hết điều kiện quay)", cam)
                self._recorders.pop(cam).stop()
        # Bật camera mới xuất hiện.
        for cam, cls in targets.items():
            rec = self._recorders.get(cam)
            if rec is None:
                rec = CameraRecorder(cam, cls)
                self._recorders[cam] = rec
                rec.start()
            elif rec.class_id != cls:
                # Camera đổi lớp (tiết mới, lớp khác) → restart để đổi RTSP/class.
                logger.info("cam=%s đổi class %s→%s, restart", cam, rec.class_id, cls)
                rec.stop()
                new = CameraRecorder(cam, cls)
                self._recorders[cam] = new
                new.start()

    def stop_all(self) -> None:
        for rec in list(self._recorders.values()):
            rec.stop()
        self._recorders.clear()


def run_loop(stop_event: threading.Event) -> None:
    mgr = RecorderManager()
    logger.info(
        "video_recorder started: root=%s segment=%ss discover=%ss in_period_only=%s",
        VIDEO_EXTS_ROOT, SEGMENT_SEC, DISCOVER_INTERVAL_SEC, RECORD_IN_PERIOD_ONLY,
    )
    try:
        while not stop_event.is_set():
            try:
                mgr.reconcile()
            except Exception:  # noqa: BLE001 — 1 vòng lỗi không giết tiến trình
                logger.exception("reconcile failed")
            stop_event.wait(DISCOVER_INTERVAL_SEC)
    finally:
        mgr.stop_all()
        logger.info("video_recorder stopped")


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    stop_event = threading.Event()

    def _stop(*_a):
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _stop)

    run_loop(stop_event)


if __name__ == "__main__":
    main()
