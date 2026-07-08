#!/usr/bin/env python3
"""Browser UI for recorded video segments and the existing QHH AI logic."""

from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import request as urlrequest
from urllib.parse import parse_qs, parse_qsl, quote, urlencode, urlparse, urlsplit, urlunsplit

import cv2
import numpy as np

from config_loader import env_or_config
from db import redis_client as db
from workers.camera_worker import AIDetectionWorker
from workers.video_reader import RecordedVideoReader, latest_recording_status


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web"
FACE_DIR = Path.home() / ".classroom_manager" / "faces"
MAX_FACE_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_JSON_REQUEST_BYTES = 12 * 1024 * 1024
BUILD_ID = "2026-06-25.video-record.8"
AI_EVENT_WEBHOOK_URL = os.getenv("QHH_AI_EVENT_WEBHOOK_URL", "").strip()
AI_EVENT_LOG = os.getenv("QHH_AI_EVENT_LOG", "").strip()


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _bool_setting(env_name: str, section: str, key: str, default=False) -> bool:
    return _truthy(env_or_config(env_name, section, key, default))


def _int_setting(env_name: str, section: str, key: str, default: int) -> int:
    return int(env_or_config(env_name, section, key, default))


def _float_setting(env_name: str, section: str, key: str, default: float) -> float:
    return float(env_or_config(env_name, section, key, default))


WEB_RECORD_ON_AI = _bool_setting(
    "QHH_WEB_RECORD_ON_AI", "web_record", "on_ai", True
)
try:
    AI_EVENT_MIN_INTERVAL = max(
        0.2, float(os.getenv("QHH_AI_EVENT_MIN_INTERVAL", "2.0"))
    )
except ValueError:
    AI_EVENT_MIN_INTERVAL = 2.0
try:
    WEB_RECORD_DURATION_SEC = max(
        1, _int_setting("QHH_WEB_RECORD_DURATION_SEC", "web_record", "duration_sec", 10)
    )
except ValueError:
    WEB_RECORD_DURATION_SEC = 10
try:
    WEB_RECORD_INTERVAL_SEC = max(
        WEB_RECORD_DURATION_SEC,
        _int_setting("QHH_WEB_RECORD_INTERVAL_SEC", "web_record", "interval_sec", 60),
    )
except ValueError:
    WEB_RECORD_INTERVAL_SEC = max(WEB_RECORD_DURATION_SEC, 60)
AI_RESULT_VIDEO_ON = _bool_setting(
    "QHH_AI_RESULT_VIDEO_ON", "ai", "result_video_on", True
)
try:
    AI_RESULT_VIDEO_FPS = max(
        1.0, _float_setting("QHH_AI_RESULT_VIDEO_FPS", "ai", "result_video_fps", 25.0)
    )
except ValueError:
    AI_RESULT_VIDEO_FPS = 25.0
try:
    AI_RESULT_VIDEO_QUEUE = max(
        1, _int_setting("QHH_AI_RESULT_VIDEO_QUEUE", "ai", "result_video_queue", 8)
    )
except ValueError:
    AI_RESULT_VIDEO_QUEUE = 8
AI_RESULT_VIDEO_CODEC = (
    str(env_or_config("QHH_AI_RESULT_VIDEO_CODEC", "ai", "result_video_codec", "mp4v"))
    or "mp4v"
)[:4]
AI_RESULT_VIDEO_EXT = (
    str(env_or_config("QHH_AI_RESULT_VIDEO_EXT", "ai", "result_video_ext", ".mp4"))
    .strip()
    or ".mp4"
)
if not AI_RESULT_VIDEO_EXT.startswith("."):
    AI_RESULT_VIDEO_EXT = "." + AI_RESULT_VIDEO_EXT


def _runtime_path(value: str | os.PathLike, default: Path) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = ROOT / path
    return path


AI_RESULT_VIDEO_DIR = _runtime_path(
    env_or_config("QHH_AI_RESULT_DIR", "ai", "result_dir", "result_test"),
    ROOT / "result_test",
)
AI_SNAPSHOT_ON = _bool_setting("QHH_AI_SNAPSHOT_ON", "ai", "snapshot_on", False)
AI_AUTO_START = _bool_setting("QHH_AI_AUTO_START", "ai", "auto_start", False)
WEB_LIVE_PREVIEW_ON = _bool_setting(
    "QHH_WEB_LIVE_PREVIEW_ON", "ai", "live_preview_on", False
)
AI_DELETE_PROCESSED_VIDEO = _bool_setting(
    "QHH_AI_DELETE_PROCESSED_VIDEO", "ai", "delete_processed_video", False
)


def _safe_camera_id(camera_id: str) -> str:
    return "".join(
        ch for ch in str(camera_id) if ch.isalnum() or ch in "-_"
    ) or "camera"


# RTSP helpers đã tách ra web/rtsp.py để video_recorder dùng chung. Giữ alias
# nội bộ để phần code cũ trong file này không phải đổi tên gọi.
from web.rtsp import build_camera_rtsp_url, force_mainstream_rtsp as _force_mainstream_rtsp


def _camera_rtsp_url_for_class(camera_id: str, class_id: str) -> str:
    """Wrapper giữ tên gọi cũ — logic thật ở web/rtsp.build_camera_rtsp_url."""
    return build_camera_rtsp_url(camera_id, class_id)


class WebOnDemandRecorder:
    """Record the selected camera while the test AI monitor is active."""

    def __init__(self, camera_id: str, class_id: str, stop_event: threading.Event):
        self.camera_id = str(camera_id)
        self.class_id = str(class_id)
        self.stop_event = stop_event
        self.root = RecordedVideoReader(self.camera_id).root
        self.url = _camera_rtsp_url_for_class(self.camera_id, self.class_id)
        self.thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None

    def start(self):
        if not WEB_RECORD_ON_AI:
            return
        if not self.url:
            raise RuntimeError("Camera chưa có RTSP URL để record")
        parsed = urlparse(self.url)
        print(
            f"[web-record] source cam={self.camera_id} "
            f"path={parsed.path}?{parsed.query}",
            flush=True,
        )
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"qhh-web-record-{_safe_camera_id(self.camera_id)[:12]}",
        )
        self.thread.start()

    def stop(self):
        self._stop_ffmpeg()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def _run(self):
        print(
            f"[web-record] start cam={self.camera_id} class={self.class_id} "
            f"duration={WEB_RECORD_DURATION_SEC}s interval={WEB_RECORD_INTERVAL_SEC}s",
            flush=True,
        )
        while not self.stop_event.is_set():
            started = time.monotonic()
            self._record_once()
            sleep_for = WEB_RECORD_INTERVAL_SEC - (time.monotonic() - started)
            if sleep_for > 0 and self.stop_event.wait(sleep_for):
                break
        self._stop_ffmpeg()
        print(f"[web-record] stop cam={self.camera_id}", flush=True)

    def _record_once(self):
        now = datetime.now()
        directory = self.root / _safe_camera_id(self.camera_id) / now.strftime("%Y%m%d")
        directory.mkdir(parents=True, exist_ok=True)
        self._cleanup_stale_parts(directory)
        stem = now.strftime("%Y%m%d_%H%M%S_web")
        final_path = directory / f"{stem}.mkv"
        temp_path = directory / f".{stem}.mkv.part"
        cmd = [
            os.getenv("FFMPEG_BIN", "ffmpeg"),
            "-hide_banner",
            "-loglevel",
            os.getenv("QHH_WEB_RECORD_FFMPEG_LOGLEVEL", "warning"),
            "-nostdin",
            "-rtsp_transport",
            os.getenv("QHH_WEB_RECORD_RTSP_TRANSPORT", "tcp"),
            "-stimeout",
            os.getenv(
                "QHH_WEB_RECORD_STIMEOUT_US",
                os.getenv("QHH_WEB_RECORD_RW_TIMEOUT_US", "15000000"),
            ),
            "-analyzeduration",
            os.getenv("QHH_WEB_RECORD_ANALYZE_US", "10000000"),
            "-probesize",
            os.getenv("QHH_WEB_RECORD_PROBESIZE", "10000000"),
            "-fflags",
            "+genpts",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            self.url,
            "-t",
            str(WEB_RECORD_DURATION_SEC),
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-reset_timestamps",
            "1",
            "-f",
            "matroska",
            "-y",
            str(temp_path),
        ]
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            _stdout, stderr = self._proc.communicate(
                timeout=WEB_RECORD_DURATION_SEC + 30
            )
            rc = self._proc.returncode
        except subprocess.TimeoutExpired:
            self._stop_ffmpeg()
            stderr = "ffmpeg timeout"
            rc = 124
        finally:
            self._proc = None

        if self.stop_event.is_set() and rc != 0:
            try:
                temp_path.unlink()
            except OSError:
                pass
            return

        if rc == 0 and temp_path.is_file() and temp_path.stat().st_size > 0:
            temp_path.replace(final_path)
            print(f"[web-record] ready cam={self.camera_id} file={final_path}", flush=True)
        else:
            try:
                temp_path.unlink()
            except OSError:
                pass
            tail = " | ".join((stderr or "").strip().splitlines()[-3:])
            print(f"[web-record] failed cam={self.camera_id} rc={rc} {tail}", flush=True)

    def _cleanup_stale_parts(self, directory: Path):
        max_age_sec = max(WEB_RECORD_DURATION_SEC + 60, 120)
        cutoff = time.time() - max_age_sec
        for part in directory.glob(".*.mkv.part"):
            try:
                if part.stat().st_mtime < cutoff:
                    part.unlink()
                    print(f"[web-record] cleanup stale temp file={part}", flush=True)
            except OSError:
                pass

    def _stop_ffmpeg(self):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class AIResultSegmentWriter:
    """Write one annotated result video per processed input segment."""

    def __init__(self, camera_id: str, class_id: str):
        self.enabled = bool(AI_RESULT_VIDEO_ON)
        self.camera_id = _safe_camera_id(camera_id)
        self.class_id = _safe_camera_id(class_id)
        self.fps = float(AI_RESULT_VIDEO_FPS)
        self._queue: queue.Queue[tuple[str, Path | None, np.ndarray | None]] = queue.Queue(
            maxsize=AI_RESULT_VIDEO_QUEUE
        )
        self._thread: threading.Thread | None = None
        self._drops = 0
        self._last_result = ""

    def start(self):
        if not self.enabled:
            return
        AI_RESULT_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"qhh-ai-result-{self.camera_id[:12]}",
        )
        self._thread.start()
        print(f"[ai-result] start dir={AI_RESULT_VIDEO_DIR}", flush=True)

    @property
    def last_result(self) -> str:
        return self._last_result

    def write(self, segment_path: Path | None, frame: np.ndarray):
        if not self.enabled or self._thread is None:
            return
        if segment_path is None:
            return
        try:
            self._queue.put_nowait(("frame", Path(segment_path), frame.copy()))
        except queue.Full:
            self._drops += 1

    def finish_segment(self, segment_path: Path):
        if not self.enabled or self._thread is None:
            return
        try:
            self._queue.put(("finish", Path(segment_path), None), timeout=2.0)
        except queue.Full:
            self._drops += 1

    def stop(self):
        if not self.enabled or self._thread is None:
            return
        try:
            self._queue.put(("stop", None, None), timeout=2.0)
        except queue.Full:
            self._drops += 1
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            print("[ai-result] writer still stopping", flush=True)

    def _run(self):
        writer = None
        current_segment: Path | None = None
        final_path: Path | None = None
        temp_path: Path | None = None
        skipped_segment: Path | None = None
        frames = 0

        def paths_for(segment_path: Path, directory: Path, ext: str) -> tuple[Path, Path]:
            stem = segment_path.stem
            final = directory / f"{stem}_ai_result{ext}"
            temp = directory / f".{stem}_ai_result.part{ext}"
            return final, temp

        def writer_candidates() -> list[tuple[str, str]]:
            candidates = [
                (AI_RESULT_VIDEO_CODEC, AI_RESULT_VIDEO_EXT),
                ("mp4v", ".mp4"),
                ("MJPG", ".avi"),
                ("XVID", ".avi"),
            ]
            out = []
            seen = set()
            for codec, ext in candidates:
                codec = (str(codec or "mp4v")[:4] or "mp4v")
                if len(codec) != 4:
                    codec = "mp4v"
                ext = str(ext or ".mp4").strip() or ".mp4"
                if not ext.startswith("."):
                    ext = "." + ext
                key = (codec, ext.lower())
                if key not in seen:
                    out.append((codec, ext))
                    seen.add(key)
            return out

        def can_write(directory: Path) -> bool:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                probe = directory / f".{self.camera_id[:12]}.write-test"
                with probe.open("wb") as handle:
                    handle.write(b"ok")
                probe.unlink()
                return True
            except OSError as exc:
                print(
                    f"[ai-result] output dir not writable dir={directory} err={exc}",
                    flush=True,
                )
                return False

        def prepare_frame(frame: np.ndarray) -> np.ndarray | None:
            if frame is None or frame.size == 0:
                return None
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim != 3 or frame.shape[2] != 3:
                return None
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            return np.ascontiguousarray(frame)

        def open_writer(segment_path: Path, frame: np.ndarray):
            nonlocal final_path, temp_path
            frame = prepare_frame(frame)
            if frame is None:
                print("[ai-result] skip frame with unsupported shape", flush=True)
                return None
            height, width = frame.shape[:2]
            directory = AI_RESULT_VIDEO_DIR / self.camera_id / segment_path.parent.name
            if not can_write(directory):
                return None
            for codec, ext in writer_candidates():
                final_path, temp_path = paths_for(segment_path, directory, ext)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                candidate = cv2.VideoWriter(
                    str(temp_path), fourcc, self.fps, (width, height)
                )
                if candidate.isOpened():
                    print(
                        f"[ai-result] writer opened codec={codec} file={temp_path}",
                        flush=True,
                    )
                    return candidate
                candidate.release()
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            print(
                f"[ai-result] cannot open writer codec={AI_RESULT_VIDEO_CODEC} "
                f"size={width}x{height} dir={directory}",
                flush=True,
            )
            final_path = None
            temp_path = None
            return None

        def close_current():
            nonlocal writer, current_segment, final_path, temp_path, frames
            if writer is not None:
                writer.release()
                writer = None
            if final_path is not None and temp_path is not None:
                if temp_path.is_file() and temp_path.stat().st_size > 0:
                    temp_path.replace(final_path)
                    self._last_result = str(final_path)
                    print(
                        f"[ai-result] ready file={final_path} "
                        f"frames={frames} drops={self._drops}",
                        flush=True,
                    )
                else:
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
            current_segment = None
            final_path = None
            temp_path = None
            frames = 0

        try:
            while True:
                action, segment_path, frame = self._queue.get()
                if action == "stop":
                    break
                if action == "finish":
                    if segment_path is not None and skipped_segment == segment_path:
                        skipped_segment = None
                        continue
                    if segment_path is not None and current_segment == segment_path:
                        close_current()
                    continue
                if action != "frame" or segment_path is None or frame is None:
                    continue
                if skipped_segment == segment_path:
                    continue
                if current_segment != segment_path:
                    close_current()
                    current_segment = segment_path
                if writer is None:
                    writer = open_writer(segment_path, frame)
                    if writer is None:
                        skipped_segment = segment_path
                        close_current()
                        continue
                prepared = prepare_frame(frame)
                if prepared is None:
                    continue
                writer.write(prepared)
                frames += 1
        finally:
            close_current()


def _emit_ai_event(payload: dict):
    """Optional integration hook for the future production web/backend."""
    if not AI_EVENT_WEBHOOK_URL and not AI_EVENT_LOG:
        return

    def _send():
        body = json.dumps(payload, ensure_ascii=False, default=_json_default)
        if AI_EVENT_LOG:
            try:
                path = Path(AI_EVENT_LOG).expanduser()
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(body + "\n")
            except OSError as exc:
                print(f"[ai-event] log failed: {exc}", flush=True)
        if AI_EVENT_WEBHOOK_URL:
            try:
                req = urlrequest.Request(
                    AI_EVENT_WEBHOOK_URL,
                    data=body.encode("utf-8"),
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                with urlrequest.urlopen(req, timeout=2.0) as response:
                    response.read(256)
            except Exception as exc:
                print(f"[ai-event] webhook failed: {exc}", flush=True)

    threading.Thread(target=_send, daemon=True, name="qhh-ai-event").start()


class _NoopAbsenceTracker:
    def prune(self, active_keys: set[tuple]) -> None:
        return


def _model_runtime_summary(bundle=None, yolo_device=None) -> str:
    summary = getattr(AIDetectionWorker, "_model_runtime_summary", None)
    if callable(summary):
        try:
            return summary(bundle, yolo_device)
        except TypeError:
            return summary(bundle)

    yolo, retinaface, arcface, gaze = bundle or (None, None, None, None)

    def _providers(model) -> str:
        sess = getattr(model, "sess", None) or getattr(model, "session", None)
        if sess is None:
            return "disabled"
        try:
            return ",".join(sess.get_providers())
        except Exception as exc:
            return f"unknown({exc})"

    yolo_actual = "unknown"
    try:
        yolo_actual = str(next(yolo.model.parameters()).device)
    except Exception:
        pass
    return (
        f"yolo_device_arg={yolo_device if yolo_device is not None else 'auto'}; "
        f"yolo_actual={yolo_actual}; "
        f"retina={_providers(retinaface)}; "
        f"arcface={_providers(arcface)}; "
        f"gaze={_providers(gaze)}"
    )


class WebDetectionEngine:
    """Qt-free adapter around the existing detection implementation.

    The desktop worker inherits QThread. Constructing and destroying that
    QObject inside temporary HTTP worker threads can corrupt native Qt/ONNX
    memory. This adapter reuses its Python detection methods and shared models
    without creating a QObject.
    """

    FACE_MATCH_THRESHOLD = AIDetectionWorker.FACE_MATCH_THRESHOLD
    DESK_CM_W = AIDetectionWorker.DESK_CM_W
    DESK_CM_D = AIDetectionWorker.DESK_CM_D
    TARGET_CORNERS = AIDetectionWorker.TARGET_CORNERS
    _INFERENCE_LOCK = AIDetectionWorker._INFERENCE_LOCK

    def __init__(self):
        self._yolo, self._retinaface, self._arcface, self._gaze = (
            AIDetectionWorker._shared_models()
        )
        try:
            import torch  # type: ignore
            self._yolo_device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self._yolo_device = "cpu"
        from workers.gaze_estimator import DistractionTracker
        from workers.camera_worker import _FpsAggregator, FPS_LOG_PERIOD
        self._fps = _FpsAggregator(FPS_LOG_PERIOD, "ai-web")
        self._fps_yolo = self._fps.get("yolo")
        self._fps_retina = self._fps.get("retina")
        self._fps_arc = self._fps.get("arc")
        self._fps_gaze = self._fps.get("gaze")
        self._fps_pipeline = self._fps.get("pipe")
        self._absence = _NoopAbsenceTracker()
        self._distraction = DistractionTracker(
            yaw_threshold_deg=float(os.getenv("GAZE_YAW_THRESHOLD", "60")),
            # Pitch disabled — student nhìn xuống vở/sách hay nhìn lên bảng
            # KHÔNG phải mất tập trung.
            pitch_threshold_deg=float(os.getenv("GAZE_PITCH_THRESHOLD", "9999")),
            alert_after_sec=float(os.getenv("GAZE_ALERT_AFTER", "2.5")),
            clear_after_sec=float(os.getenv("GAZE_CLEAR_AFTER", "1.0")),
            ema_alpha=float(os.getenv("GAZE_EMA_ALPHA", "0.35")),
        )
        self._seats = []
        self._students = {}
        self._seat_lookup = {}
        self._face_gallery = []
        self._gallery_embeddings = np.empty(
            (0, 512), dtype=np.float32
        )
        self._ground_transform = None
        self._zone_ground_polys = {}
        self._calibration_dirty = True
        self._detection_mode = "centerpoint"

    def __getattr__(self, name):
        descriptor = inspect.getattr_static(AIDetectionWorker, name, None)
        if isinstance(descriptor, staticmethod):
            return descriptor.__func__
        if isinstance(descriptor, classmethod):
            return descriptor.__get__(None, AIDetectionWorker)
        if callable(descriptor):
            return descriptor.__get__(self, type(self))
        raise AttributeError(name)

    def set_detection_mode(self, mode: str):
        self._detection_mode = (
            mode if mode in {"centerpoint", "perspective"} else "centerpoint"
        )

    def update_context(self, seats: list[dict], students: list[dict]):
        students_by_id = {
            str(student.get("id", "")): dict(student)
            for student in students if student.get("id")
        }
        seat_lookup = {}
        for seat in seats:
            desk_num = int(seat.get("desk_num", 0) or 0)
            for slot in seat.get("slots", []) or []:
                student_id = str(slot.get("student_id", "") or "")
                if student_id:
                    seat_lookup[student_id] = (
                        desk_num, int(slot.get("slot_num", 1) or 1)
                    )
        gallery = self._build_face_gallery(list(students_by_id.values()))
        self._seats = seats
        self._students = students_by_id
        self._seat_lookup = seat_lookup
        self._face_gallery = gallery
        self._gallery_embeddings = (
            np.stack([item["embedding"] for item in gallery]).astype(
                np.float32, copy=False
            )
            if gallery else np.empty((0, 512), dtype=np.float32)
        )
        active_slot_keys = {
            (
                int(seat.get("desk_num", 0) or 0),
                int(slot.get("slot_num", 1) or 1),
            )
            for seat in seats for slot in (seat.get("slots") or [])
            if str(slot.get("student_id", "") or "")
        }
        self._absence.prune(active_slot_keys)
        self._calibration_dirty = True


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


class WebAIMonitor:
    """Runs the existing detector against completed video segments."""

    def __init__(self):
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._detector_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._detector: WebDetectionEngine | None = None
        self._generation = 0
        self._state = self._empty_state()
        self._snapshot_jpeg: bytes | None = None
        self._last_event_at = 0.0
        self._preload_thread = threading.Thread(
            target=self._preload_detector,
            daemon=True,
            name="qhh-ai-preload",
        )
        self._preload_thread.start()

    @staticmethod
    def _empty_state() -> dict:
        return {
            "active": False,
            "requested": False,
            "loading": False,
            "class_id": "",
            "camera_id": "",
            "mode": "centerpoint",
            "status": "AI chưa chạy",
            "error": "",
            "sequence": 0,
            "frame_count": None,
            "updated_at": 0.0,
            "inference_ms": None,
            "recording": False,
            "result_video": "",
            "results": [],
        }

    def status(self) -> dict:
        with self._lock:
            state = dict(self._state)
        thread = self._thread
        state["running"] = bool(thread and thread.is_alive())
        return state

    def snapshot(self) -> bytes | None:
        with self._lock:
            return self._snapshot_jpeg

    def start(self, class_id: str, camera_id: str, mode: str = "centerpoint"):
        with self._lifecycle_lock:
            mode = mode if mode in {"centerpoint", "perspective"} else "centerpoint"
            with self._lock:
                same_session = (
                    self._state.get("class_id") == class_id
                    and self._state.get("camera_id") == camera_id
                    and self._state.get("mode") == mode
                    and (self._state.get("active") or self._state.get("loading"))
                )
            if same_session and self._thread and self._thread.is_alive():
                return False
            self.stop()
            self._generation += 1
            generation = self._generation
            self._stop_event = threading.Event()
            with self._lock:
                self._state = self._empty_state() | {
                    "loading": True,
                    "requested": True,
                    "class_id": class_id,
                    "camera_id": camera_id,
                    "mode": mode,
                    "status": (
                        "Đang chuẩn bị AI"
                        if self._detector is not None
                        else "Đang tải mô hình AI lần đầu"
                    ),
                }
                self._snapshot_jpeg = None
            self._thread = threading.Thread(
                target=self._run,
                args=(generation, class_id, camera_id, mode, self._stop_event),
                daemon=True,
                name="qhh-web-ai",
            )
            self._thread.start()
            return True

    def stop(self):
        with self._lifecycle_lock:
            self._generation += 1
            self._stop_event.set()
            thread = self._thread
            if thread and thread.is_alive() and thread is not threading.current_thread():
                with self._lock:
                    self._state.update({
                        "loading": False,
                        "status": "Đang dừng AI sau lượt suy luận hiện tại",
                    })
                # Native inference cannot be cancelled safely midway. Wait for
                # the current call to return before releasing video/thread state.
                thread.join()
            self._thread = None
            with self._lock:
                self._state = self._empty_state()
                self._snapshot_jpeg = None

    def _get_detector(self) -> WebDetectionEngine:
        with self._detector_lock:
            if self._detector is None:
                self._detector = WebDetectionEngine()
            return self._detector

    def _preload_detector(self):
        started = time.monotonic()
        try:
            detector = self._get_detector()
            students = db.list_students()
            if students:
                # Fill the immutable embedding cache before the first Start AI
                # request, so registration images do not delay activation.
                detector._build_face_gallery(students)
            print(
                "[ai] models ready in %.1fs, face profiles=%s; %s"
                % (
                    time.monotonic() - started,
                    len(students),
                    _model_runtime_summary(
                        (
                            detector._yolo,
                            detector._retinaface,
                            detector._arcface,
                            detector._gaze,
                        ),
                        detector._yolo_device,
                    ),
                ),
                flush=True,
            )
        except Exception as exc:
            # A later Start AI request will retry and expose the actual error
            # through the normal AI status payload.
            print("[ai] model preload failed: %s" % exc, flush=True)

    def _run(
        self,
        generation: int,
        class_id: str,
        camera_id: str,
        mode: str,
        stop_event: threading.Event,
    ):
        reader = None
        recorder = None
        result_writer = None
        try:
            session_started_at = time.time()
            seats = db.monitor_seats(camera_id, class_id)
            if not any(seat.get("zone") for seat in seats):
                raise RuntimeError("Camera và lớp này chưa có vùng bàn")
            students = self._students_for_ai(class_id, seats)
            detector = self._get_detector()
            detector.set_detection_mode(mode)
            detector.update_context(seats, students)
            recorder = WebOnDemandRecorder(camera_id, class_id, stop_event)
            recorder.start()
            result_writer = AIResultSegmentWriter(camera_id, class_id)
            result_writer.start()
            reader = RecordedVideoReader(
                camera_id,
                start_after=session_started_at,
                delete_after_read=AI_DELETE_PROCESSED_VIDEO,
                on_segment_done=result_writer.finish_segment,
            )

            with self._lock:
                if generation != self._generation:
                    return
                self._state.update({
                    "active": True,
                    "requested": True,
                    "loading": False,
                    "status": "AI đang chạy, đang chờ video record đầu tiên",
                    "recording": bool(WEB_RECORD_ON_AI),
                    "result_video": result_writer.last_result
                    if result_writer.enabled else "",
                    "error": "",
                })

            # Re-read seats + student assignments from Redis on this cadence so
            # moving a student between classes (or editing seat layout) takes
            # effect without restarting AI.
            try:
                context_refresh_sec = max(
                    1.0, float(os.getenv("AI_CONTEXT_REFRESH_SEC", "5"))
                )
            except ValueError:
                context_refresh_sec = 5.0
            last_context_refresh = time.monotonic()

            from workers.camera_worker import _FpsAggregator, FPS_LOG_PERIOD
            fps_stream = _FpsAggregator(FPS_LOG_PERIOD, f"stream cam={camera_id}")
            stream_meter = fps_stream.get("read")

            last_count = None
            while not stop_event.is_set() and generation == self._generation:
                if time.monotonic() - last_context_refresh >= context_refresh_sec:
                    try:
                        fresh_seats = db.monitor_seats(camera_id, class_id)
                        fresh_students = self._students_for_ai(class_id, fresh_seats)
                        detector.update_context(fresh_seats, fresh_students)
                    except Exception as exc:
                        print(f"[ai] context refresh failed: {exc}", flush=True)
                    last_context_refresh = time.monotonic()

                _read_t0 = time.monotonic()
                try:
                    item = reader.read()
                except Exception as exc:
                    print(
                        f"[ai] video read failed cam={camera_id}: {exc}",
                        flush=True,
                    )
                    with self._lock:
                        if generation != self._generation:
                            return
                        self._state.update({
                            "active": True,
                            "requested": True,
                            "loading": False,
                            "status": "AI đang chạy, chờ segment video hợp lệ",
                            "error": str(exc),
                        })
                    time.sleep(0.5)
                    continue
                if item is None:
                    time.sleep(0.02)
                    continue
                stream_meter.tick((time.monotonic() - _read_t0) * 1000.0)
                fps_stream.maybe_flush()
                frame_count, frame = item
                if frame_count == last_count:
                    time.sleep(0.01)
                    continue
                last_count = frame_count
                started = time.monotonic()
                try:
                    annotated, results = detector._detect(
                        frame,
                        list(detector._seats),
                        dict(detector._students),
                        dict(detector._seat_lookup),
                        list(detector._face_gallery),
                        detector._calibration_dirty,
                        mode,
                    )
                    detector._calibration_dirty = False
                    if result_writer is not None:
                        result_writer.write(reader.current_path, annotated)
                    if AI_SNAPSHOT_ON:
                        ok, encoded = cv2.imencode(
                            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 84]
                        )
                    else:
                        ok, encoded = False, None
                    elapsed_ms = round((time.monotonic() - started) * 1000)
                except Exception as exc:
                    print(
                        f"[ai] inference failed cam={camera_id} frame={frame_count}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                    with self._lock:
                        if generation != self._generation:
                            return
                        self._state.update({
                            "active": True,
                            "requested": True,
                            "loading": False,
                            "status": "AI đang chạy, bỏ qua frame lỗi",
                            "error": str(exc),
                        })
                    time.sleep(0.1)
                    continue
                with self._lock:
                    if generation != self._generation:
                        return
                    self._state.update({
                        "active": True,
                        "requested": True,
                        "loading": False,
                        "status": (
                            "AI đang chạy, web đang record video test"
                            if WEB_RECORD_ON_AI else "AI đang chạy"
                        ),
                        "recording": bool(WEB_RECORD_ON_AI),
                        "sequence": int(self._state["sequence"]) + 1,
                        "frame_count": int(frame_count),
                        "updated_at": time.time(),
                        "inference_ms": elapsed_ms,
                        "result_video": result_writer.last_result
                        if result_writer and result_writer.enabled else "",
                        "results": results,
                    })
                    if AI_SNAPSHOT_ON:
                        self._snapshot_jpeg = encoded.tobytes() if ok else None
        except Exception as exc:
            print(f"[ai] worker failed cam={camera_id}: {exc}", flush=True)
            traceback.print_exc()
            with self._lock:
                if generation == self._generation:
                    self._state.update({
                        "active": False,
                        "requested": False,
                        "loading": False,
                        "status": "AI gặp lỗi",
                        "error": str(exc),
                    })
        finally:
            if reader is not None:
                reader.close()
            if recorder is not None:
                recorder.stop()
            if result_writer is not None:
                result_writer.stop()

    def _maybe_emit_event(
        self,
        class_id: str,
        camera_id: str,
        mode: str,
        frame_count: int,
        inference_ms: int,
        results: list[dict],
    ):
        now = time.time()
        if now - self._last_event_at < AI_EVENT_MIN_INTERVAL:
            return
        self._last_event_at = now
        _emit_ai_event({
            "eventType": "qhh.ai.attendance.snapshot",
            "version": 1,
            "timestamp": now,
            "classId": class_id,
            "cameraId": camera_id,
            "mode": mode,
            "frameCount": frame_count,
            "inferenceMs": inference_ms,
            "results": results,
        })

    @staticmethod
    def _students_for_ai(class_id: str, seats: list[dict]) -> list[dict]:
        # `list_students(class_id)` is the source of truth. A seat slot may
        # still reference a student who has moved to another class — those
        # must NOT be re-added, otherwise the gallery never forgets them.
        students = {s["id"]: s for s in db.list_students(class_id)}
        target = str(class_id or "")
        for seat in seats:
            for slot in seat.get("slots", []):
                sid = str(slot.get("student_id", "") or "")
                if not sid or sid in students:
                    continue
                student = db.get_student(sid)
                if student and str(student.get("class_id") or "") == target:
                    students[sid] = student
        return list(students.values())


AI_MONITOR = WebAIMonitor()


def _camera_payload(camera: dict) -> dict:
    camera_id = str(camera.get("id", ""))
    recording = latest_recording_status(camera_id)
    return {
        "id": camera_id,
        "name": camera.get("name", camera_id),
        "class_ids": camera.get("class_ids", []),
        "stream_id": camera_id,
        "stream_key": "",
        "stream_ready": recording["ready"],
        "recording_latest": recording["latest"],
        "recording_updated_at": recording["updated_at"],
    }


def _bootstrap(class_id: str = "", camera_id: str = "") -> dict:
    classrooms = db.list_classrooms()
    cameras = [_camera_payload(camera) for camera in db.list_cameras()]
    if not class_id and classrooms:
        class_id = str(classrooms[0].get("id", ""))
    if not camera_id:
        candidates = [
            camera for camera in cameras
            if not class_id or class_id in camera.get("class_ids", [])
        ]
        if candidates:
            camera_id = candidates[0]["id"]
    seats = (
        db.monitor_seats(camera_id, class_id)
        if class_id and camera_id
        else []
    )
    students = {
        student["id"]: {
            "id": student["id"],
            "name": student.get("name", ""),
            "student_code": student.get("student_code", ""),
        }
        for student in db.list_students()
    }
    return {
        "classrooms": classrooms,
        "cameras": [
            {key: value for key, value in camera.items() if key != "password"}
            for camera in cameras
        ],
        "class_id": class_id,
        "camera_id": camera_id,
        "seats": seats,
        "students": students,
        "recording_root": str(RecordedVideoReader(camera_id).root) if camera_id else "",
        "settings": {
            "ai_auto_start": AI_AUTO_START,
            "live_preview_on": WEB_LIVE_PREVIEW_ON,
            "snapshot_on": AI_SNAPSHOT_ON,
            "web_record_on_ai": WEB_RECORD_ON_AI,
            "delete_processed_video": AI_DELETE_PROCESSED_VIDEO,
            "result_dir": str(AI_RESULT_VIDEO_DIR),
        },
        "redis_ok": db.ping(),
    }


def _management_payload() -> dict:
    classrooms = db.list_classrooms()
    cameras = db.list_cameras()
    students = db.list_students()
    student_counts = {}
    placements = {}
    for student in students:
        class_id = str(student.get("class_id", "") or "")
        if class_id:
            student_counts[class_id] = student_counts.get(class_id, 0) + 1
    for classroom in classrooms:
        class_id = str(classroom.get("id", "") or "")
        for seat in db.list_seats(class_id):
            for slot in seat.get("slots", []):
                student_id = str(slot.get("student_id", "") or "")
                if student_id:
                    placements[student_id] = {
                        "desk_num": int(seat.get("desk_num", 0) or 0),
                        "slot_num": int(slot.get("slot_num", 1) or 1),
                    }
    return {
        "classrooms": [
            {
                **classroom,
                "student_count": student_counts.get(str(classroom.get("id", "")), 0),
            }
            for classroom in classrooms
        ],
        "cameras": [
            {key: value for key, value in camera.items() if key != "password"}
            for camera in cameras
        ],
        "students": [
            {
                **student,
                **placements.get(str(student.get("id", "")), {}),
                "has_face": bool(
                    student.get("face_image")
                    and Path(str(student.get("face_image"))).is_file()
                ),
            }
            for student in students
        ],
    }


def _save_student_face(student_id: str, data_url: str) -> str:
    if not data_url or "," not in data_url:
        raise ValueError("Dữ liệu ảnh khuôn mặt không hợp lệ")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Chỉ chấp nhận file ảnh")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("Không giải mã được ảnh khuôn mặt") from exc
    if not image_bytes or len(image_bytes) > MAX_FACE_UPLOAD_BYTES:
        raise ValueError("Ảnh khuôn mặt phải nhỏ hơn 8 MB")

    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File ảnh không hợp lệ")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    ) if not detector.empty() else []
    if len(faces):
        x, y, width, height = max(faces, key=lambda rect: rect[2] * rect[3])
        margin_x, margin_y = int(width * 0.25), int(height * 0.30)
        x1, y1 = max(0, x - margin_x), max(0, y - margin_y)
        x2 = min(image.shape[1], x + width + margin_x)
        y2 = min(image.shape[0], y + height + margin_y)
        image = image[y1:y2, x1:x2]

    max_side = 640
    height, width = image.shape[:2]
    scale = min(max_side / max(width, height), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    FACE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(
        character for character in student_id
        if character.isalnum() or character in "-_"
    ) or "student"
    destination = FACE_DIR / f"{safe_id}_{int(time.time())}.jpg"
    if not cv2.imwrite(
        str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    ):
        raise RuntimeError("Không thể lưu ảnh khuôn mặt")
    return str(destination)


def _remove_managed_face(path_value: str):
    try:
        path = Path(str(path_value or "")).resolve()
        if path.is_file() and FACE_DIR.resolve() in path.parents:
            path.unlink()
    except OSError:
        pass


class WebHTTPServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        error = sys.exc_info()[1]
        if isinstance(
            error,
            (BrokenPipeError, ConnectionResetError, ConnectionAbortedError),
        ):
            return
        super().handle_error(request, client_address)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "QHHWeb/1.0"

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Last-resort guard for disconnects outside the response helper.
            pass

    def log_message(self, fmt, *args):
        message = fmt % args
        if (
            '"GET /api/ai/status ' in message
            or '"GET /api/ai/snapshot.jpg' in message
        ):
            return
        print(f"[web] {self.address_string()} {message}", flush=True)

    def _send(self, body: bytes, content_type: str, status=HTTPStatus.OK):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-QHH-Build", BUILD_ID)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # Browsers cancel an older snapshot request when a newer AI frame
            # becomes available or when the tab/view changes. The response
            # socket is already gone, so there is nothing left to send.
            return False
        return True

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(
            payload, ensure_ascii=False, default=_json_default
        ).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > MAX_JSON_REQUEST_BYTES:
            raise ValueError("Dữ liệu gửi lên vượt quá 12 MB")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                return self._static("index.html")
            if parsed.path.startswith("/static/"):
                return self._static(parsed.path.removeprefix("/static/"))
            if parsed.path == "/api/bootstrap":
                return self._json(_bootstrap(
                    query.get("class_id", [""])[0],
                    query.get("camera_id", [""])[0],
                ))
            if parsed.path == "/api/version":
                return self._json({"build": BUILD_ID})
            if parsed.path == "/api/management":
                return self._json(_management_payload())
            if parsed.path == "/api/ai/status":
                return self._json(AI_MONITOR.status())
            if parsed.path == "/api/ai/snapshot.jpg":
                image = AI_MONITOR.snapshot()
                if image is None:
                    return self._json(
                        {"error": "Chưa có khung AI"},
                        HTTPStatus.NOT_FOUND,
                    )
                return self._send(image, "image/jpeg")
            if parsed.path == "/api/students/face":
                student_id = query.get("id", [""])[0]
                student = db.get_student(student_id)
                face_path = Path(
                    str(student.get("face_image", "") or "")
                ).resolve()
                if (
                    not face_path.is_file()
                    or FACE_DIR.resolve() not in face_path.parents
                ):
                    return self._json(
                        {"error": "Người học chưa có ảnh khuôn mặt"},
                        HTTPStatus.NOT_FOUND,
                    )
                return self._send(face_path.read_bytes(), "image/jpeg")
            if parsed.path == "/api/recorded-stream.mjpg":
                camera_id = query.get("camera_id", [""])[0]
                fps = max(1, min(25, int(query.get("fps", ["12"])[0])))
                return self._send_recorded_stream(camera_id, fps)
            if parsed.path == "/api/stream-url":
                camera_id = query.get("camera_id", [""])[0]
                fps = max(1, min(25, int(query.get("fps", ["25"])[0])))
                if not db.get_camera(camera_id):
                    return self._json(
                        {"error": "Camera không tồn tại"},
                        HTTPStatus.NOT_FOUND,
                    )
                url = (
                    f"/api/recorded-stream.mjpg?"
                    f"camera_id={quote(str(camera_id), safe='')}&fps={fps}"
                )
                return self._json({
                    "url": url,
                    "stream_id": camera_id,
                    "recording": latest_recording_status(camera_id),
                })
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            return self._json(
                {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
            if parsed.path == "/api/classrooms":
                classroom_id = str(data.get("id", "") or "")
                name = str(data.get("name", "") or "").strip()
                num_desks = int(data.get("num_desks", 0))
                if not name or num_desks < 1:
                    raise ValueError("Tên lớp và số bàn là bắt buộc")
                classroom = (
                    db.update_classroom(classroom_id, name, num_desks)
                    if classroom_id
                    else db.create_classroom(name, num_desks)
                )
                return self._json({"ok": True, "classroom": classroom})
            if parsed.path == "/api/classrooms/delete":
                db.delete_classroom(str(data["id"]))
                return self._json({"ok": True})
            if parsed.path == "/api/cameras":
                camera_id = str(data.get("id", "") or "")
                class_ids = [
                    str(value) for value in data.get("class_ids", [])
                    if str(value)
                ]
                previous_camera = db.get_camera(camera_id) if camera_id else {}
                camera_data = {
                    key: data.get(key, previous_camera.get(key, ""))
                    for key in (
                        "name", "location", "brand", "model", "ipAddress",
                        "port", "onvifPort", "username", "password", "wsPort",
                        "streamUrl", "rtspPath", "notes", "isActive",
                    )
                }
                if camera_id and not camera_data.get("password"):
                    camera_data["password"] = db.get_camera(camera_id).get("password", "")
                if not str(camera_data["name"]).strip():
                    raise ValueError("Tên camera là bắt buộc")
                camera = (
                    db.update_camera(camera_id, camera_data)
                    if camera_id
                    else db.create_camera(camera_data)
                )
                camera_id = str(camera["id"])
                existing = set(db.get_camera(camera_id).get("class_ids", []))
                desired = set(class_ids)
                for class_id in desired - existing:
                    db.link_camera_class(camera_id, class_id)
                for class_id in existing - desired:
                    db.unlink_camera_class(camera_id, class_id)
                return self._json({"ok": True, "camera": db.get_camera(camera_id)})
            if parsed.path == "/api/cameras/delete":
                db.delete_camera(str(data["id"]))
                return self._json({"ok": True})
            if parsed.path == "/api/students":
                student_id = str(data.get("id", "") or "")
                name = str(data.get("name", "") or "").strip()
                student_code = str(data.get("student_code", "") or "").strip()
                class_id = str(data.get("class_id", "") or "")
                if not name or not student_code:
                    raise ValueError("Họ tên và mã người học là bắt buộc")
                student = (
                    db.update_student(student_id, name, student_code, class_id)
                    if student_id
                    else db.create_student(name, student_code, class_id)
                )
                desk_num = int(data.get("desk_num", 0) or 0)
                slot_num = int(data.get("slot_num", 1) or 1)
                if class_id and desk_num > 0:
                    db.assign_student_to_slot(
                        class_id, desk_num, max(1, slot_num), student["id"]
                    )
                old_face = str(db.get_student(student["id"]).get("face_image", "") or "")
                if data.get("remove_face"):
                    db.clear_student_face(student["id"])
                    _remove_managed_face(old_face)
                elif data.get("face_data"):
                    new_face = _save_student_face(
                        str(student["id"]), str(data["face_data"])
                    )
                    db.set_student_face(student["id"], new_face)
                    if old_face and old_face != new_face:
                        _remove_managed_face(old_face)
                return self._json({"ok": True, "student": db.get_student(student["id"])})
            if parsed.path == "/api/students/delete":
                student_id = str(data["id"])
                face_path = str(db.get_student(student_id).get("face_image", "") or "")
                db.delete_student(student_id)
                _remove_managed_face(face_path)
                return self._json({"ok": True})
            if parsed.path == "/api/zones":
                zone = db.set_desk_region(
                    str(data["camera_id"]),
                    str(data["class_id"]),
                    int(data["desk_num"]),
                    dict(data["zone"]),
                )
                return self._json({"ok": True, "config": zone})
            if parsed.path == "/api/zones/delete":
                config = db.clear_desk_region(
                    str(data["camera_id"]),
                    str(data["class_id"]),
                    int(data["desk_num"]),
                )
                return self._json({"ok": True, "config": config})
            if parsed.path == "/api/ai/start":
                started = AI_MONITOR.start(
                    str(data["class_id"]),
                    str(data["camera_id"]),
                    str(data.get("mode", "centerpoint")),
                )
                return self._json({
                    "ok": True,
                    "started": started,
                    "state": AI_MONITOR.status(),
                })
            if parsed.path == "/api/ai/stop":
                AI_MONITOR.stop()
                return self._json({"ok": True, "state": AI_MONITOR.status()})
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as exc:
            return self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            return self._json(
                {"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _static(self, relative: str):
        path = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in path.parents or not path.is_file():
            return self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        mime = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(path.suffix.lower(), "application/octet-stream")
        self._send(path.read_bytes(), mime)

    def _send_recorded_stream(self, camera_id: str, fps: int):
        if not db.get_camera(camera_id):
            return self._json({"error": "Camera không tồn tại"}, HTTPStatus.NOT_FOUND)
        boundary = "qhhframe"
        reader = RecordedVideoReader(camera_id, max_fps=float(fps))
        try:
            self.send_response(HTTPStatus.OK)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={boundary}",
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_wait_log = 0.0
            while True:
                item = reader.read()
                if item is None:
                    now = time.monotonic()
                    if now - last_wait_log > 10.0:
                        print(
                            f"[stream] waiting for recorded video cam={camera_id}",
                            flush=True,
                        )
                        last_wait_log = now
                    time.sleep(0.25)
                    continue
                _, frame = item
                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                )
                if not ok:
                    continue
                body = encoded.tobytes()
                self.wfile.write(
                    (
                        f"--{boundary}\r\n"
                        "Content-Type: image/jpeg\r\n"
                        f"Content-Length: {len(body)}\r\n\r\n"
                    ).encode("ascii")
                )
                self.wfile.write(body)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        finally:
            reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=str(env_or_config("QHH_WEB_HOST", "web", "host", "0.0.0.0")),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(env_or_config("QHH_WEB_PORT", "web", "port", 8090)),
    )
    args = parser.parse_args()
    server = WebHTTPServer((args.host, args.port), RequestHandler)
    print(f"QHH web: http://{args.host}:{args.port}", flush=True)
    print(f"QHH build: {BUILD_ID}", flush=True)
    print(f"Recording root: {RecordedVideoReader('').root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        AI_MONITOR.stop()
        server.server_close()


if __name__ == "__main__":
    main()
