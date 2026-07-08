"""Read completed FFmpeg recording segments for camera preview and AI."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import cv2

from config_loader import ROOT_DIR, env_or_config


VIDEO_EXTENSIONS = (".mkv", ".mp4", ".mov")


def recording_root() -> Path:
    path = Path(
        str(
            env_or_config(
                "QHH_RECORD_DIR",
                "recording",
                "root_dir",
                "recordings",
            )
        )
    ).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return path


def _safe_camera_id(camera_id: str) -> str:
    return "".join(
        ch for ch in str(camera_id) if ch.isalnum() or ch in "-_"
    ) or "camera"


def _segment_from_ready(marker: Path) -> Path | None:
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        video_path = payload.get("videoPath") or payload.get("video_path")
        if video_path:
            path = Path(str(video_path))
            if path.is_file():
                return path
    except (OSError, ValueError, TypeError):
        pass
    path = marker.with_suffix("")
    return path if path.is_file() else None


def ready_segments(camera_id: str, root: Path | None = None) -> list[Path]:
    """Return atomically finalized video files, sorted by mtime/name."""
    base = (root or recording_root()) / _safe_camera_id(camera_id)
    if not base.is_dir():
        return []
    paths: list[Path] = []

    # Current recorders atomically rename a hidden *.part file to its final
    # extension. Visible video files are therefore complete and safe to open.
    for ext in VIDEO_EXTENSIONS:
        paths.extend(path for path in base.rglob(f"*{ext}") if path.is_file())

    # Compatibility with recordings produced by the older marker protocol.
    for marker in base.rglob("*.ready"):
        segment = _segment_from_ready(marker)
        if segment is not None:
            paths.append(segment)
    existing = [path for path in set(paths) if path.is_file()]
    return sorted(existing, key=lambda path: (_safe_mtime(path), str(path)))


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _segment_started_at(path: Path) -> float:
    try:
        return datetime.strptime(path.stem[:15], "%Y%m%d_%H%M%S").timestamp()
    except (ValueError, OSError):
        return _safe_mtime(path)


def latest_recording_status(camera_id: str, root: Path | None = None) -> dict:
    segments = ready_segments(camera_id, root)
    if not segments:
        return {"ready": False, "latest": "", "updated_at": 0.0}
    latest = segments[-1]
    try:
        updated_at = latest.stat().st_mtime
    except OSError:
        updated_at = 0.0
    return {"ready": True, "latest": str(latest), "updated_at": updated_at}


class RecordedVideoReader:
    """Small latest-segment reader with the frame-reader interface AI expects."""

    def __init__(
        self,
        camera_id: str,
        root: str | Path | None = None,
        realtime: bool = True,
        max_fps: float | None = None,
        start_after: float | None = None,
        delete_after_read: bool = False,
        on_segment_done: Callable[[Path], None] | None = None,
    ):
        self.camera_id = str(camera_id)
        self.root = Path(root).expanduser() if root else recording_root()
        self.realtime = bool(realtime)
        self.start_after = float(start_after) if start_after is not None else None
        self.delete_after_read = bool(delete_after_read)
        self.on_segment_done = on_segment_done
        if max_fps is None:
            try:
                max_fps = float(os.getenv("QHH_VIDEO_READER_MAX_FPS", "25"))
            except ValueError:
                max_fps = 25.0
        self.max_fps = max(0.1, float(max_fps))
        self._cap: cv2.VideoCapture | None = None
        self._path: Path | None = None
        self._bad_paths: set[Path] = set()
        self._done_paths: set[Path] = set()
        self._sequence = 0
        self._frame_delay = 1.0 / self.max_fps
        self._next_frame_at = 0.0

    @property
    def current_path(self) -> Path | None:
        return self._path

    def close(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._path = None

    def _finish_current_segment(self):
        path = self._path
        if path is None:
            self.close()
            return
        self.close()
        self._done_paths.add(path)
        if self.on_segment_done is not None:
            try:
                self.on_segment_done(path)
            except Exception as exc:
                print(f"[video-reader] segment callback failed file={path}: {exc}", flush=True)
        if self.delete_after_read:
            try:
                path.unlink()
                print(f"[video-reader] deleted processed segment file={path}", flush=True)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"[video-reader] delete failed file={path}: {exc}", flush=True)

    def _next_segment(self) -> Path | None:
        segments = [
            path for path in ready_segments(self.camera_id, self.root)
            if path not in self._bad_paths and path not in self._done_paths
        ]
        if self.start_after is not None:
            cutoff = self.start_after - 1.0
            segments = [
                path for path in segments
                if _segment_started_at(path) >= cutoff
            ]
        if not segments:
            return None
        if self._path is None:
            if self.start_after is not None or self.delete_after_read:
                return segments[0]
            return segments[-1]
        try:
            current_index = segments.index(self._path)
        except ValueError:
            if self.start_after is not None or self.delete_after_read:
                return segments[0]
            return segments[-1]
        if current_index + 1 < len(segments):
            return segments[current_index + 1]
        return None

    def _open(self, path: Path) -> bool:
        self.close()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps > 0:
            self._frame_delay = 1.0 / min(float(fps), self.max_fps)
        else:
            self._frame_delay = 1.0 / self.max_fps
        self._cap = cap
        self._path = path
        self._next_frame_at = time.monotonic()
        return True

    def read(self):
        if self._cap is None:
            path = self._next_segment()
            if path is None:
                return None
            if not self._open(path):
                self._bad_paths.add(path)
                return None

        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._finish_current_segment()
            next_path = self._next_segment()
            if next_path is None:
                return None
            if not self._open(next_path):
                self._bad_paths.add(next_path)
                return None
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._path is not None:
                    self._bad_paths.add(self._path)
                self.close()
                return None

        if self.realtime:
            now = time.monotonic()
            if self._next_frame_at > now:
                time.sleep(self._next_frame_at - now)
            self._next_frame_at = max(time.monotonic(), self._next_frame_at) + self._frame_delay

        self._sequence += 1
        return self._sequence, frame
