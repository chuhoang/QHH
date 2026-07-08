"""Adapter to Middleware2026's canonical AI_Read SHM reader."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from config_loader import env_or_config


def _load_ai_read_shm_module():
    ai_read_dir = Path(
        env_or_config(
            "MIDDLEWARE_AI_READ_DIR",
            "middleware",
            "ai_read_dir",
            "/home/mq/Middleware2026/AI_Read",
        )
    )
    module_path = ai_read_dir / "shm" / "SHMReader_3.py"
    if not module_path.is_file():
        raise RuntimeError(
            f"Không tìm thấy Middleware AI_Read SHM reader: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "middleware2026_ai_read_shm", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Không thể load SHM reader: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Lazy-load the SHM reader so importing this module in a container without
# Middleware2026 (clip-only batch worker) doesn't fail. The real Middleware
# reader is only needed by the live camera pipeline (main.py / web_server.py).
_AI_READ_SHM = None


def _ai_read_shm():
    global _AI_READ_SHM
    if _AI_READ_SHM is None:
        _AI_READ_SHM = _load_ai_read_shm_module()
    return _AI_READ_SHM


class MiddlewareSHMReader:
    """Expose AI_Read/SHMReader_3 through the small interface used by QHH."""

    def __init__(self, key: int, queue_size: int = 5, depth: int = 3):
        channel = _ai_read_shm().Channel(int(key))
        channel.list_cid = []
        channel.size_shm = int(queue_size)
        channel.width_img = 1920
        channel.height_img = 1080
        channel.depth_img = int(depth)
        self._reader = _AI_READ_SHM.SHMReader(channel)

    def close(self):
        memory = getattr(self._reader, "memory", None)
        if memory is not None:
            try:
                memory.detach()
            except Exception:
                pass
            self._reader.memory = None

    def read(self):
        ret, frame_groups = self._reader.Read()
        if ret <= 0:
            return None
        for frame_group in frame_groups:
            if not frame_group:
                continue
            frame = frame_group[0]
            if frame.image is not None:
                return int(frame.count), frame.image
        return None
