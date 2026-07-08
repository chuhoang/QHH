"""Scheduler — quét folder mỗi N giây, claim + apply_async clip mới.

Chạy:  python -m scheduler         (foreground)
       python scheduler.py
"""

from __future__ import annotations

import signal
import time
import traceback

from config_loader import env_or_config
import clip_queue as cq
from tasks import process_clip_task


def _scan_interval() -> float:
    return float(env_or_config(
        "CLIP_SCAN_INTERVAL_SEC", "local", "scan_interval_sec", 5
    ))


def tick() -> int:
    """One scan + dispatch pass. Returns number of clips dispatched."""
    dispatched = 0
    clips = cq.scan_new_clips()
    cq.update_queue_depth(len(clips))
    for path, meta in clips:
        if not cq.mark_pushed(path):
            continue  # someone else claimed it
        try:
            cq.set_status(
                path, "PENDING",
                cameraId=meta["cameraId"], classId=meta["classId"],
            )
            res = process_clip_task.apply_async(
                args=[path, meta["cameraId"], meta["classId"]],
                queue="clip",
            )
            cq.set_status(
                path, "PENDING",
                taskId=res.id,
                cameraId=meta["cameraId"], classId=meta["classId"],
            )
            dispatched += 1
        except Exception as exc:  # noqa: BLE001
            # apply_async failed — release claim so a later tick can retry.
            cq.unmark_pushed(path)
            cq.set_status(
                path, "FAILED",
                error=f"dispatch failed: {exc}",
                cameraId=meta["cameraId"], classId=meta["classId"],
            )
            traceback.print_exc()
    return dispatched


_running = True


def _stop(*_a):
    global _running
    _running = False


def main() -> None:
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    interval = _scan_interval()
    print(f"[scheduler] watching {cq.VIDEO_DIR} every {interval}s")
    while _running:
        try:
            n = tick()
            if n:
                print(f"[scheduler] dispatched {n} clip(s)")
        except Exception:
            traceback.print_exc()
        for _ in range(int(interval * 10)):
            if not _running:
                break
            time.sleep(0.1)
    print("[scheduler] stopped")


if __name__ == "__main__":
    main()
