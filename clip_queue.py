"""Stub clip queue — local folder scanner + Redis dedup/status helpers.

Đây là bản demo/stub: production version do component khác phụ trách. Module
này đủ để smoke-test pipeline Celery end-to-end với clip mẫu (xem
scripts/seed_demo_clip.py).

Key namespace (xem CELERY_LOCAL_PLAN_v2.md §3):
- qhh:ai:clip:pushed            (Set)   đường dẫn clip đã apply_async
- qhh:ai:clip:status:{md5}      (Hash)  trạng thái live mỗi clip
- qhh:ai:clip:result:{md5}      (String JSON, ghi từ tasks.py)
- qhh:ai:clip:index:{cam}:{cls} (ZSet, ghi từ tasks.py) — score = epochMs
- qhh:ai:clip:queue:depth       (String, ghi từ scheduler) — TTL 60s

Filename convention: cam{cameraId}_class{classId}_{epochMs}.mp4
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from config_loader import env_or_config
from db import redis_client as db


VIDEO_DIR = Path(
    env_or_config("VIDEO_DIR", "local", "video_dir", "videos")
).expanduser().resolve()

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}

PUSHED_SET = "qhh:ai:clip:pushed"
STATUS_KEY = "qhh:ai:clip:status:{md5}"
QUEUE_DEPTH_KEY = "qhh:ai:clip:queue:depth"
STATUS_TTL = 7 * 24 * 3600
QUEUE_DEPTH_TTL = 60

# cameraId / classId in QHH are typically GUID strings, but may be shorter
# alphanumeric identifiers. Accept any URL-safe token (letters/digits/dash).
_ID_RE = r"[0-9a-zA-Z][0-9a-zA-Z\-]{3,63}"
_NAME_RE = re.compile(
    rf"^cam(?P<cam>{_ID_RE})_class(?P<cls>{_ID_RE})_(?P<ts>\d+)\.[^.]+$"
)


def md5_of(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mtime_min_age() -> float:
    return float(env_or_config(
        "CLIP_MTIME_MIN_AGE_SEC", "local", "mtime_min_age_sec", 3
    ))


def parse_clip_name(path: Path) -> dict | None:
    """Return {cameraId, classId, epochMs} or None if filename doesn't match."""
    m = _NAME_RE.match(path.name)
    if not m:
        return None
    return {
        "cameraId": m["cam"],
        "classId": m["cls"],
        "epochMs": int(m["ts"]),
    }


def scan_new_clips() -> list[tuple[str, dict]]:
    """Return [(abs_path, meta), ...] for clips not yet in PUSHED_SET.

    Skips files whose mtime is younger than mtime_min_age_sec (still being
    written) and files whose name doesn't match cam{guid}_class{guid}_{ts}.ext.
    Sorted by epochMs ascending so older clips are dispatched first.
    """
    r = db.get_client()
    out: list[tuple[str, dict]] = []
    now = time.time()
    min_age = _mtime_min_age()

    if not VIDEO_DIR.exists():
        return out

    for p in VIDEO_DIR.rglob("*"):
        if p.suffix.lower() not in VIDEO_EXTS:
            continue
        try:
            stat = p.stat()
        except FileNotFoundError:
            continue
        if now - stat.st_mtime < min_age:
            continue
        ap = str(p.resolve())
        if r.sismember(PUSHED_SET, ap):
            continue
        meta = parse_clip_name(p)
        if meta is None:
            # Skip filenames that don't match convention; do not claim them.
            continue
        out.append((ap, meta))

    out.sort(key=lambda x: x[1]["epochMs"])
    return out


def mark_pushed(path: str) -> bool:
    """SADD path to dedup set. True if newly added, False if already present."""
    return db.get_client().sadd(PUSHED_SET, path) == 1


def unmark_pushed(path: str) -> None:
    """Remove a claim (used when apply_async fails so scheduler can retry)."""
    db.get_client().srem(PUSHED_SET, path)


def set_status(path: str, status: str, **fields) -> None:
    """HSET status fields + EXPIRE. Coerces all values to str for Redis."""
    r = db.get_client()
    key = STATUS_KEY.format(md5=md5_of(path))
    data = {"path": path, "status": status, "updatedAt": _now_iso()}
    for k, v in fields.items():
        if v is None:
            continue
        data[k] = str(v)
    r.hset(key, mapping=data)
    r.expire(key, STATUS_TTL)


def get_status(path: str) -> dict:
    return db.get_client().hgetall(STATUS_KEY.format(md5=md5_of(path)))


def update_queue_depth(depth: int) -> None:
    r = db.get_client()
    r.set(QUEUE_DEPTH_KEY, str(int(depth)), ex=QUEUE_DEPTH_TTL)
