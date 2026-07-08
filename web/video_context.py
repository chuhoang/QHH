"""Resolve video context (class + period + AI config) from Redis only.

Input:
    cameraId, videoStartTime (UTC or with tz)
Output:
    dict | None — see Response Model bên dưới.

Tất cả truy vấn đi qua Redis. Không đụng tới DB.
"""

from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from typing import Any

from db import redis_client as db

VN_TZ = timezone(timedelta(hours=7))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_vn(dt: datetime) -> datetime:
    """Chuyển videoStartTime sang giờ VN. Naive datetime giả định là UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(VN_TZ)


def _iso_week(dt: datetime) -> str:
    """ISO-8601 week label: 'YYYY-Www'."""
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _parse_clock(text: str) -> time | None:
    """Parse 'HH:MM' hoặc 'HH:MM:SS' → datetime.time."""
    if not text:
        return None
    try:
        parts = [int(x) for x in str(text).split(":")]
    except ValueError:
        return None
    if len(parts) == 2:
        return time(parts[0], parts[1])
    if len(parts) >= 3:
        return time(parts[0], parts[1], parts[2])
    return None


def _decode_str(v) -> str:
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="replace")
    return "" if v is None else str(v)


# ---------------------------------------------------------------------------
# Step 2: cameraId → classId(s)
# ---------------------------------------------------------------------------
def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_class_ids(camera_id: str) -> list[str]:
    """Lookup camera-class index. Trả về danh sách classId còn hiệu lực.

    Index key: qhh:attendance:camera-class:index:{cameraId} (Set of classIds).
    """
    if not camera_id:
        return []
    r = db.get_client()
    raw = r.smembers(f"qhh:attendance:camera-class:index:{camera_id}")
    return sorted(_decode_str(c) for c in raw if c)


# ---------------------------------------------------------------------------
# Step 3: timetable → active slot
# ---------------------------------------------------------------------------
def _find_active_slot(
    iso_week: str,
    class_id: str,
    day_of_week: int,
    clock: time,
) -> dict | None:
    """GET qhh:timetable:week:{iso}:course:{classId} → tìm slot match.

    Không có key TKB → None (lớp không có lịch tuần này = ngoài giờ, KHÔNG lỗi).
    Có key nhưng JSON hỏng → raise (data hỏng, phải báo chứ không nuốt).
    """
    key = f"qhh:timetable:week:{iso_week}:course:{class_id}"
    raw = db.get_client().get(key)
    if not raw:
        return None
    try:
        timetable = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptDataException(f"Corrupt timetable JSON at {key}") from exc

    for slot in timetable.get("slots", []) or []:
        if int(slot.get("dayOfWeek", -1)) != day_of_week:
            continue
        if str(slot.get("status", "")).lower() != "active":
            continue
        start = _parse_clock(slot.get("startTime", ""))
        end = _parse_clock(slot.get("endTime", ""))
        if start is None or end is None:
            continue
        if start <= clock < end:
            return slot
    return None


# ---------------------------------------------------------------------------
# Step 4: camera-class config
# ---------------------------------------------------------------------------
def _load_camera_class_config(camera_id: str, class_id: str) -> dict | None:
    """None nếu key không tồn tại; raise nếu key có nhưng JSON hỏng."""
    key = f"qhh:attendance:camera-class:{camera_id}:{class_id}"
    raw = db.get_client().get(key)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorruptDataException(f"Corrupt camera-class JSON at {key}") from exc


def _build_context(
    camera_id: str,
    class_id: str,
    config: dict,
    slot: dict,
    iso_week: str,
    local_dt: datetime,
) -> dict[str, Any]:
    return {
        "cameraId": camera_id,
        "classId": class_id,
        "classCode": config.get("classCode", ""),
        "periodNumber": slot.get("periodNumber"),
        "subjectId": slot.get("subjectId"),
        "teacherId": slot.get("teacherId"),
        "roomId": slot.get("roomId"),
        "startTime": slot.get("startTime"),
        "endTime": slot.get("endTime"),
        "students": config.get("students", []),
        "regions": config.get("regions", []),
        # Gaze wedge (plan_distraction): 2 điểm mép bảng + 2 điểm giới hạn
        # cao độ, server bổ sung khi cấu hình TKB. None nếu chưa cấu hình
        # → pipeline tự fallback chế độ ngưỡng yaw/pitch cũ.
        "boardLine": config.get("boardLine"),
        "pitchLimit": config.get("pitchLimit"),
        "aiEnabled": _truthy(config.get("aiEnabled", False)),
        "rtspChannel": config.get("rtspChannel"),
        "rtspPath": config.get("rtspPath", ""),
        "isoWeek": iso_week,
        "videoStartTime": local_dt.isoformat(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class ConfigurationException(RuntimeError):
    """Camera-class JSON thiếu — không thể tiếp tục pipeline AI."""


class CorruptDataException(RuntimeError):
    """Key TỒN TẠI trong Redis nhưng JSON hỏng — lỗi data, phải raise chứ không nuốt."""


def resolve_video_context(
    camera_id: str,
    video_start_time: datetime,
) -> dict[str, Any] | None:
    """Trả về context cho 1 video.

    None nếu:
        • Không tìm thấy class cho camera, hoặc
        • Không có timetable cho tuần đó, hoặc
        • Không có slot Active diễn ra tại videoStartTime, hoặc
        • Slot đang diễn ra nhưng camera-class đang tắt aiEnabled.

    Raises ConfigurationException nếu slot có nhưng thiếu camera-class JSON.
    """
    ctx, _reason = resolve_video_context_status(camera_id, video_start_time)
    return ctx


def resolve_video_context_status(
    camera_id: str,
    video_start_time: datetime,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve context kèm reason code để scheduler log đúng lý do skip."""
    if not camera_id:
        return None, "NO_CAMERA"

    # Step 1: timezone VN + ISO week
    local_dt = _to_vn(video_start_time)
    iso_week = _iso_week(local_dt)
    # ISO dayOfWeek: 1 = Mon ... 7 = Sun.
    dow = local_dt.isoweekday()
    clock = local_dt.time()

    # Step 2: cameraId → classId
    class_ids = _resolve_class_ids(camera_id)
    if not class_ids:
        return None, "NO_CAMERA_CLASS"

    saw_active_disabled = False
    for class_id in class_ids:
        # Step 3: active slot
        slot = _find_active_slot(iso_week, class_id, dow, clock)
        if slot is None:
            continue

        # Step 4: camera-class config + aiEnabled gate
        config = _load_camera_class_config(camera_id, class_id)
        if config is None:
            raise ConfigurationException(
                f"Missing qhh:attendance:camera-class:{camera_id}:{class_id}"
            )
        if not _truthy(config.get("aiEnabled", False)):
            saw_active_disabled = True
            continue
        return (
            _build_context(camera_id, class_id, config, slot, iso_week, local_dt),
            "OK",
        )

    return None, "AI_DISABLED" if saw_active_disabled else "NO_ACTIVE_PERIOD"
