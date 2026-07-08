"""
Redis data layer - all CRUD operations for the app.

Seat model:
- A classroom has N desks.
- Each desk has one or more physical slots (chỗ ngồi).
- A slot can exist before any student is assigned.
- AI zones belong to a camera-class desk region, not to individual slots.

Legacy slot/seat zones are still read once so existing data can be migrated.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit
import redis

from config_loader import env_or_config


DEFAULT_SLOT_COUNT = 4
REDIS_HOST = str(env_or_config("REDIS_HOST", "redis", "host", "127.0.0.1"))
REDIS_PORT = int(env_or_config("REDIS_PORT", "redis", "port", 6379))
REDIS_DB = int(env_or_config("REDIS_DB", "redis", "db", 0))
REDIS_PASSWORD = env_or_config("REDIS_PASSWORD", "redis", "password", "") or None
QHH_PREFIX = str(
    env_or_config("QHH_REDIS_PREFIX", "redis", "prefix", "qhh")
).strip(":")
MIDDLEWARE_STREAMS_KEY = str(env_or_config(
    "MIDDLEWARE_STREAMS_KEY",
    "middleware",
    "streams_key",
    f"{QHH_PREFIX}:middleware:streams",
))
_client: redis.Redis | None = None


def _is_misconf_error(exc: BaseException) -> bool:
    return "MISCONF" in str(exc).upper()


def get_client() -> redis.Redis:
    global _client
    if _client is None:
        # Try without AUTH first. This avoids sending AUTH to Redis servers
        # whose default user has no password configured.
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=None,
            decode_responses=True,
        )
        try:
            client.ping()
        except redis.exceptions.AuthenticationError as exc:
            if REDIS_PASSWORD:
                client = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=REDIS_DB,
                    password=REDIS_PASSWORD,
                    decode_responses=True,
                )
                try:
                    client.ping()
                except redis.exceptions.ResponseError as redis_exc:
                    if not _is_misconf_error(redis_exc):
                        raise
                    print(
                        "[redis] MISCONF during ping; continuing read path. "
                        "Redis writes may fail until RDB persistence is fixed.",
                        flush=True,
                    )
            else:
                raise
        except redis.exceptions.ResponseError as exc:
            if not _is_misconf_error(exc):
                raise
            print(
                "[redis] MISCONF during ping; continuing read path. "
                "Redis writes may fail until RDB persistence is fixed.",
                flush=True,
            )
        _client = client
    return _client


def ping() -> bool:
    try:
        return get_client().ping()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# helpers


def _new_id() -> str:
    return str(uuid.uuid4())


def _qkey(*parts) -> str:
    return ":".join([QHH_PREFIX, *(str(part).strip(":") for part in parts)])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hset(key: str, data: dict):
    get_client().hset(key, mapping=data)


def _hget(key: str) -> dict:
    return get_client().hgetall(key)


def _del(key: str):
    get_client().delete(key)


def _sadd(set_key: str, value: str):
    get_client().sadd(set_key, value)


def _srem(set_key: str, value: str):
    get_client().srem(set_key, value)


def _smembers(set_key: str) -> list[str]:
    return list(get_client().smembers(set_key))


def _json_loads(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return default


def _attendance_camera_class_configs(
    camera_id: str | None = None,
    class_id: str | None = None,
) -> list[dict]:
    """Read camera-class JSON records from the QHH attendance schema."""
    prefix = _qkey("attendance", "camera-class") + ":"
    pattern = prefix + (
        f"{camera_id}:*" if camera_id else "*"
    )
    configs = []
    for key in get_client().scan_iter(match=pattern):
        key = str(key)
        suffix = key[len(prefix):] if key.startswith(prefix) else ""
        if not suffix or suffix.startswith("index:"):
            continue
        raw = get_client().get(key)
        config = _json_loads(raw, {})
        if not isinstance(config, dict):
            continue
        if camera_id and str(config.get("cameraId", "")) != str(camera_id):
            continue
        if class_id and str(config.get("classId", "")) != str(class_id):
            continue
        configs.append(config)
    return configs


def _attendance_student(raw: dict, class_id: str) -> dict:
    student_id = str(raw.get("id", "") or "")
    overlay = _hget(f"student:{student_id}") if student_id else {}
    return {
        "id": student_id,
        "name": str(
            overlay.get("name")
            or raw.get("fullName", raw.get("name", ""))
            or ""
        ),
        "student_code": str(
            overlay.get("student_code")
            or raw.get("studentCode", raw.get("student_code", ""))
            or ""
        ),
        "class_id": str(overlay.get("class_id") or class_id or ""),
        "face_image": str(
            overlay.get("face_image")
            or raw.get("avatarUrl", raw.get("face_image", ""))
            or ""
        ),
    }


def _attendance_classroom(class_id: str) -> dict:
    configs = _attendance_camera_class_configs(class_id=class_id)
    if not configs:
        return {}
    name = next(
        (str(config.get("classCode", "") or "") for config in configs
         if config.get("classCode")),
        str(class_id),
    )
    num_desks = 0
    for config in configs:
        for region in config.get("regions", []) or []:
            if not isinstance(region, dict):
                continue
            num_desks = max(num_desks, _desk_number_from_region(region))
    return {
        "id": str(class_id),
        "name": name,
        "num_desks": str(num_desks),
        "source": "qhh:attendance:camera-class",
    }


def _sync_class_if_available(class_id: str):
    """Call the camera-class synchronizer after the module is fully loaded."""
    sync = globals().get("sync_class_camera_configs")
    if class_id and callable(sync):
        sync(class_id)


# ---------------------------------------------------------------------------
# STUDENTS


def create_student(name: str, student_code: str, class_id: str = "") -> dict:
    sid = _new_id()
    data = {
        "id": sid,
        "name": name,
        "student_code": student_code,
        "class_id": class_id,
        "face_image": "",
    }
    _hset(f"student:{sid}", data)
    _sadd("students", sid)
    if class_id:
        _sync_class_if_available(class_id)
    return data


def get_student(sid: str) -> dict:
    student = _hget(f"student:{sid}")
    if student and student.get("name") and student.get("student_code"):
        return student
    for config in _attendance_camera_class_configs():
        class_id = str(config.get("classId", "") or "")
        for raw in config.get("students", []) or []:
            if isinstance(raw, dict) and str(raw.get("id", "")) == str(sid):
                return _attendance_student(raw, class_id)
    return student


def update_student(sid: str, name: str, student_code: str, class_id: str = "") -> dict:
    previous = get_student(sid)
    data = {"id": sid, "name": name, "student_code": student_code, "class_id": class_id}
    _hset(f"student:{sid}", data)
    prev_class = str(previous.get("class_id", "") or "")
    new_class = str(class_id or "")
    # Moving a student to another class (or removing the class) must release
    # the seat slots they occupied in the old class. Otherwise monitoring of
    # the old class still pulls them into the recognition gallery via the
    # leftover slot.student_id and the empty seat is mislabelled.
    if prev_class and prev_class != new_class:
        for seat in list_seats(prev_class):
            slots = seat.get("slots", []) or []
            changed = False
            for slot in slots:
                if str(slot.get("student_id", "") or "") == sid:
                    slot["student_id"] = ""
                    changed = True
            if changed:
                set_seat_slots(prev_class, int(seat["desk_num"]), slots)
    for cid in {prev_class, new_class}:
        if cid:
            _sync_class_if_available(cid)
    return data


def set_student_face(sid: str, face_image: str) -> dict:
    """Attach a saved face image path to a student."""
    _hset(f"student:{sid}", {"face_image": face_image or ""})
    student = get_student(sid)
    if student.get("class_id"):
        _sync_class_if_available(student["class_id"])
    return student


def clear_student_face(sid: str) -> dict:
    """Remove the saved face image reference from a student."""
    return set_student_face(sid, "")


def delete_student(sid: str):
    previous = get_student(sid)
    _del(f"student:{sid}")
    _srem("students", sid)
    # Keep the configured slots/zones, only clear the removed student.
    for cid in _smembers("classrooms"):
        for seat in list_seats(cid):
            changed = False
            slots = seat.get("slots", [])
            for slot in slots:
                if slot.get("student_id") == sid:
                    slot["student_id"] = ""
                    changed = True
            if changed:
                set_seat_slots(cid, int(seat["desk_num"]), slots)
    if previous.get("class_id"):
        _sync_class_if_available(previous["class_id"])


def list_students(class_id: str | None = None) -> list[dict]:
    by_id = {}
    for sid in _smembers("students"):
        s = get_student(sid)
        if not s:
            continue
        if class_id is not None and s.get("class_id", "") != class_id:
            continue
        by_id[str(s.get("id", sid))] = s
    for config in _attendance_camera_class_configs(class_id=class_id):
        config_class_id = str(config.get("classId", "") or "")
        for raw in config.get("students", []) or []:
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            student = _attendance_student(raw, config_class_id)
            by_id.setdefault(student["id"], student)
    return sorted(by_id.values(), key=lambda x: x.get("name", ""))


# ---------------------------------------------------------------------------
# CLASSROOMS


def create_classroom(name: str, num_desks: int) -> dict:
    cid = _new_id()
    data = {"id": cid, "name": name, "num_desks": str(num_desks)}
    _hset(f"classroom:{cid}", data)
    _sadd("classrooms", cid)
    ensure_classroom_seats(cid)
    return data


def get_classroom(cid: str) -> dict:
    classroom = _hget(f"classroom:{cid}")
    return classroom or _attendance_classroom(cid)


def update_classroom(cid: str, name: str, num_desks: int) -> dict:
    data = {"id": cid, "name": name, "num_desks": str(num_desks)}
    _hset(f"classroom:{cid}", data)
    ensure_classroom_seats(cid)
    # Remove seats beyond the new desk count.
    for dn in list(_smembers(f"seats:{cid}")):
        try:
            desk_num = int(dn)
        except Exception:
            continue
        if desk_num > int(num_desks):
            delete_seat(cid, desk_num)
    _sync_class_if_available(cid)
    return get_classroom(cid)


def delete_classroom(cid: str):
    camera_ids = (
        _smembers(_class_cameras_index_key(cid))
        if "_class_cameras_index_key" in globals()
        else []
    )
    for seat in list_seats(cid):
        delete_seat(cid, int(seat["desk_num"]))
    unlink = globals().get("_unlink_camera_class")
    if callable(unlink):
        for cam_id in camera_ids:
            unlink(cam_id, cid)
        get_client().delete(_class_cameras_index_key(cid))
    _del(f"classroom:{cid}")
    _srem("classrooms", cid)


def list_classrooms() -> list[dict]:
    by_id = {}
    for cid in _smembers("classrooms"):
        c = get_classroom(cid)
        if c:
            by_id[str(c.get("id", cid))] = c
    for config in _attendance_camera_class_configs():
        cid = str(config.get("classId", "") or "")
        if cid and cid not in by_id:
            classroom = _attendance_classroom(cid)
            if classroom:
                by_id[cid] = classroom
    return sorted(by_id.values(), key=lambda x: x.get("name", ""))


# ---------------------------------------------------------------------------
# SEAT SLOTS


def _empty_slot(slot_num: int) -> dict:
    return {"slot_num": int(slot_num), "student_id": "", "zone": {}}


def _normalise_student_ids(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        parsed = _json_loads(text, None)
        if isinstance(parsed, list):
            return [str(v) for v in parsed if str(v).strip()]
        return [text]
    return [str(value)]


def _normalise_slots(raw) -> list[dict]:
    parsed = _json_loads(raw, [])
    if not isinstance(parsed, list):
        parsed = []

    slots: list[dict] = []
    for idx, slot in enumerate(parsed, start=1):
        if not isinstance(slot, dict):
            continue
        zone = slot.get("zone", {})
        if not isinstance(zone, dict):
            zone = _json_loads(zone, {})
        anchor = slot.get("anchor", {})
        if not isinstance(anchor, dict):
            anchor = _json_loads(anchor, {})
        try:
            slot_num = int(slot.get("slot_num", idx))
        except Exception:
            slot_num = idx
        out_slot = {
            "slot_num": slot_num,
            "student_id": str(slot.get("student_id", "") or ""),
            "zone": zone if isinstance(zone, dict) else {},
        }
        if isinstance(anchor, dict) and anchor:
            out_slot["anchor"] = anchor
        slots.append(out_slot)

    slots.sort(key=lambda s: int(s.get("slot_num", 0)))
    for idx, slot in enumerate(slots, start=1):
        slot["slot_num"] = idx
    return slots


def _legacy_slots_from_seat(raw: dict) -> list[dict]:
    ids = _normalise_student_ids(raw.get("student_ids"))
    if not ids and raw.get("student_id"):
        ids = _normalise_student_ids(raw.get("student_id"))

    zone = _json_loads(raw.get("zone"), {})
    if not ids:
        return [_empty_slot(1) | {"zone": zone if isinstance(zone, dict) else {}}]

    slots = []
    for idx, sid in enumerate(ids, start=1):
        # Legacy one-zone desk: keep the old zone on the first slot only.
        slots.append({
            "slot_num": idx,
            "student_id": sid,
            "zone": zone if idx == 1 and isinstance(zone, dict) else {},
        })
    return slots


def _serialise_slots(slots: list[dict]) -> str:
    clean = _normalise_slots(slots)
    return json.dumps(clean, ensure_ascii=False)


def _seat_key(class_id: str, desk_num: int) -> str:
    return f"seat:{class_id}:{desk_num}"


def _write_seat(class_id: str, desk_num: int, slots: list[dict]) -> dict:
    slots = _normalise_slots(slots) or [_empty_slot(1)]
    student_ids = [slot.get("student_id", "") for slot in slots if slot.get("student_id")]
    first_zone = next((slot.get("zone", {}) for slot in slots if slot.get("zone")), {})
    data = {
        "class_id": class_id,
        "desk_num": str(desk_num),
        "capacity": str(len(slots)),
        "slots": _serialise_slots(slots),
        # Compatibility fields for older UI/code.
        "student_id": student_ids[0] if student_ids else "",
        "student_ids": json.dumps(student_ids, ensure_ascii=False),
        "zone": json.dumps(first_zone if isinstance(first_zone, dict) else {}, ensure_ascii=False),
    }
    _hset(_seat_key(class_id, desk_num), data)
    _sadd(f"seats:{class_id}", str(desk_num))
    return get_seat(class_id, desk_num)


def get_seat(class_id: str, desk_num: int) -> dict:
    raw = _hget(_seat_key(class_id, desk_num))
    if not raw:
        return {}

    slots = _normalise_slots(raw.get("slots"))
    if not slots:
        slots = _legacy_slots_from_seat(raw)
        _write_seat(class_id, desk_num, slots)
        raw = _hget(_seat_key(class_id, desk_num))

    student_ids = [slot.get("student_id", "") for slot in slots if slot.get("student_id")]
    first_zone = next((slot.get("zone", {}) for slot in slots if slot.get("zone")), {})
    raw["class_id"] = class_id
    raw["desk_num"] = str(desk_num)
    raw["slots"] = slots
    raw["capacity"] = str(len(slots))
    raw["student_ids"] = student_ids
    raw["student_id"] = student_ids[0] if student_ids else ""
    raw["zone"] = first_zone
    return raw


def get_or_create_seat(class_id: str, desk_num: int, default_capacity: int = DEFAULT_SLOT_COUNT) -> dict:
    seat = get_seat(class_id, desk_num)
    if seat:
        return seat
    slots = [_empty_slot(i) for i in range(1, max(1, int(default_capacity)) + 1)]
    return _write_seat(class_id, desk_num, slots)


def ensure_classroom_seats(class_id: str, default_capacity: int = DEFAULT_SLOT_COUNT) -> list[dict]:
    classroom = get_classroom(class_id)
    if not classroom:
        return []
    try:
        num_desks = int(classroom.get("num_desks", 0))
    except Exception:
        num_desks = 0
    seats = []
    for desk_num in range(1, num_desks + 1):
        seats.append(get_or_create_seat(class_id, desk_num, default_capacity))
    return seats


def _backfill_anchors_if_missing(class_id: str, seat: dict) -> dict:
    """Compute anchor=centroid(zone) for any slot that has a zone but no anchor.

    Idempotent: returns immediately if all slots are already up to date.
    """
    slots = seat.get("slots", []) or []
    changed = False
    for slot in slots:
        if isinstance(slot.get("anchor"), dict) and slot["anchor"]:
            continue
        zone = slot.get("zone", {})
        if not isinstance(zone, dict) or not zone:
            continue
        cx, cy = _zone_centroid(zone)
        slot["anchor"] = {"cx": round(cx, 3), "cy": round(cy, 3)}
        changed = True
    if changed:
        return _write_seat(class_id, int(seat.get("desk_num", 0) or 0), slots)
    return seat


def list_seats(class_id: str, ensure_defaults: bool = True) -> list[dict]:
    legacy_desk_numbers = _smembers(f"seats:{class_id}")
    if ensure_defaults and (_hget(f"classroom:{class_id}") or legacy_desk_numbers):
        ensure_classroom_seats(class_id)
    result = []
    for dn in _smembers(f"seats:{class_id}"):
        try:
            desk_num = int(dn)
        except Exception:
            continue
        s = get_seat(class_id, desk_num)
        if s:
            s = _backfill_anchors_if_missing(class_id, s)
            result.append(s)
    if result:
        return sorted(result, key=lambda x: int(x.get("desk_num", 0)))

    # Redis trung tâm lưu bàn, sinh viên và vùng trong camera-class JSON.
    # Dựng seat view trong bộ nhớ, không tạo thêm key schema cũ.
    configs = _attendance_camera_class_configs(class_id=class_id)
    regions_by_desk = {}
    for config in configs:
        for region in config.get("regions", []) or []:
            if not isinstance(region, dict):
                continue
            desk_num = _desk_number_from_region(region)
            if desk_num and desk_num not in regions_by_desk:
                regions_by_desk[desk_num] = region
    for desk_num, region in sorted(regions_by_desk.items()):
        student_ids = _normalise_student_ids(region.get("studentIds"))
        slots = [
            {
                "slot_num": index,
                "student_id": student_id,
                "zone": {},
            }
            for index, student_id in enumerate(student_ids, start=1)
        ] or [_empty_slot(1)]
        result.append({
            "class_id": str(class_id),
            "desk_num": str(desk_num),
            "capacity": str(len(slots)),
            "slots": slots,
            "student_ids": student_ids,
            "student_id": student_ids[0] if student_ids else "",
            "zone": {},
            "source": "qhh:attendance:camera-class",
        })
    return result


def set_seat_slots(class_id: str, desk_num: int, slots: list[dict]) -> dict:
    seat = _write_seat(class_id, desk_num, slots)
    sync = globals().get("sync_class_camera_configs")
    if callable(sync):
        sync(class_id)
    return seat


def set_seat_capacity(class_id: str, desk_num: int, capacity: int) -> dict:
    capacity = max(1, int(capacity))
    seat = get_or_create_seat(class_id, desk_num)
    slots = seat.get("slots", [])
    current = len(slots)
    if capacity > current:
        for slot_num in range(current + 1, capacity + 1):
            slots.append(_empty_slot(slot_num))
    elif capacity < current:
        slots = slots[:capacity]
    return set_seat_slots(class_id, desk_num, slots)


def get_seat_slot(class_id: str, desk_num: int, slot_num: int) -> dict:
    seat = get_or_create_seat(class_id, desk_num)
    for slot in seat.get("slots", []):
        if int(slot.get("slot_num", 0)) == int(slot_num):
            return slot
    return {}


def update_seat_slot(class_id: str, desk_num: int, slot_num: int,
                     student_id: str | None = None, zone: dict | None = None) -> dict:
    seat = get_or_create_seat(class_id, desk_num)
    slots = seat.get("slots", [])
    while len(slots) < int(slot_num):
        slots.append(_empty_slot(len(slots) + 1))
    for slot in slots:
        if int(slot.get("slot_num", 0)) == int(slot_num):
            if student_id is not None:
                slot["student_id"] = student_id or ""
            if zone is not None:
                slot["zone"] = zone or {}
            break
    return set_seat_slots(class_id, desk_num, slots)


def assign_student_to_slot(class_id: str, desk_num: int, slot_num: int, student_id: str) -> dict:
    # A student should only occupy one slot inside the same class.
    for seat in list_seats(class_id):
        changed = False
        slots = seat.get("slots", [])
        for slot in slots:
            if slot.get("student_id") == student_id:
                slot["student_id"] = ""
                changed = True
        if changed:
            set_seat_slots(class_id, int(seat["desk_num"]), slots)
    return update_seat_slot(class_id, desk_num, slot_num, student_id=student_id)


def clear_slot_student(class_id: str, desk_num: int, slot_num: int) -> dict:
    return update_seat_slot(class_id, desk_num, slot_num, student_id="")


def _zone_centroid(zone: dict) -> tuple[float, float]:
    """Return the (cx, cy) centroid of a zone in normalised frame coords."""
    if zone.get("type") == "oriented":
        return float(zone.get("cx", 0.5)), float(zone.get("cy", 0.5))
    x = float(zone.get("x", 0.0))
    y = float(zone.get("y", 0.0))
    w = float(zone.get("w", 0.0))
    h = float(zone.get("h", 0.0))
    return x + w / 2.0, y + h / 2.0


def set_slot_zone(class_id: str, desk_num: int, slot_num: int, zone: dict) -> dict:
    zone_type = zone.get("type", "normal")
    clean_zone: dict = {"type": zone_type}
    if zone_type == "oriented":
        clean_zone["cx"] = round(float(zone.get("cx", 0.5)), 3)
        clean_zone["cy"] = round(float(zone.get("cy", 0.5)), 3)
        clean_zone["w"] = round(float(zone.get("w", 0.1)), 3)
        clean_zone["h"] = round(float(zone.get("h", 0.1)), 3)
        clean_zone["angle"] = round(float(zone.get("angle", 0.0)), 1)
    else:
        clean_zone["x"] = round(float(zone.get("x", 0.0)), 3)
        clean_zone["y"] = round(float(zone.get("y", 0.0)), 3)
        clean_zone["w"] = round(float(zone.get("w", 0.0)), 3)
        clean_zone["h"] = round(float(zone.get("h", 0.0)), 3)
    # Save the slot zone AND auto-derive the slot anchor (centroid of the zone)
    # so the parent-zone AI pipeline gets a usable per-slot reference point
    # without any extra UI step.
    seat = update_seat_slot(class_id, desk_num, slot_num, zone=clean_zone)
    cx, cy = _zone_centroid(clean_zone)
    slots = seat.get("slots", [])
    for slot in slots:
        if int(slot.get("slot_num", 0)) == int(slot_num):
            slot["anchor"] = {
                "cx": round(cx, 3),
                "cy": round(cy, 3),
            }
            break
    return _write_seat(class_id, desk_num, slots)


def clear_slot_zone(class_id: str, desk_num: int, slot_num: int) -> dict:
    return update_seat_slot(class_id, desk_num, slot_num, zone={})


# Legacy API: desk-level assignment maps to slot 1.
def set_seat(class_id: str, desk_num: int, student_id: str, zone: dict | None = None) -> dict:
    update_seat_slot(class_id, desk_num, 1, student_id=student_id, zone=zone or {})
    return get_seat(class_id, desk_num)


def delete_seat(class_id: str, desk_num: int):
    _del(_seat_key(class_id, desk_num))
    _srem(f"seats:{class_id}", str(desk_num))


# ---------------------------------------------------------------------------
# CAMERAS


def _camera_key(cam_id: str) -> str:
    return _qkey("camera", cam_id)


def _camera_class_key(cam_id: str, class_id: str) -> str:
    return _qkey("attendance", "camera-class", cam_id, class_id)


def _camera_class_index_key(cam_id: str) -> str:
    return _qkey("attendance", "camera-class", "index", cam_id)


def _class_cameras_index_key(class_id: str) -> str:
    return _qkey("attendance", "class-cameras", "index", class_id)


def _bool_text(value) -> str:
    if isinstance(value, str):
        return "1" if value.strip().lower() in {"1", "true", "yes", "on"} else "0"
    return "1" if bool(value) else "0"


def _camera_rtsp_url(data: dict) -> str:
    """Build an RTSP URL while keeping the Redis schema compatible with QHH."""
    explicit = str(data.get("url", "") or "").strip()
    if explicit:
        return explicit
    host = str(data.get("ipAddress", "") or "").strip()
    if not host:
        return ""
    port = str(data.get("port", "554") or "554").strip()
    username = str(data.get("username", "") or "")
    password = str(data.get("password", "") or "")
    path = str(data.get("rtspPath", data.get("notes", "")) or "").strip()
    if not path:
        path = "/cam/realmonitor?channel=1&subtype=0"
    if not path.startswith("/"):
        path = "/" + path
    auth = ""
    if username:
        auth = quote(username, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    return f"rtsp://{auth}{host}:{port}{path}"


def _camera_class_ids(cam_id: str) -> list[str]:
    ids = set(_smembers(_camera_class_index_key(cam_id)))
    legacy = _hget(f"camera:{cam_id}")
    if legacy.get("class_id"):
        ids.add(legacy["class_id"])
    return sorted(cid for cid in ids if cid)


def _normalise_camera(raw: dict, cam_id: str = "") -> dict:
    if not raw:
        return {}
    result = dict(raw)
    result["id"] = str(result.get("id", cam_id) or cam_id)
    result["name"] = str(result.get("name", "") or "")
    result["ipAddress"] = str(result.get("ipAddress", "") or "")
    result["port"] = str(result.get("port", "554") or "554")
    result["onvifPort"] = str(result.get("onvifPort", "80") or "80")
    result["username"] = str(result.get("username", "") or "")
    result["password"] = str(result.get("password", "") or "")
    result["wsPort"] = str(result.get("wsPort", "") or "")
    result["streamUrl"] = str(result.get("streamUrl", "") or "")
    result["brand"] = str(result.get("brand", "") or "")
    result["model"] = str(result.get("model", "") or "")
    result["location"] = str(result.get("location", "") or "")
    result["notes"] = str(result.get("notes", "") or "")
    result["rtspPath"] = str(result.get("rtspPath", result["notes"]) or "")
    result["isActive"] = _bool_text(result.get("isActive", "1"))
    result["url"] = _camera_rtsp_url(result)
    class_ids = _camera_class_ids(result["id"])
    result["class_ids"] = class_ids
    result["class_id"] = str(result.get("class_id", "") or (class_ids[0] if class_ids else ""))
    return result


def _camera_hash(data: dict, cam_id: str, previous: dict | None = None) -> dict:
    previous = previous or {}
    explicit_url = str(data.get("url", "") or "")
    parsed_url = urlsplit(explicit_url) if explicit_url.startswith("rtsp://") else None
    if parsed_url:
        data = dict(data)
        data.setdefault("ipAddress", parsed_url.hostname or "")
        data.setdefault("port", parsed_url.port or 554)
        data.setdefault("username", unquote(parsed_url.username or ""))
        data.setdefault("password", unquote(parsed_url.password or ""))
        path = parsed_url.path or ""
        if parsed_url.query:
            path += "?" + parsed_url.query
        data.setdefault("rtspPath", path)
    rtsp_path = str(data.get("rtspPath", data.get("notes", previous.get("notes", ""))) or "")
    return {
        "id": cam_id,
        "name": str(data.get("name", previous.get("name", "")) or ""),
        "roomId": str(data.get("roomId", previous.get("roomId", "")) or ""),
        "roomName": str(data.get("roomName", previous.get("roomName", "")) or ""),
        "location": str(data.get("location", previous.get("location", "")) or ""),
        "ipAddress": str(data.get("ipAddress", previous.get("ipAddress", "")) or ""),
        "port": str(data.get("port", previous.get("port", "554")) or "554"),
        "onvifPort": str(data.get("onvifPort", previous.get("onvifPort", "80")) or "80"),
        "username": str(data.get("username", previous.get("username", "")) or ""),
        "password": str(data.get("password", previous.get("password", "")) or ""),
        "wsPort": str(data.get("wsPort", previous.get("wsPort", "")) or ""),
        "streamUrl": str(data.get("streamUrl", previous.get("streamUrl", "")) or ""),
        "brand": str(data.get("brand", previous.get("brand", "")) or ""),
        "model": str(data.get("model", previous.get("model", "")) or ""),
        "isActive": _bool_text(data.get("isActive", previous.get("isActive", "1"))),
        "notes": rtsp_path,
        "createdAt": str(previous.get("createdAt", "") or _utc_now()),
        "updatedAt": _utc_now() if previous else "",
    }


def _empty_camera_class_config(cam_id: str, class_id: str) -> dict:
    classroom = get_classroom(class_id)
    return {
        "cameraId": cam_id,
        "classId": class_id,
        "classCode": classroom.get("name", "") if classroom else "",
        "aiEnabled": True,
        "rtspChannel": 1,
        "rtspPath": get_camera(cam_id).get("rtspPath", ""),
        "regions": [],
        "students": [],
        "updatedAt": _utc_now(),
    }


def get_camera_class_config(cam_id: str, class_id: str) -> dict:
    raw = get_client().get(_camera_class_key(cam_id, class_id))
    config = _json_loads(raw, {})
    return config if isinstance(config, dict) else {}


def save_camera_class_config(config: dict) -> dict:
    """Persist the QHH camera-class JSON plus both lookup indexes."""
    cam_id = str(config.get("cameraId", "") or "")
    class_id = str(config.get("classId", "") or "")
    if not cam_id or not class_id:
        raise ValueError("cameraId and classId are required")
    clean = dict(config)
    clean["cameraId"] = cam_id
    clean["classId"] = class_id
    clean["updatedAt"] = _utc_now()
    get_client().set(
        _camera_class_key(cam_id, class_id),
        json.dumps(clean, ensure_ascii=False, separators=(",", ":")),
    )
    _sadd(_camera_class_index_key(cam_id), class_id)
    _sadd(_class_cameras_index_key(class_id), cam_id)
    return clean


def _unlink_camera_class(cam_id: str, class_id: str):
    get_client().delete(_camera_class_key(cam_id, class_id))
    _srem(_camera_class_index_key(cam_id), class_id)
    _srem(_class_cameras_index_key(class_id), cam_id)


def _sync_camera_class(cam_id: str, class_id: str):
    if not class_id:
        return
    config = get_camera_class_config(cam_id, class_id) or _empty_camera_class_config(cam_id, class_id)
    classroom = get_classroom(class_id)
    camera = get_camera(cam_id)
    config["classCode"] = classroom.get("name", "") if classroom else config.get("classCode", "")
    config["rtspPath"] = camera.get("rtspPath", camera.get("notes", ""))
    save_camera_class_config(config)
    if not config.get("regions"):
        # One-time migration from the former slot/seat-level zone storage.
        for seat in list_seats(class_id):
            legacy_zone = seat.get("zone") if isinstance(seat.get("zone"), dict) else {}
            if not legacy_zone:
                legacy_zone = next(
                    (
                        slot.get("zone", {})
                        for slot in seat.get("slots", [])
                        if isinstance(slot.get("zone"), dict) and slot.get("zone")
                    ),
                    {},
                )
            if legacy_zone:
                set_desk_region(
                    cam_id, class_id, int(seat.get("desk_num", 0)), legacy_zone
                )


def link_camera_class(cam_id: str, class_id: str) -> dict:
    """Link a camera to a class without removing its other class links."""
    if not get_camera(cam_id):
        raise ValueError("Camera không tồn tại")
    if not get_classroom(class_id):
        raise ValueError("Lớp học không tồn tại")
    _sync_camera_class(str(cam_id), str(class_id))
    return get_camera_class_config(str(cam_id), str(class_id))


def unlink_camera_class(cam_id: str, class_id: str):
    """Remove one camera-class link while preserving camera and class."""
    _unlink_camera_class(str(cam_id), str(class_id))


def create_camera(name: str | dict, url: str = "", class_id: str = "") -> dict:
    data = dict(name) if isinstance(name, dict) else {"name": name, "url": url, "class_id": class_id}
    cam_id = _new_id()
    _hset(_camera_key(cam_id), _camera_hash(data, cam_id))
    _sadd(_qkey("cameras"), cam_id)
    _sync_camera_class(cam_id, str(data.get("class_id", "") or ""))
    return get_camera(cam_id)


def get_camera(cam_id: str) -> dict:
    raw = _hget(_camera_key(cam_id))
    if raw:
        return _normalise_camera(raw, cam_id)
    return _normalise_camera(_hget(f"camera:{cam_id}"), cam_id)


def update_camera(cam_id: str, name: str | dict, url: str = "", class_id: str = "") -> dict:
    data = dict(name) if isinstance(name, dict) else {"name": name, "url": url, "class_id": class_id}
    previous = get_camera(cam_id)
    old_class_ids = set(_camera_class_ids(cam_id))
    update_single_class = "class_id" in data
    new_class_id = str(data.get("class_id", "") or "")
    _hset(_camera_key(cam_id), _camera_hash(data, cam_id, previous))
    _sadd(_qkey("cameras"), cam_id)
    if update_single_class:
        for old_class_id in old_class_ids:
            if old_class_id != new_class_id:
                _unlink_camera_class(cam_id, old_class_id)
        _sync_camera_class(cam_id, new_class_id)
    return get_camera(cam_id)


def delete_camera(cam_id: str):
    for class_id in _camera_class_ids(cam_id):
        _unlink_camera_class(cam_id, class_id)
    get_client().delete(_camera_class_index_key(cam_id), _camera_key(cam_id), f"camera:{cam_id}")
    _srem(_qkey("cameras"), cam_id)
    _srem("cameras", cam_id)


def list_cameras() -> list[dict]:
    result = []
    camera_ids = set(_smembers(_qkey("cameras"))) | set(_smembers("cameras"))
    for cam_id in camera_ids:
        c = get_camera(cam_id)
        if c:
            result.append(c)
    return sorted(result, key=lambda x: x.get("name", ""))


def get_middleware_stream(cam_id: str) -> dict:
    """Return Middleware2026's UUID -> numeric SHM mapping for a camera."""
    raw = get_client().get(MIDDLEWARE_STREAMS_KEY)
    streams = _json_loads(raw, {})
    if not isinstance(streams, dict):
        return {}
    cameras = streams.get("list_camera", {})
    if not isinstance(cameras, dict):
        return {}
    stream = cameras.get(str(cam_id), {})
    return dict(stream) if isinstance(stream, dict) else {}


# ---------------------------------------------------------------------------
# CAMERA ↔ CLASS DESK REGIONS (QHH attendance schema)


def _clean_zone(zone: dict) -> dict:
    zone_type = str(zone.get("type", "normal") or "normal")
    if zone_type == "oriented":
        cx = float(zone.get("cx", 0.5))
        cy = float(zone.get("cy", 0.5))
        width = float(zone.get("w", 0.1))
        height = float(zone.get("h", 0.1))
        return {
            "type": "oriented",
            "cx": round(cx, 6),
            "cy": round(cy, 6),
            "w": round(width, 6),
            "h": round(height, 6),
            "angle": round(float(zone.get("angle", 0.0)), 2),
            # QHH consumers understand x/y/w/h. These are the enclosing AABB.
            "x": round(max(0.0, cx - width / 2), 6),
            "y": round(max(0.0, cy - height / 2), 6),
        }
    return {
        "type": "normal",
        "x": round(float(zone.get("x", 0.0)), 6),
        "y": round(float(zone.get("y", 0.0)), 6),
        "w": round(float(zone.get("w", 0.0)), 6),
        "h": round(float(zone.get("h", 0.0)), 6),
    }


def _region_zone(region: dict) -> dict:
    if region.get("type") == "oriented" or "angle" in region:
        return {
            "type": "oriented",
            "cx": float(region.get("cx", float(region.get("x", 0)) + float(region.get("w", 0)) / 2)),
            "cy": float(region.get("cy", float(region.get("y", 0)) + float(region.get("h", 0)) / 2)),
            "w": float(region.get("w", 0)),
            "h": float(region.get("h", 0)),
            "angle": float(region.get("angle", 0)),
        }
    return {
        "type": "normal",
        "x": float(region.get("x", 0)),
        "y": float(region.get("y", 0)),
        "w": float(region.get("w", 0)),
        "h": float(region.get("h", 0)),
    }


def _desk_number_from_region(region: dict) -> int:
    try:
        return int(region.get("deskNum"))
    except (TypeError, ValueError):
        label = str(region.get("label", "") or "")
        digits = "".join(ch for ch in label if ch.isdigit())
        return int(digits) if digits else 0


def get_desk_regions(cam_id: str, class_id: str) -> dict[int, dict]:
    config = get_camera_class_config(cam_id, class_id)
    result: dict[int, dict] = {}
    for region in config.get("regions", []) if isinstance(config, dict) else []:
        if not isinstance(region, dict):
            continue
        desk_num = _desk_number_from_region(region)
        if desk_num:
            result[desk_num] = dict(region)
    return result


def _embedded_students_for_class(class_id: str) -> list[dict]:
    students = []
    seen = set()
    for seat in list_seats(class_id):
        for slot in seat.get("slots", []):
            sid = str(slot.get("student_id", "") or "")
            if not sid or sid in seen:
                continue
            student = get_student(sid)
            if student:
                students.append({
                    "id": sid,
                    "studentCode": student.get("student_code", ""),
                    "fullName": student.get("name", ""),
                    "avatarUrl": student.get("face_image") or None,
                })
                seen.add(sid)
    return students


def set_desk_region(cam_id: str, class_id: str, desk_num: int, zone: dict) -> dict:
    config = get_camera_class_config(cam_id, class_id) or _empty_camera_class_config(cam_id, class_id)
    regions = [dict(r) for r in config.get("regions", []) if isinstance(r, dict)]
    existing = next((r for r in regions if _desk_number_from_region(r) == int(desk_num)), None)
    clean = _clean_zone(zone)
    seat = get_or_create_seat(class_id, int(desk_num))
    student_ids = [
        str(slot.get("student_id", "") or "")
        for slot in seat.get("slots", [])
        if slot.get("student_id")
    ]
    region = {
        "id": str(existing.get("id") if existing else uuid.uuid4()),
        "deskNum": int(desk_num),
        "label": f"Bàn {int(desk_num)}",
        **clean,
        "mapX": float(existing.get("mapX", clean.get("x", 0.0))) if existing else clean.get("x", 0.0),
        "mapY": float(existing.get("mapY", clean.get("y", 0.0))) if existing else clean.get("y", 0.0),
        "mapW": float(existing.get("mapW", clean.get("w", 0.12))) if existing else clean.get("w", 0.12),
        "mapH": float(existing.get("mapH", clean.get("h", 0.13))) if existing else clean.get("h", 0.13),
        "studentIds": student_ids,
    }
    if existing:
        regions[regions.index(existing)] = region
    else:
        regions.append(region)
    regions.sort(key=_desk_number_from_region)
    config["regions"] = regions
    config["students"] = _embedded_students_for_class(class_id)
    return save_camera_class_config(config)


def clear_desk_region(cam_id: str, class_id: str, desk_num: int) -> dict:
    config = get_camera_class_config(cam_id, class_id) or _empty_camera_class_config(cam_id, class_id)
    config["regions"] = [
        region for region in config.get("regions", [])
        if _desk_number_from_region(region) != int(desk_num)
    ]
    config["students"] = _embedded_students_for_class(class_id)
    return save_camera_class_config(config)


def monitor_seats(cam_id: str, class_id: str) -> list[dict]:
    """Return seat assignments enriched with the selected camera's desk regions."""
    if cam_id and class_id and not get_camera_class_config(cam_id, class_id):
        if class_id in _camera_class_ids(cam_id):
            try:
                _sync_camera_class(cam_id, class_id)
            except redis.exceptions.ResponseError as exc:
                if not _is_misconf_error(exc):
                    raise
                print(
                    "[redis] skip camera-class sync because Redis writes are "
                    f"blocked by MISCONF: cam={cam_id} class={class_id}",
                    flush=True,
                )
    regions = get_desk_regions(cam_id, class_id) if cam_id and class_id else {}
    result = []
    for raw_seat in list_seats(class_id) if class_id else []:
        seat = dict(raw_seat)
        region = regions.get(int(seat.get("desk_num", 0)))
        seat["zone"] = _region_zone(region) if region else {}
        # Slots carry assignments only. Region membership belongs to the desk.
        seat["slots"] = [dict(slot, zone={}) for slot in seat.get("slots", [])]
        result.append(seat)
    return result


def sync_class_camera_configs(class_id: str):
    """Keep QHH camera-class JSON consistent with local class/seat changes."""
    cam_ids = _smembers(_class_cameras_index_key(class_id))
    if not cam_ids:
        return
    seats = {int(s.get("desk_num", 0)): s for s in list_seats(class_id)}
    embedded = _embedded_students_for_class(class_id)
    classroom = get_classroom(class_id)
    for cam_id in cam_ids:
        config = get_camera_class_config(cam_id, class_id)
        if not config:
            continue
        regions = [
            region for region in config.get("regions", [])
            if _desk_number_from_region(region) in seats
        ]
        for region in regions:
            desk_num = _desk_number_from_region(region)
            seat = seats.get(desk_num, {})
            region["studentIds"] = [
                str(slot.get("student_id", "") or "")
                for slot in seat.get("slots", [])
                if slot.get("student_id")
            ]
        config["regions"] = regions
        config["students"] = embedded
        config["classCode"] = classroom.get("name", config.get("classCode", ""))
        camera = get_camera(cam_id)
        config["rtspPath"] = camera.get("rtspPath", camera.get("notes", ""))
        save_camera_class_config(config)
