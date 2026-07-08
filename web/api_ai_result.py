"""API trả về kết quả luồng AI theo schema QHH (camera + regions + students).

Endpoint: GET /api/ai/result?cameraId=<guid>&classId=<guid>

Response JSON:
{
  "cameraId": "guid",
  "classId":  "guid",
  "classCode": "10A1",
  "aiEnabled": true,
  "rtspChannel": 1,
  "rtspPath": "/cam/realmonitor?channel=1&subtype=1",
  "regions": [
    {
      "id": "desk-1",
      "label": "Bàn 1",
      "x": 0.1, "y": 0.2, "w": 0.15, "h": 0.1,
      "studentIds": ["<student-guid>"]
    }
  ],
  "students": [
    {
      "id": "<guid>",
      "studentCode": "HS001",
      "fullName": "Nguyễn Văn A",
      "avatarUrl": "/api/students/face?id=<guid>",
      "attention": false,
      "absence": false,
      "yaw":  30.0,
      "pitch": 30.0
    }
  ],
  "updatedAt": "2026-06-16T07:00:00.0000000Z"
}

Cách dùng từ web_server.py:

    from web.api_ai_result import build_ai_result_payload
    ...
    if parsed.path == "/api/ai/result":
        cam = query.get("cameraId", [""])[0]
        cls = query.get("classId",  [""])[0]
        return self._json(build_ai_result_payload(cam, cls, AI_MONITOR.status()))
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from db import redis_client as db


def _iso_utc(ts: float | None) -> str:
    """Format timestamp giống .NET DateTime ("yyyy-MM-ddTHH:mm:ss.fffffffZ")."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else datetime.now(tz=timezone.utc)
    base = dt.strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{dt.microsecond:06d}0Z"


def _avatar_url(student_id: str) -> str:
    return f"/api/students/face?id={student_id}" if student_id else ""


def _region_xywh(zone: dict) -> tuple[float, float, float, float]:
    """Trả về AABB (x, y, w, h) cho cả vùng normal lẫn oriented."""
    if not isinstance(zone, dict):
        return 0.0, 0.0, 0.0, 0.0
    w = float(zone.get("w", 0.0) or 0.0)
    h = float(zone.get("h", 0.0) or 0.0)
    if zone.get("type") == "oriented":
        cx = float(zone.get("cx", 0.0) or 0.0)
        cy = float(zone.get("cy", 0.0) or 0.0)
        return max(0.0, cx - w / 2), max(0.0, cy - h / 2), w, h
    return float(zone.get("x", 0.0) or 0.0), float(zone.get("y", 0.0) or 0.0), w, h


def _results_by_student(results: list[dict]) -> dict[str, dict]:
    """Index kết quả AI theo student_id (ưu tiên assigned, fallback recognized)."""
    indexed: dict[str, dict] = {}
    for item in results or []:
        sid = str(item.get("student_id") or item.get("assigned_student_id") or "")
        if not sid:
            sid = str(item.get("recognized_student_id") or "")
        if sid:
            indexed.setdefault(sid, item)
    return indexed


def _student_ai_fields(result: dict | None) -> dict:
    """Map kết quả AI thô → 4 trường attention/absence/yaw/pitch."""
    if not result:
        return {"attention": False, "absence": True, "yaw": 0.0, "pitch": 0.0}

    present = bool(result.get("present"))
    gaze_focused = result.get("gaze_focused")
    face_pose_ok = result.get("face_pose_ok")
    if gaze_focused is None and face_pose_ok is None:
        attention = False
    else:
        attention = bool(gaze_focused) if gaze_focused is not None else bool(face_pose_ok)

    yaw = result.get("gaze_yaw_deg")
    if yaw is None:
        yaw = result.get("face_yaw_score") or 0.0
    pitch = result.get("gaze_pitch_deg")
    if pitch is None:
        pitch = result.get("face_pitch_score") or 0.0

    return {
        "attention": bool(attention) and present,
        "absence": not present,
        "yaw": round(float(yaw or 0.0), 2),
        "pitch": round(float(pitch or 0.0), 2),
    }


def _load_classroom_config(camera_id: str, class_id: str) -> dict:
    """Đọc qhh:attendance:camera-class:{cam}:{cls} JSON."""
    import json as _json
    try:
        raw = db.get_client().get(
            f"qhh:attendance:camera-class:{camera_id}:{class_id}"
        )
    except Exception:
        return {}
    if not raw:
        return {}
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError:
        return {}


def _user_avatar(student_id: str) -> str:
    """Đọc avatar URL từ qhh:user:{id} hash. Fallback rỗng."""
    if not student_id:
        return ""
    try:
        v = db.get_client().hget(f"qhh:user:{student_id}", "avatar")
    except Exception:
        return ""
    if v is None:
        return ""
    return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)


def build_ai_result_payload(camera_id: str, class_id: str, ai_state: dict | None) -> dict[str, Any]:
    """Tổng hợp payload theo schema yêu cầu.

    Nguồn dữ liệu:
      • qhh:attendance:camera-class:{cam}:{cls}  → regions, students, rtsp..., classCode
      • qhh:user:{sid}                            → avatar URL chính thức
      • ai_state['results']                       → attention/absence/yaw/pitch per student
    """
    classroom = _load_classroom_config(camera_id, class_id)
    state = ai_state or {}
    indexed_results = _results_by_student(state.get("results") or [])

    # regions: đã đúng schema {id,label,x,y,w,h,studentIds}, copy thẳng.
    regions: list[dict[str, Any]] = []
    for region in classroom.get("regions") or []:
        regions.append({
            "id": region.get("id", ""),
            "label": region.get("label", ""),
            "x": round(float(region.get("x", 0.0) or 0.0), 6),
            "y": round(float(region.get("y", 0.0) or 0.0), 6),
            "w": round(float(region.get("w", 0.0) or 0.0), 6),
            "h": round(float(region.get("h", 0.0) or 0.0), 6),
            "studentIds": [str(s) for s in (region.get("studentIds") or [])],
        })

    students_payload: list[dict[str, Any]] = []
    for stu in classroom.get("students") or []:
        sid = str(stu.get("id", "") or "")
        if not sid:
            continue
        ai_fields = _student_ai_fields(indexed_results.get(sid))
        avatar = stu.get("avatarUrl") or _user_avatar(sid)
        students_payload.append({
            "id": sid,
            "studentCode": stu.get("studentCode") or stu.get("student_code") or "",
            "fullName": stu.get("fullName") or stu.get("name") or "",
            "avatarUrl": avatar,
            **ai_fields,
        })

    rtsp_channel = classroom.get("rtspChannel") or 1
    rtsp_path = classroom.get("rtspPath") or ""

    return {
        "cameraId": camera_id,
        "classId": class_id,
        "classCode": classroom.get("classCode", ""),
        "aiEnabled": bool(classroom.get("aiEnabled", state.get("active", False))),
        "rtspChannel": int(rtsp_channel or 1),
        "rtspPath": str(rtsp_path or ""),
        "regions": regions,
        "students": students_payload,
        "updatedAt": _iso_utc(state.get("updated_at")),
    }
