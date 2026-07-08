"""RTSP URL helpers — dùng chung giữa web_server (web-test) và video_recorder (production).

Tách ra module riêng để recorder standalone không phải import web_server (nặng, kéo theo
HTTP handler + AI monitor). Nguồn gốc: web_server.py `_force_mainstream_rtsp` +
`_camera_rtsp_url_for_class`.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

from db import redis_client as db


def force_mainstream_rtsp(value: str) -> str:
    """Ép RTSP URL/path kiểu Dahua về mainstream (subtype=0)."""
    raw = str(value or "").strip()
    if not raw:
        return raw
    parsed = urlsplit(raw)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    replaced = False
    next_query = []
    for key, item_value in query:
        if key.lower() == "subtype":
            next_query.append((key, "0"))
            replaced = True
        else:
            next_query.append((key, item_value))
    if not replaced and "realmonitor" in parsed.path.lower():
        next_query.append(("subtype", "0"))
    if not next_query and not parsed.query:
        return raw
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(next_query, doseq=True),
            parsed.fragment,
        )
    )


def build_camera_rtsp_url(camera_id: str, class_id: str) -> str:
    """Dựng RTSP URL mainstream cho (camera, class). "" nếu thiếu thông tin.

    Ưu tiên rtspPath trong camera-class config, rồi camera.rtspPath/notes.
    """
    camera = db.get_camera(camera_id)
    if not camera:
        return ""
    config = db.get_camera_class_config(camera_id, class_id) if class_id else {}
    path = str(
        config.get("rtspPath")
        or camera.get("rtspPath")
        or camera.get("notes")
        or ""
    ).strip()
    data = dict(camera, rtspPath=path)
    explicit = str(data.get("url", "") or "").strip()
    if explicit and not path:
        return force_mainstream_rtsp(explicit)
    host = str(data.get("ipAddress", "") or "").strip()
    if not host:
        return explicit
    port = str(data.get("port", "554") or "554").strip()
    username = str(data.get("username", "") or "")
    password = str(data.get("password", "") or "")
    if not path:
        path = "/cam/realmonitor?channel=1&subtype=0"
    path = force_mainstream_rtsp(path)
    if not path.startswith("/"):
        path = "/" + path
    auth = ""
    if username:
        auth = quote(username, safe="")
        if password:
            auth += ":" + quote(password, safe="")
        auth += "@"
    return f"rtsp://{auth}{host}:{port}{path}"
