"""Process one local clip file end-to-end.

Reads camera-class config from `qhh:attendance:camera-class:{cam}:{cls}`,
runs the existing AI pipeline (`_detect` from `workers/camera_worker.py`)
on every frame of the clip, then aggregates per-student stats and decides
PRESENT / ABSENT according to `local.presence_ratio` in `config.json`.
Presence only counts frames where the student is INSIDE their assigned
region — showing up in someone else's seat still counts as ABSENT.

Pipeline reuse: builds a Qt-free engine the same way `web_server.py` does
via `WebDetectionEngine`, so we get `_detect()` and `_shared_models()` for
free without rewriting geometry.

Avatar handling: `build_face_gallery` reads `student["face_image"]` as a
local file path, but QHH camera-class JSON only has `avatarUrl`. We
download avatars once into `local.face_cache_dir` and rewrite the field
before calling the gallery builder.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from config_loader import env_or_config
from db import redis_client as db


CAM_CLASS_KEY = "qhh:attendance:camera-class:{cam}:{cls}"

# Threshold thấp hơn chỉ để gán track_id → student_id (suy ra có mặt).
# Không dùng để xác nhận tên chính thức (vẫn dùng FACE_MATCH_THRESHOLD=0.55).
_TRACK_ASSIGN_THRESHOLD = float(os.getenv("TRACK_ASSIGN_THRESHOLD", "0.35"))

# Default thresholds — overridable in config.json -> local.*
_DEFAULTS = {
    "yaw_thresh_deg": 25.0,
    "pitch_thresh_deg": 20.0,
    "distracted_ratio_alert": 0.5,
    "presence_ratio": 0.6,
    "assigned_seat_ratio": 0.5,
    "detection_mode": "centerpoint",
    "face_cache_dir": ".cache/face_avatars",
    "write_annotated_video": True,
    "annotated_dir": "detection",
    # Thư mục lưu ảnh frame mất tập trung + base URL public để build path trả ra.
    "distraction_snapshot_dir": "detection/distraction",
    "distraction_snapshot_base_url": "",
}


def _cfg(key: str, env_name: str | None = None):
    return env_or_config(env_name or key.upper(), "local", key, _DEFAULTS[key])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Camera-class config ────────────────────────────────────────────────


def load_camera_class(cam_id: str, cls_id: str) -> dict:
    raw = db.get_client().get(CAM_CLASS_KEY.format(cam=cam_id, cls=cls_id))
    if not raw:
        raise RuntimeError(
            f"missing camera-class config qhh:attendance:camera-class:{cam_id}:{cls_id}"
        )
    return json.loads(raw)


# ── Avatar download + face_image path patch ────────────────────────────


def _avatar_cache_dir() -> Path:
    raw = _cfg("face_cache_dir")
    p = Path(str(raw)).expanduser()
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_avatar(url: str, dest: Path) -> bool:
    """Best-effort download; returns False on any failure (no exception)."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        import httpx
        with httpx.Client(timeout=10.0, follow_redirects=True) as cli:
            resp = cli.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        return dest.exists() and dest.stat().st_size > 0
    except Exception:
        return False


def _lookup_user_registry(student_id: str) -> dict:
    """Đọc hồ sơ user từ qhh:user:{id} hash do qhh-server publish.

    Trả {} nếu không có. Bytes → str. Là nguồn duy nhất nắm `avatar` URL
    chính thức — dùng trước khi fallback sang field trong classroom JSON.
    """
    try:
        raw = db.get_client().hgetall(f"qhh:user:{student_id}")
    except Exception:
        return {}
    out = {}
    for k, v in (raw or {}).items():
        k = k.decode() if isinstance(k, (bytes, bytearray)) else k
        v = v.decode() if isinstance(v, (bytes, bytearray)) else v
        out[k] = v
    return out


def _materialise_face_images(students: list[dict]) -> list[dict]:
    """Return copies of students with `face_image` populated as a local path.

    Nguồn URL ảnh ưu tiên theo thứ tự:
        1. `face_image` (absolute path) đã có sẵn trong dict
        2. Registry Redis qhh:user:{id}.avatar
        3. `avatarUrl` trong dict (legacy)
    `https://` được tải về cache, `file://` dùng path trực tiếp.
    """
    cache = _avatar_cache_dir()
    out: list[dict] = []
    for s in students:
        sid = str(s.get("id", "") or "")
        if not sid:
            continue
        new = dict(s)
        registry = _lookup_user_registry(sid)
        new.setdefault("name", registry.get("fullName") or s.get("fullName", ""))
        new.setdefault("student_code", s.get("studentCode", ""))
        existing_path = str(s.get("face_image", "") or "")
        url = str(registry.get("avatar") or s.get("avatarUrl", "") or "")
        if existing_path and Path(existing_path).exists():
            new["face_image"] = existing_path
        elif url:
            scheme = urlparse(url).scheme.lower()
            if scheme in ("http", "https"):
                stem = hashlib.md5(f"{sid}|{url}".encode()).hexdigest()
                ext = Path(urlparse(url).path).suffix.lower() or ".jpg"
                if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                    ext = ".jpg"
                dest = cache / f"{stem}{ext}"
                new["face_image"] = str(dest) if _download_avatar(url, dest) else ""
            elif scheme in ("", "file"):
                local = url[len("file://"):] if url.startswith("file://") else url
                new["face_image"] = local if Path(local).exists() else ""
            else:
                new["face_image"] = ""
        else:
            new["face_image"] = ""
        out.append(new)
    return out


# ── Region → seat (schema mà _detect mong đợi) ─────────────────────────


def regions_to_seats(regions: list[dict]) -> tuple[list[dict], dict[int, dict]]:
    """Convert camera-class `regions[]` to the `seats` shape `_detect()` uses.

    Returns (seats, region_meta_by_desk_num):
      seats[i] = {"desk_num": int, "zone": {...}, "slots": [...]}
      region_meta_by_desk_num[desk_num] = original region dict (id/label/...)
    """
    seats: list[dict] = []
    meta: dict[int, dict] = {}
    for idx, region in enumerate(regions, start=1):
        desk_num = idx
        meta[desk_num] = region
        slot_num = 1
        slots = []
        for sid in (region.get("studentIds") or []):
            slots.append({
                "slot_num": slot_num,
                "student_id": str(sid),
                "anchor": None,
            })
            slot_num += 1
        zone = _region_to_zone(region)
        seats.append({"desk_num": desk_num, "zone": zone, "slots": slots})
    return seats, meta


def _region_to_zone(region: dict) -> dict:
    """Convert `{x,y,w,h}` (0..1 normalized) → an oriented rect zone.

    The `_detect()` code expects `_zone_corners_and_aabb(zone, w, h)` to
    handle the zone — using type='rect' with normalized coordinates
    matches the shape produced by existing seat-editor code.
    """
    import os as _os
    if _os.getenv("CLIP_ZONE_FULL_FRAME", "0") == "1":
        return {"type": "rect", "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0, "norm": True}
    x = float(region.get("x", 0.0))
    y = float(region.get("y", 0.0))
    w = float(region.get("w", 0.0))
    h = float(region.get("h", 0.0))
    return {
        "type": "rect",
        "x": x, "y": y, "w": w, "h": h,
        "norm": True,
    }


def _point_in_zone(cx: float, cy: float, zone: dict | None,
                   frame_w: int, frame_h: int) -> bool:
    """True nếu điểm (cx, cy) pixel nằm trong zone rect của region.

    Zone từ `_region_to_zone` là rect normalized 0..1 (`norm: True`);
    phòng hờ zone pixel (norm falsy) cũng xử lý được.
    """
    if not zone:
        return False
    x = float(zone.get("x", 0.0))
    y = float(zone.get("y", 0.0))
    w = float(zone.get("w", 0.0))
    h = float(zone.get("h", 0.0))
    if zone.get("norm", True):
        x *= frame_w; y *= frame_h; w *= frame_w; h *= frame_h
    return (x <= cx <= x + w) and (y <= cy <= y + h)


# ── Gaze wedge attention (plan_distraction.md v2 + supplement §6c) ─────
# Tham số tune qua env; L/R/TL/TR KHÔNG có default — bắt buộc đến từ
# camera-class config trong Redis (server bổ sung khi cấu hình TKB).
_CENTRAL_CONE_RAD = math.radians(float(os.getenv("CENTRAL_CONE_DEG", "10")))
_BOARD_MARGIN_PX = float(os.getenv("BOARD_MARGIN_PX", "80"))
_VFOV_RAD = math.radians(float(os.getenv("CAMERA_VFOV_DEG", "55")))
_LOOKUP_MARGIN_RAD = math.radians(float(os.getenv("LOOKUP_MARGIN_DEG", "5")))
# Distraction chỉ "tính" khi kéo dài đủ N frame LIÊN TỤC (theo student).
_DISTRACT_ALERT_FRAMES = int(os.getenv("DISTRACT_ALERT_FRAMES", "100"))


def _board_from_config(classroom: dict) -> dict | None:
    """Đọc boardLine (bắt buộc) + pitchLimit (tùy chọn) từ camera-class config.

    Schema server bổ sung khi query thời khóa biểu:
        "boardLine":  {"L": [x, y], "R": [x, y]}
        "pitchLimit": {"TL": [x, y], "TR": [x, y]}
    KHÔNG tự khởi tạo giá trị mặc định — thiếu boardLine → trả None và
    pipeline giữ nguyên chế độ ngưỡng yaw/pitch cũ (backward compatible).
    """
    bl = classroom.get("boardLine") or {}
    L = bl.get("L")
    R = bl.get("R")
    if not L or not R or len(L) < 2 or len(R) < 2:
        return None
    board = {
        "L": (float(L[0]), float(L[1])),
        "R": (float(R[0]), float(R[1])),
        "t_y": None,
    }
    pl = classroom.get("pitchLimit") or {}
    TL = pl.get("TL")
    TR = pl.get("TR")
    if TL and TR and len(TL) >= 2 and len(TR) >= 2:
        board["t_y"] = (float(TL[1]) + float(TR[1])) / 2.0
    return board


def _gaze_attention_state(O: tuple[float, float], yaw_rad: float,
                          pitch_rad: float, board: dict,
                          frame_h: int) -> str:
    """'Focused' | 'LookingUp' | 'Distracted' theo plan §8.

    Pipeline: central cone (§5) → pitch limit camera-frame (§6c)
    → ray-hit đoạn bảng L-R (§6b).
    """
    dx = -math.sin(yaw_rad) * math.cos(pitch_rad)
    dy = -math.sin(pitch_rad)
    if math.hypot(dx, dy) < math.sin(_CENTRAL_CONE_RAD):
        return "Focused"                     # nhìn gần trục camera
    t_y = board.get("t_y")
    if t_y is not None:
        alpha_top = (O[1] - t_y) * (_VFOV_RAD / float(frame_h))
        if pitch_rad > alpha_top + _LOOKUP_MARGIN_RAD:
            return "LookingUp"
    L, R = board["L"], board["R"]
    if abs(dy) < 1e-6:
        return "Distracted"
    t = (L[1] - O[1]) / dy
    if t <= 0:
        return "Distracted"                  # gaze ngược hướng bảng
    x_hit = O[0] + dx * t
    if (L[0] - _BOARD_MARGIN_PX) <= x_hit <= (R[0] + _BOARD_MARGIN_PX):
        return "Focused"
    return "Distracted"


def assigned_desk_for_students(seats: list[dict]) -> dict[str, int]:
    """student_id -> desk_num it is assigned to in this clip's class config."""
    out: dict[str, int] = {}
    for seat in seats:
        d = int(seat["desk_num"])
        for slot in seat.get("slots") or []:
            sid = str(slot.get("student_id", "") or "")
            if sid:
                out[sid] = d
    return out


# ── Per-student aggregator ─────────────────────────────────────────────


class _StudentAgg:
    __slots__ = (
        "student_id", "name", "code", "assigned_desk_num", "avatar_url",
        "frames_present", "frames_in_assigned_desk",
        "sum_face_score", "n_face_score",
        "sum_yaw", "sum_pitch", "n_gaze",
        "distracted_frames",
    )

    def __init__(self, sid: str, student_meta: dict, assigned_desk: int):
        self.student_id = sid
        self.name = student_meta.get("fullName") or student_meta.get("name") or ""
        self.code = student_meta.get("studentCode") or student_meta.get("student_code") or ""
        self.avatar_url = student_meta.get("avatarUrl") or ""
        self.assigned_desk_num = assigned_desk
        self.frames_present = 0
        self.frames_in_assigned_desk = 0
        self.sum_face_score = 0.0
        self.n_face_score = 0
        self.sum_yaw = 0.0
        self.sum_pitch = 0.0
        self.n_gaze = 0
        self.distracted_frames = 0


# ── Pipeline lazy singleton (1 bundle per worker process) ──────────────


_ENGINE = None


def _get_engine():
    """Return a Qt-free engine reusing AIDetectionWorker logic.

    Same pattern as `web_server.WebDetectionEngine` but built locally so the
    Celery worker doesn't have to import `web_server` (which would pull in
    Middleware SHM reader and other live-camera-only deps).
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    import inspect as _inspect
    import os as _os
    import threading as _threading
    import numpy as _np

    from workers.camera_worker import AIDetectionWorker, _FpsAggregator, FPS_LOG_PERIOD
    from workers.gaze_estimator import DistractionTracker
    from workers.face_models import ArcFaceExtractor

    class _ClipEngine:
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
            self._fps = _FpsAggregator(FPS_LOG_PERIOD, "ai-clip")
            self._fps_yolo = self._fps.get("yolo")
            self._fps_retina = self._fps.get("retina")
            self._fps_arc = self._fps.get("arc")
            self._fps_gaze = self._fps.get("gaze")
            self._fps_pipeline = self._fps.get("pipe")
            self._distraction = DistractionTracker(
                yaw_threshold_deg=float(_os.getenv("GAZE_YAW_THRESHOLD", "60")),
                pitch_threshold_deg=float(_os.getenv("GAZE_PITCH_THRESHOLD", "9999")),
                alert_after_sec=float(_os.getenv("GAZE_ALERT_AFTER", "2.5")),
                clear_after_sec=float(_os.getenv("GAZE_CLEAR_AFTER", "1.0")),
                ema_alpha=float(_os.getenv("GAZE_EMA_ALPHA", "0.35")),
            )
            self._seats = []
            self._students = {}
            self._seat_lookup = {}
            self._face_gallery = []
            self._gallery_embeddings = _np.empty(
                (0, ArcFaceExtractor.EMBEDDING_DIM), dtype=_np.float32
            )
            self._ground_transform = None
            self._zone_ground_polys = {}
            self._calibration_dirty = True
            self._detection_mode = "centerpoint"

        def __getattr__(self, name):
            descriptor = _inspect.getattr_static(AIDetectionWorker, name, None)
            if isinstance(descriptor, staticmethod):
                return descriptor.__func__
            if isinstance(descriptor, classmethod):
                return descriptor.__get__(None, AIDetectionWorker)
            if callable(descriptor):
                return descriptor.__get__(self, type(self))
            raise AttributeError(name)

        def update_context(self, seats, students):
            from workers.face_models import build_face_gallery
            students_by_id = {
                str(s.get("id", "")): dict(s)
                for s in students if s.get("id")
            }
            seat_lookup = {}
            for seat in seats:
                desk_num = int(seat.get("desk_num", 0) or 0)
                for slot in seat.get("slots", []) or []:
                    sid = str(slot.get("student_id", "") or "")
                    if sid:
                        seat_lookup[sid] = (
                            desk_num, int(slot.get("slot_num", 1) or 1)
                        )
            gallery = build_face_gallery(
                list(students_by_id.values()), self._arcface, self._retinaface,
            )
            self._seats = seats
            self._students = students_by_id
            self._seat_lookup = seat_lookup
            self._face_gallery = gallery
            self._gallery_embeddings = (
                _np.stack([item["embedding"] for item in gallery]).astype(
                    _np.float32, copy=False
                )
                if gallery else _np.empty(
                    (0, ArcFaceExtractor.EMBEDDING_DIM), dtype=_np.float32
                )
            )
            self._calibration_dirty = True

    _ENGINE = _ClipEngine()
    return _ENGINE


# ── Snapshot annotation helper ─────────────────────────────────────────

def _draw_snap_overlay(
    frame: "np.ndarray",
    results: list[dict],
    tracks: "np.ndarray | list",
    track_to_student: dict,
    students_by_id: dict,
) -> "np.ndarray":
    """Vẽ YOLO person bbox + SORT track ID + RetinaFace box + tên lên frame snapshot."""
    img = frame.copy()
    # Vẽ SORT track bbox (cam) + tên student nếu đã biết
    for tk in tracks:
        tx1, ty1, tx2, ty2, ttid = int(tk[0]), int(tk[1]), int(tk[2]), int(tk[3]), int(tk[4])
        sid = track_to_student.get(ttid)
        name = students_by_id.get(sid, {}).get("name") or (sid[:8] if sid else "?")
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 165, 255), 2)
        cv2.putText(img, f"T{ttid} {name}", (tx1, ty1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    # Vẽ person bbox (xanh lá nhạt) + face box (xanh/đỏ) từ results
    for r in results:
        pb = r.get("person_bbox")
        if pb is not None:
            try:
                px, py, pw, ph = (int(float(v)) for v in pb[:4])
                cv2.rectangle(img, (px, py), (px + pw, py + ph), (100, 255, 100), 1)
            except (TypeError, ValueError):
                pass
        fb = r.get("face_box")
        if fb is not None:
            try:
                fx, fy, fw, fh = (int(float(v)) for v in fb[:4])
                rid = str(r.get("recognized_student_id") or "")
                score = r.get("recognition_score")
                matched = bool(rid)
                color = (0, 220, 0) if matched else (40, 40, 220)
                cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), color, 2)
                name = students_by_id.get(rid, {}).get("name") or rid[:8] if rid else "?"
                label = f"{name} {score:.2f}" if (score is not None and matched) else f"? {score:.2f}" if score else "?"
                cv2.putText(img, label, (fx, fy - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            except (TypeError, ValueError):
                pass
    return img


# ── Main entrypoint ────────────────────────────────────────────────────


def run_clip(video_path: str, cam_id: str, cls_id: str) -> dict:
    classroom = load_camera_class(cam_id, cls_id)
    regions = classroom.get("regions") or []
    raw_students = classroom.get("students") or []

    seats, region_meta = regions_to_seats(regions)
    student_to_desk = assigned_desk_for_students(seats)
    # desk_num → zone rect, để check vị trí track có nằm trong region gán không.
    desk_zone = {int(s["desk_num"]): (s.get("zone") or {}) for s in seats}

    # Build face gallery using only students assigned to a desk in this class.
    materialised = _materialise_face_images(raw_students)
    engine = _get_engine()
    engine.update_context(seats, materialised)
    students_by_id = {str(s["id"]): s for s in materialised if s.get("id")}

    # Aggregator seeded for every assigned student so absent ones still emit a row.
    agg: dict[str, _StudentAgg] = {
        sid: _StudentAgg(sid, students_by_id.get(sid, {}), desk_num)
        for sid, desk_num in student_to_desk.items()
    }

    yaw_th = float(_cfg("yaw_thresh_deg"))
    pitch_th = float(_cfg("pitch_thresh_deg"))
    detection_mode = str(_cfg("detection_mode"))

    # ── Gaze mode: wedge nếu config Redis có boardLine, ngược lại legacy ──
    board = _board_from_config(classroom)
    if board is not None:
        print(f"[clip] gaze mode: WEDGE boardLine L={board['L']} R={board['R']} "
              f"pitch_limit_y={board['t_y']}", flush=True)
    else:
        print(f"[clip] gaze mode: legacy threshold yaw>{yaw_th} pitch>{pitch_th} "
              f"(no boardLine in camera-class config)", flush=True)
    # Đếm distraction LIÊN TỤC theo student_id (bền qua mất mặt tạm thời
    # nhờ track_to_student). Chỉ khi đủ _DISTRACT_ALERT_FRAMES liên tục
    # mới coi là mất tập trung thật (attention=false) + chụp ảnh.
    distract_run: dict[str, int] = {}
    sustained_alerted: set[str] = set()

    def _result_distracted(r: dict, frame_h: int) -> bool | None:
        """True/False = distracted/focused frame này; None = không có gaze."""
        yaw = r.get("gaze_yaw_deg")
        pitch = r.get("gaze_pitch_deg")
        if yaw is None or pitch is None:
            return None
        if board is not None:
            fb = r.get("face_box")
            if fb is None:
                return None
            fx, fy, fw2, fh2 = (float(v) for v in fb[:4])
            O = (fx + fw2 / 2.0, fy + fh2 / 2.0)
            state = _gaze_attention_state(
                O, math.radians(float(yaw)), math.radians(float(pitch)),
                board, frame_h,
            )
            return state != "Focused"
        return abs(float(yaw)) > yaw_th or abs(float(pitch)) > pitch_th

    # ── SORT tracker — fill khi face mất tạm thời (quay đầu, cúi, occlusion).
    # max_age = số frame được giữ track sau khi mất detection.
    # Với 25 fps, 60 frame ≈ 2.4s — đủ để bỏ qua ngắt face do quay đầu/cúi.
    import sys as _sys
    _sort_dir = str(Path(__file__).resolve().parent / "sort")
    if _sort_dir not in _sys.path:
        _sys.path.insert(0, _sort_dir)
    from sort import Sort  # type: ignore
    tracker = Sort(
        max_age=int(os.getenv("SORT_MAX_AGE", "60")),
        min_hits=int(os.getenv("SORT_MIN_HITS", "3")),
        iou_threshold=float(os.getenv("SORT_IOU_THRESHOLD", "0.3")),
    )
    # track_id → student_id (chỉ gán khi ArcFace match đủ tin cậy)
    track_to_student: dict[int, str] = {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture cannot open {video_path}")
    fps_decl = cap.get(cv2.CAP_PROP_FPS) or 30.0
    unmatched_faces = 0
    frames = 0
    t0 = time.time()

    # Annotated video writer (lazy — open on first frame so we know W,H,fps).
    write_video = str(_cfg("write_annotated_video", "QHH_AI_RESULT_VIDEO_ON")).strip().lower() not in {"0", "false", "no"}
    annotated_dir = Path(str(_cfg("annotated_dir"))).expanduser()
    if not annotated_dir.is_absolute():
        annotated_dir = Path(__file__).resolve().parent / annotated_dir
    annotated_dir.mkdir(parents=True, exist_ok=True)
    import hashlib as _hashlib
    annotated_path = annotated_dir / f"{_hashlib.md5(video_path.encode()).hexdigest()}.mp4"
    video_writer = None

    # ── Distraction snapshots ───────────────────────────────────────────
    # Folder đặt theo tên video (stem) — mỗi video 1 folder ảnh riêng.
    video_stem = Path(video_path).stem
    snap_dir_root = Path(str(_cfg("distraction_snapshot_dir"))).expanduser()
    if not snap_dir_root.is_absolute():
        snap_dir_root = Path(__file__).resolve().parent / snap_dir_root
    snap_dir = snap_dir_root / video_stem
    snap_base_url = str(_cfg("distraction_snapshot_base_url")).rstrip("/")
    # student_id đã có ảnh cảnh báo rồi → không cần chụp lại.
    distracted_captured: set[str] = set()
    # Danh sách ảnh đã lưu: [{path, url, frame, studentIds, ...}]
    distraction_snapshots: list[dict] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1
            annotated, aggregated = engine._detect(
                frame, seats, students_by_id, engine._seat_lookup,
                engine._face_gallery, cal_dirty=(frames == 1),
                detection_mode=detection_mode,
            )
            # _detect returns per-desk aggregated list; flatten to per-slot results
            results = []
            for desk in aggregated:
                results.extend(desk.get("slot_results") or [])
                results.extend(desk.get("extra_results") or [])
            # DEBUG — log 1 frame mỗi 60 frame để xem structure
            if frames == 30:
                for _r in results:
                    print(f"[clip-debug] frame=30 face_found={_r.get('face_found')} "
                          f"recognized_id={_r.get('recognized_student_id')!r} "
                          f"score={_r.get('recognition_score')} "
                          f"best_cand={_r.get('best_candidate_id')!r} "
                          f"person_bbox={_r.get('person_bbox')}", flush=True)
            if write_video:
                _draw_debug_overlay(annotated, results, frame_idx=frames)
                _draw_raw_face_overlay(
                    annotated, frame, engine, frame_idx=frames,
                )
                if video_writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(annotated_path), fourcc, float(fps_decl), (w, h),
                    )
                video_writer.write(annotated)
            # ── SORT update: gom person_bbox của mọi result trong frame ──
            person_dets = []
            result_with_bbox: list[tuple[tuple[int,int,int,int], dict]] = []
            for r in results:
                pb = r.get("person_bbox")
                if pb is None:
                    continue
                # pb là (x, y, w, h) từ camera_worker — convert sang (x1,y1,x2,y2)
                try:
                    px, py, pw, ph = (float(v) for v in pb[:4])
                    x1, y1, x2, y2 = px, py, px + pw, py + ph
                except (TypeError, ValueError):
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue
                score = float(r.get("recognition_score") or 0.5)
                person_dets.append([x1, y1, x2, y2, score])
                result_with_bbox.append(((int(x1), int(y1), int(x2), int(y2)), r))

            dets_arr = (np.asarray(person_dets, dtype=np.float32)
                        if person_dets else np.empty((0, 5), dtype=np.float32))
            tracks = tracker.update(dets_arr)  # [[x1,y1,x2,y2,track_id], ...]

            # Map mỗi result → track_id (nearest IoU với track active của frame).
            def _iou(a, b):
                ax1, ay1, ax2, ay2 = a
                bx1, by1, bx2, by2 = b
                ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                ua = (ax2-ax1) * (ay2-ay1) + (bx2-bx1) * (by2-by1) - inter
                return (inter / ua) if ua > 0 else 0.0

            result_to_track: dict[int, int] = {}  # id(result) → track_id
            for det_bbox, r in result_with_bbox:
                best_tid, best_iou = -1, 0.0
                for tk in tracks:
                    tx1, ty1, tx2, ty2, tid = tk
                    iou = _iou(det_bbox, (tx1, ty1, tx2, ty2))
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = int(tid)
                if best_tid >= 0 and best_iou >= 0.3:
                    result_to_track[id(r)] = best_tid

            # ── Học track_id → student_id ────────────────────────────────
            # Dùng 2 mức threshold:
            #   - FACE_MATCH_THRESHOLD (0.55): xác nhận tên chính thức (camera_worker)
            #   - TRACK_ASSIGN_THRESHOLD (0.35): đủ để gán track → frames_present tăng
            #     kể cả khi camera xa/góc lệch làm sim thấp hơn mức nhận diện chính.
            for r in results:
                if not r.get("face_found"):
                    continue
                tid = result_to_track.get(id(r))
                if tid is None:
                    continue
                rid = str(r.get("recognized_student_id") or "")
                if rid:
                    # Đã vượt FACE_MATCH_THRESHOLD → gán chắc chắn
                    track_to_student[tid] = rid
                elif tid not in track_to_student:
                    # Chưa vượt FACE_MATCH_THRESHOLD — thử với threshold thấp hơn
                    # dùng best_candidate_id (best ArcFace match dù dưới ngưỡng chính)
                    score = r.get("recognition_score")
                    cand = str(r.get("best_candidate_id") or "")
                    if score is not None and float(score) >= _TRACK_ASSIGN_THRESHOLD and cand:
                        track_to_student[tid] = cand
                        print(f"[clip] track T{tid} → {cand[:8]} via low-threshold (sim={score:.3f})", flush=True)

            # ── Fallback: ArcFace trên full frame để gán track chưa biết tên ──
            # camera_worker chạy RetinaFace trên YOLO crop nhỏ → dễ fail khi
            # người ngồi xa. Chạy thêm RetinaFace trên full frame, map face
            # center vào SORT track, gán track_to_student một lần là đủ.
            unassigned_tids = [int(tk[4]) for tk in tracks if int(tk[4]) not in track_to_student]
            if unassigned_tids and engine._gallery_embeddings.shape[0] > 0:
                try:
                    face_dets = engine._retinaface.detect_all(frame)
                    if face_dets:
                        aligned = [d["aligned_face"] for d in face_dets if d.get("aligned_face") is not None]
                        if aligned:
                            embeds = engine._arcface.extract(aligned)
                            gallery = engine._face_gallery
                            gal_embs = engine._gallery_embeddings  # (N, 512)
                            for det, emb in zip(face_dets, embeds):
                                if emb is None:
                                    continue
                                # tanh-calibrated similarity vs gallery
                                emb_n = emb / (np.linalg.norm(emb) + 1e-8)
                                gal_n = gal_embs / (np.linalg.norm(gal_embs, axis=1, keepdims=True) + 1e-8)
                                dists = np.linalg.norm(gal_n - emb_n, axis=1)
                                sims = (np.tanh((1.23132175 - dists) * 6.602259425) + 1) / 2
                                best_idx = int(np.argmax(sims))
                                best_sim = float(sims[best_idx])
                                if best_sim < _TRACK_ASSIGN_THRESHOLD:
                                    continue
                                best_sid = str(gallery[best_idx].get("student_id") or gallery[best_idx].get("studentId") or "")
                                # Map face center → SORT track bbox
                                fb = det.get("face_box")  # (x, y, w, h)
                                if fb is None:
                                    continue
                                fx, fy, fw, fh = fb
                                cx = fx + fw // 2
                                cy = fy + fh // 2
                                for tk in tracks:
                                    tx1, ty1, tx2, ty2, ttid = int(tk[0]), int(tk[1]), int(tk[2]), int(tk[3]), int(tk[4])
                                    if ttid in track_to_student:
                                        continue
                                    if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                                        track_to_student[ttid] = best_sid
                                        print(f"[clip] full-frame ArcFace: track T{ttid} → {best_sid[:8]} (sim={best_sim:.3f})", flush=True)
                                        break
                except Exception as _e:
                    print(f"[clip] full-frame ArcFace error: {_e}", flush=True)

            # ── Tập student_ids "có mặt" trong frame (face thấy + track suy ra) ──
            present_sids: set[str] = set()
            for r in results:
                rid = str(r.get("recognized_student_id") or "")
                if r.get("face_found") and rid:
                    present_sids.add(rid)
            # sid → tâm bbox của track (để check nằm trong region gán).
            track_center_by_sid: dict[str, tuple[float, float]] = {}
            for tk in tracks:
                tid = int(tk[4])
                sid = track_to_student.get(tid)
                if sid:
                    present_sids.add(sid)
                    track_center_by_sid.setdefault(
                        sid,
                        ((float(tk[0]) + float(tk[2])) / 2.0,
                         (float(tk[1]) + float(tk[3])) / 2.0),
                    )

            seen_in_frame: set[str] = set()
            for r in results:
                rid = str(r.get("recognized_student_id") or "")
                if not r.get("face_found") or not rid:
                    if not r.get("assigned_student_id"):
                        unmatched_faces += 1
                    continue
                a = agg.get(rid)
                if a is None:
                    # Student detected but not assigned to any desk in this class.
                    unmatched_faces += 1
                    continue
                if rid in seen_in_frame:
                    continue
                seen_in_frame.add(rid)

                a.frames_present += 1
                if int(r.get("desk_num", -1)) == a.assigned_desk_num:
                    a.frames_in_assigned_desk += 1

                score = r.get("recognition_score")
                if score is not None:
                    a.sum_face_score += float(score)
                    a.n_face_score += 1

                yaw = r.get("gaze_yaw_deg")
                pitch = r.get("gaze_pitch_deg")
                if yaw is not None and pitch is not None:
                    a.sum_yaw += float(yaw)
                    a.sum_pitch += float(pitch)
                    a.n_gaze += 1
                    # Wedge (boardLine từ Redis) hoặc legacy threshold.
                    if _result_distracted(r, frame.shape[0]):
                        a.distracted_frames += 1

            # ── Fill present cho HS suy ra qua tracker nhưng KHÔNG có face
            # trong frame này (quay đầu / cúi / occlusion).
            # Đếm 1 lần/HS/frame; bỏ qua nếu loop face đã count.
            for sid in present_sids:
                if sid in seen_in_frame:
                    continue
                a = agg.get(sid)
                if a is None:
                    continue
                a.frames_present += 1
                # Track vẫn cho biết vị trí → check tâm bbox có nằm trong
                # region gán của HS không; đúng chỗ mới đếm in_assigned_desk.
                c = track_center_by_sid.get(sid)
                if c is not None:
                    fh, fw = frame.shape[:2]
                    if _point_in_zone(c[0], c[1],
                                      desk_zone.get(a.assigned_desk_num),
                                      fw, fh):
                        a.frames_in_assigned_desk += 1
                # Không có face → không cập nhật gaze.
                seen_in_frame.add(sid)

            # ── Distraction LIÊN TỤC theo student ───────────────────────
            # sid resolve qua recognized_id, fallback track_to_student —
            # face thấy nhưng sim dưới ngưỡng chính vẫn tính đúng người.
            distracted_now: set[str] = set()
            focused_now: set[str] = set()
            for r in results:
                if not r.get("face_found"):
                    continue
                sid = str(r.get("recognized_student_id") or "")
                if not sid:
                    tid = result_to_track.get(id(r))
                    sid = track_to_student.get(tid, "") if tid is not None else ""
                if not sid or sid not in agg:
                    continue
                d = _result_distracted(r, frame.shape[0])
                if d is None:
                    continue
                (distracted_now if d else focused_now).add(sid)

            # Frame Focused → reset chuỗi (attention lại true).
            # Frame mất mặt → giữ nguyên chuỗi (không cộng, không reset).
            for sid in focused_now - distracted_now:
                distract_run[sid] = 0
            for sid in distracted_now:
                distract_run[sid] = distract_run.get(sid, 0) + 1

            # Chỉ báo distraction + chụp ảnh khi chuỗi đạt ngưỡng liên tục.
            new_offenders = {
                sid for sid in distracted_now
                if distract_run[sid] >= _DISTRACT_ALERT_FRAMES
            } - distracted_captured
            if new_offenders:
                sustained_alerted |= new_offenders
                for _sid in sorted(new_offenders):
                    print(f"[clip] DISTRACTION ALERT: {_sid[:8]} distracted "
                          f">={_DISTRACT_ALERT_FRAMES} frames lien tuc "
                          f"(frame {frames})", flush=True)
            if new_offenders:
                snap_dir.mkdir(parents=True, exist_ok=True)
                fname = f"frame_{frames:06d}.jpg"
                fpath = snap_dir / fname
                _img = _draw_snap_overlay(frame, results, tracks, track_to_student, students_by_id)
                cv2.imwrite(str(fpath), _img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                rel = f"{cam_id}/{video_stem}/{fname}"
                url = f"{snap_base_url}/{rel}" if snap_base_url else rel
                distraction_snapshots.append({
                    "frame": frames,
                    "path": str(fpath),
                    "url": url,
                    "newStudentIds": sorted(new_offenders),
                    "allDistractedIds": sorted(distracted_now),
                })
                distracted_captured |= new_offenders
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()

    # Nếu không có snapshot nào (không detect face / không ai mất tập trung),
    # lưu 1 frame giữa clip làm bằng chứng "classroom frame" để absent student
    # vẫn có ảnh trong resultImageUrls.
    if not distraction_snapshots and frames > 0:
        _cap2 = cv2.VideoCapture(video_path)
        _mid = max(0, frames // 2)
        _cap2.set(cv2.CAP_PROP_POS_FRAMES, _mid)
        _ok, _fr = _cap2.read()
        _cap2.release()
        if _ok and _fr is not None:
            snap_dir.mkdir(parents=True, exist_ok=True)
            _fname = f"frame_{_mid:06d}.jpg"
            _fpath = snap_dir / _fname
            _fr_ann = _draw_snap_overlay(_fr, [], [], {}, students_by_id)
            cv2.imwrite(str(_fpath), _fr_ann, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _rel = f"{cam_id}/{video_stem}/{_fname}"
            _url = f"{snap_base_url}/{_rel}" if snap_base_url else _rel
            distraction_snapshots.append({
                "frame": _mid,
                "path": str(_fpath),
                "url": _url,
                "newStudentIds": [],
                "allDistractedIds": [],
            })

    snap_folder_url = (
        f"{snap_base_url}/{video_stem}" if snap_base_url else
        (str(snap_dir) if distraction_snapshots else "")
    )
    return _finalize(
        agg=agg,
        region_meta=region_meta,
        frames=frames,
        fps_decl=float(fps_decl),
        unmatched_faces=unmatched_faces,
        processing_ms=int((time.time() - t0) * 1000),
        video_path=video_path,
        cam_id=cam_id,
        cls_id=cls_id,
        annotated_video=str(annotated_path) if (write_video and annotated_path.exists()) else None,
        distraction_snapshots=distraction_snapshots,
        distraction_folder_url=snap_folder_url,
        sustained_alerted=sustained_alerted,
    )


# ── Raw overlay — chạy RetinaFace + ArcFace + Gaze trực tiếp trên ─────
#    frame để vẽ MỌI khuôn mặt (kể cả ngoài region), log similarity. ──


def _draw_raw_face_overlay(annotated, frame, engine, frame_idx: int = 0) -> None:
    import cv2 as _cv2
    import numpy as _np
    import os as _os
    try:
        dets = engine._retinaface.detect_all(frame)
    except Exception as e:
        if frame_idx % 60 == 0:
            print(f"[raw-overlay] retinaface error: {e}", flush=True)
        return
    if not dets:
        return

    aligned_list = [d["aligned_face"] for d in dets if d.get("aligned_face") is not None]
    embeds = engine._arcface.extract(aligned_list) if aligned_list else None
    gallery_embs = engine._gallery_embeddings if hasattr(engine, "_gallery_embeddings") else None
    has_gallery = gallery_embs is not None and len(gallery_embs) > 0
    gallery = engine._face_gallery or []

    yaws, pitches = [], []
    if engine._gaze is not None and aligned_list:
        try:
            crops = [d["aligned_face"] for d in dets]
            results = engine._gaze.estimate_batch(crops)
            yaws = [r[0] for r in results]
            pitches = [r[1] for r in results]
        except Exception as e:
            if frame_idx % 60 == 0:
                print(f"[raw-overlay] gaze error: {e}", flush=True)

    drawn = 0
    max_sim = -1.0
    max_label = ""
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(v) for v in d["loc"]]

        sim = None
        sid = ""
        name = ""
        if embeds is not None and has_gallery:
            e = embeds[i]
            e = e / max(float(_np.linalg.norm(e)), 1e-12)
            g = gallery_embs / (_np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-12)
            sims = g @ e
            best = int(sims.argmax())
            sim = float(sims[best])
            if sim >= float(getattr(engine, "FACE_MATCH_THRESHOLD", 0.55)):
                sid = str(gallery[best].get("student_id", ""))
                name = str(gallery[best].get("name", "") or sid[:8])

        if sid:
            color = (16, 200, 16)
            label = f"{name}"
        elif sim is not None:
            color = (40, 40, 220)
            label = "UNKNOWN"
        else:
            color = (200, 200, 200)
            label = "face"

        _cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        sim_txt = f"sim={sim:.2f}" if sim is not None else "sim=?"
        _cv2.putText(annotated, f"{label} {sim_txt}",
            (x1, max(y1 - 6, 14)),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, _cv2.LINE_AA)

        if i < len(yaws):
            yaw_deg = float(yaws[i])
            pitch_deg = float(pitches[i])
            yaw_rad = _np.radians(yaw_deg)
            pitch_rad = _np.radians(pitch_deg)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            length = int((x2 - x1))
            dx = int(-length * _np.sin(yaw_rad) * _np.cos(pitch_rad))
            dy = int(-length * _np.sin(pitch_rad))
            _cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            _cv2.arrowedLine(annotated, (cx, cy), (cx + dx, cy + dy),
                (0, 0, 255), 2, _cv2.LINE_AA, tipLength=0.25)
            _cv2.putText(annotated,
                f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}",
                (x1, y2 + 18),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 0, 255), 1, _cv2.LINE_AA)

        if sim is not None and sim > max_sim:
            max_sim = sim
            max_label = label
        drawn += 1

    log_every = int(_os.getenv("CLIP_LOG_EVERY_N_FRAMES", "30"))
    if drawn and (frame_idx % log_every == 0):
        print(f"[raw-overlay] frame={frame_idx:>4}  faces={drawn}  "
              f"max_sim={max_sim:.3f} ({max_label})", flush=True)


# ── Debug overlay — vẽ face bbox + arcface score + gaze arrow ─────────


def _draw_debug_overlay(annotated, results, frame_idx: int = 0) -> None:
    """Draw face boxes for ALL detected faces (not just `match_status=correct`).

    Colors:
      - GREEN  : correct (student match đúng bàn)
      - YELLOW : recognized but wrong seat / unassigned
      - RED    : face detected but unknown / unmatched
    Adds per-face: ArcFace similarity score, gaze yaw/pitch text, gaze arrow.
    Logs one summary line every N frames so worker log shows live scores.
    """
    import cv2 as _cv2
    import numpy as _np
    import os as _os

    drawn = 0
    max_sim = -1.0
    max_sim_name = ""
    for r in results:
        if not r.get("face_found"):
            continue
        status = r.get("match_status", "")
        rid = str(r.get("recognized_student_id") or "")
        if status == "correct":
            color = (16, 200, 16)         # green BGR
            label = r.get("recognized_name") or rid[:8]
        elif rid:
            color = (0, 200, 220)         # yellow
            label = f"WRONG: {r.get('recognized_name') or rid[:8]}"
        else:
            color = (40, 40, 220)         # red
            label = "UNKNOWN"

        # Track best similarity in this frame (for periodic log line).
        s = r.get("recognition_score")
        if isinstance(s, (int, float)) and s > max_sim:
            max_sim = float(s)
            max_sim_name = label

        fb = r.get("face_box")
        if fb is not None:
            fx, fy, fw, fh = (int(v) for v in fb)
            _cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), color, 2)

            # ArcFace similarity score above the face box.
            score_txt = f"sim={s:.2f}" if isinstance(s, (int, float)) else "sim=?"
            _cv2.putText(
                annotated, f"{label} {score_txt}",
                (fx, max(fy - 6, 14)),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, _cv2.LINE_AA,
            )

            # Gaze yaw/pitch text + red arrow from face center.
            yaw_deg = r.get("gaze_yaw_deg")
            pitch_deg = r.get("gaze_pitch_deg")
            if yaw_deg is not None and pitch_deg is not None:
                yaw_rad = float(_np.radians(yaw_deg))
                pitch_rad = float(_np.radians(pitch_deg))
                cx = int(fx + fw / 2)
                cy = int(fy + fh / 2)
                length = int(fw)
                dx = int(-length * _np.sin(yaw_rad) * _np.cos(pitch_rad))
                dy = int(-length * _np.sin(pitch_rad))
                _cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                _cv2.arrowedLine(
                    annotated, (cx, cy), (cx + dx, cy + dy),
                    (0, 0, 255), 2, _cv2.LINE_AA, tipLength=0.25,
                )
                _cv2.putText(
                    annotated,
                    f"yaw={yaw_deg:+.0f} pitch={pitch_deg:+.0f}",
                    (fx, fy + fh + 16),
                    _cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 255), 1, _cv2.LINE_AA,
                )

        # Person bbox (YOLO) — draw thin so it's clear from face bbox.
        pb = r.get("person_bbox")
        if pb is not None:
            px, py, pw, ph = (int(v) for v in pb)
            _cv2.rectangle(annotated, (px, py), (px + pw, py + ph), color, 1)

        drawn += 1

    # Periodic log line — visible in `docker compose logs worker`.
    log_every = int(_os.getenv("CLIP_LOG_EVERY_N_FRAMES", "30"))
    if drawn and (frame_idx % log_every == 0):
        print(
            f"[clip-overlay] frame={frame_idx:>4}  faces_drawn={drawn}  "
            f"max_sim={max_sim:.3f} ({max_sim_name})",
            flush=True,
        )


# ── Finalize → JSON payload (schema §3.3) ──────────────────────────────


def _finalize(*, agg, region_meta, frames, fps_decl, unmatched_faces,
              processing_ms, video_path, cam_id, cls_id,
              annotated_video=None, distraction_snapshots=None,
              distraction_folder_url="", sustained_alerted=None) -> dict:
    sustained_alerted = sustained_alerted or set()
    presence_ratio_th = float(_cfg("presence_ratio"))
    distracted_ratio_alert = float(_cfg("distracted_ratio_alert"))

    students_out: list[dict] = []
    for a in agg.values():
        presence_ratio = (a.frames_present / frames) if frames else 0.0
        in_assigned_ratio = (
            a.frames_in_assigned_desk / a.frames_present if a.frames_present else 0.0
        )
        distracted_ratio = (
            a.distracted_frames / a.frames_present if a.frames_present else 0.0
        )
        # "Có mặt" = ngồi TRONG region được gán. Frame xuất hiện ở bàn khác
        # không tính → ngồi sai chỗ cả buổi vẫn là ABSENT.
        in_seat_presence_ratio = (
            a.frames_in_assigned_desk / frames if frames else 0.0
        )

        print(f"[finalize] {a.code}: in_seat={a.frames_in_assigned_desk}/{frames} "
              f"(present_anywhere={a.frames_present}) "
              f"in_seat_ratio={in_seat_presence_ratio:.3f} threshold={presence_ratio_th}",
              flush=True)
        if in_seat_presence_ratio < presence_ratio_th:
            status = "ABSENT"
        else:
            status = "PRESENT"

        region = region_meta.get(a.assigned_desk_num, {})
        students_out.append({
            "studentId": a.student_id,
            "studentCode": a.code,
            "fullName": a.name,
            "assignedDeskId": region.get("id"),
            "assignedDeskLabel": region.get("label"),
            "framesPresent": a.frames_present,
            "framesInAssignedDesk": a.frames_in_assigned_desk,
            "presenceRatio": round(presence_ratio, 4),
            "inAssignedDeskRatio": round(in_assigned_ratio, 4),
            "inSeatPresenceRatio": round(in_seat_presence_ratio, 4),
            "attendanceStatus": status,
            "avgFaceMatchScore": (
                round(a.sum_face_score / a.n_face_score, 4)
                if a.n_face_score else None
            ),
            "avgGazeYawDeg": (
                round(a.sum_yaw / a.n_gaze, 2) if a.n_gaze else None
            ),
            "avgGazePitchDeg": (
                round(a.sum_pitch / a.n_gaze, 2) if a.n_gaze else None
            ),
            "distractedFrames": a.distracted_frames,
            "distractedRatio": round(distracted_ratio, 4),
            "distractionAlert": distracted_ratio >= distracted_ratio_alert,
            # attention=false CHỈ khi từng distraction >= DISTRACT_ALERT_FRAMES
            # liên tục trong clip; distraction lẻ tẻ vẫn là attention=true.
            "attention": a.student_id not in sustained_alerted,
            "sustainedDistraction": a.student_id in sustained_alerted,
        })

    clip_duration_sec = round(frames / fps_decl, 3) if fps_decl else None

    return {
        "clipPath": video_path,
        "annotatedVideoPath": annotated_video,
        "cameraId": cam_id,
        "classId": cls_id,
        "clipDurationSec": clip_duration_sec,
        "framesProcessed": frames,
        "fps": round(fps_decl, 3),
        "processingMs": processing_ms,
        "students": students_out,
        "unmatchedFaces": unmatched_faces,
        # Ảnh các lần mất tập trung — folder theo tên video + danh sách frame.
        "distractionFolderUrl": distraction_folder_url,
        "distractionSnapshots": distraction_snapshots or [],
        "modelVersion": {
            "yolo": "yolo11n.pt",
            "retinaface": "detectFace_model_op16.onnx",
            "arcface": "arcface_r100.onnx",
            "gaze": "resnet50_gaze.onnx",
        },
        "generatedAt": _now_iso(),
    }
