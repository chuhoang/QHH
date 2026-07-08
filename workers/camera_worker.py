"""Worker threads for low-latency camera streaming and AI detection.

AI pipeline: YOLOv11 person detection + RetinaFace face detection + ArcFace R100 recognition.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal, QMutex

from db import redis_client as db
from workers.shm_reader import MiddlewareSHMReader
from workers.face_models import (
    RetinaFaceDetector,
    ArcFaceExtractor,
    build_face_gallery,
)
from workers.gaze_estimator import GazeEstimator, DistractionTracker

AI_DEBUG = os.getenv("AI_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
AI_DRAW_ALL_DETECTIONS = os.getenv("AI_DRAW_ALL_DETECTIONS", "1").strip().lower() in {
    "1", "true", "yes", "on"
}
FPS_LOG = os.getenv("FPS_LOG", "1").strip().lower() in {"1", "true", "yes", "on"}
try:
    FPS_LOG_PERIOD = max(0.5, float(os.getenv("FPS_LOG_PERIOD", "1.0")))
except ValueError:
    FPS_LOG_PERIOD = 1.0


class _FpsMeter:
    """Track rolling call/iteration rate over a fixed window.

    Each tracked stage calls :meth:`tick` once per inference; ``elapsed_ms``
    is recorded if you also want average latency. The aggregator decides when
    to flush a summary line — see :class:`_FpsAggregator`.
    """

    __slots__ = ("count", "total_ms", "window_started", "last_fps", "last_ms")

    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.window_started = time.monotonic()
        # Last flushed values — kept so overlays can read them without
        # waiting for the next window to roll over.
        self.last_fps = 0.0
        self.last_ms = 0.0

    def tick(self, elapsed_ms: float = 0.0) -> None:
        self.count += 1
        self.total_ms += float(elapsed_ms)

    def snapshot_and_reset(self, now: float) -> tuple[float, float]:
        """Return (fps, mean_latency_ms) for the window since the last reset."""
        dt = max(1e-6, now - self.window_started)
        fps = self.count / dt
        mean_ms = self.total_ms / self.count if self.count else 0.0
        self.count = 0
        self.total_ms = 0.0
        self.window_started = now
        self.last_fps = fps
        self.last_ms = mean_ms
        return fps, mean_ms

    def current(self, now: float) -> tuple[float, float]:
        """Live fps over the IN-PROGRESS window; falls back to last flush."""
        dt = now - self.window_started
        if self.count >= 2 and dt > 0.0:
            fps = self.count / dt
            mean_ms = self.total_ms / self.count
            return fps, mean_ms
        return self.last_fps, self.last_ms


class _FpsAggregator:
    """Group a handful of named meters and flush a one-line summary on cadence."""

    def __init__(self, period_sec: float, name: str):
        self.period = float(period_sec)
        self.name = name
        self._meters: dict[str, _FpsMeter] = {}
        self._last_flush = time.monotonic()

    def get(self, key: str) -> _FpsMeter:
        m = self._meters.get(key)
        if m is None:
            m = _FpsMeter()
            self._meters[key] = m
        return m

    def maybe_flush(self) -> None:
        if not FPS_LOG:
            return
        now = time.monotonic()
        if now - self._last_flush < self.period:
            return
        parts: list[str] = []
        for key, meter in self._meters.items():
            fps, ms = meter.snapshot_and_reset(now)
            parts.append(f"{key}={fps:5.1f}fps/{ms:5.1f}ms")
        self._last_flush = now
        if parts:
            print(f"[fps] {self.name}: " + "  ".join(parts), flush=True)


def _time_call(meter: _FpsMeter, fn, *args, **kwargs):
    """Run ``fn``, record latency on ``meter``, return its result."""
    start = time.monotonic()
    result = fn(*args, **kwargs)
    meter.tick((time.monotonic() - start) * 1000.0)
    return result
CENTERPOINT_MIN_OVERLAP = max(
    0.0, min(1.0, float(os.getenv("CENTERPOINT_MIN_OVERLAP", "0.35")))
)
try:
    AI_LOOP_DELAY = max(0.0, float(os.getenv("AI_LOOP_DELAY", "0.05")))
except ValueError:
    AI_LOOP_DELAY = 0.05


def _debug(message: str):
    if AI_DEBUG:
        print(message, flush=True)


class CameraWorker(QThread):
    """Continuously reads Middleware2026 SHM and keeps only its newest frame.

    The UI polls :meth:`latest_frame` at its own display rate. Older frames are
    overwritten instead of entering Qt's queued-signal backlog, which is
    especially important over SSH X11 forwarding.
    """

    error = Signal(str)
    status = Signal(str)

    def __init__(self, camera_id: str, target_fps: float = 20.0, stale_grabs: int = 0, parent=None):
        super().__init__(parent)
        self.camera_id = str(camera_id)
        # Kept for API compatibility; display pacing is controlled by the UI.
        self.target_fps = max(1.0, float(target_fps))
        self._running = False
        self._mutex = QMutex()
        self._latest: np.ndarray | None = None
        self._sequence = 0

    def latest_frame(self, after_sequence: int = -1) -> tuple[int, np.ndarray | None]:
        """Return the newest frame when it is newer than ``after_sequence``."""
        self._mutex.lock()
        try:
            if self._latest is None or self._sequence == after_sequence:
                return self._sequence, None
            # The capture thread replaces the ndarray reference; it never
            # mutates an already-published frame, so a costly copy is needless.
            return self._sequence, self._latest
        finally:
            self._mutex.unlock()

    def _publish(self, frame: np.ndarray):
        self._mutex.lock()
        try:
            self._latest = frame
            self._sequence += 1
        finally:
            self._mutex.unlock()

    def run(self):
        self._running = True
        reader = None
        shm_key = None
        last_mapping_check = 0.0
        waiting_reported = False
        fps_stream = _FpsAggregator(FPS_LOG_PERIOD, f"stream cam={self.camera_id}")
        stream_meter = fps_stream.get("read")

        try:
            while self._running:
                now = time.monotonic()
                if reader is None or now - last_mapping_check >= 2.0:
                    last_mapping_check = now
                    try:
                        stream = db.get_middleware_stream(self.camera_id)
                    except Exception:
                        stream = {}
                    new_key = stream.get("key")
                    try:
                        new_key = int(new_key)
                    except (TypeError, ValueError):
                        new_key = None
                    if new_key != shm_key:
                        if reader is not None:
                            reader.close()
                        shm_key = new_key
                        reader = MiddlewareSHMReader(shm_key) if shm_key is not None else None
                    if reader is None:
                        if not waiting_reported:
                            self.status.emit("Đang chờ Middleware công bố SHM...")
                            waiting_reported = True
                        time.sleep(0.1)
                        continue

                _t0 = time.monotonic()
                item = reader.read()
                if item is None:
                    if not waiting_reported:
                        self.status.emit("Đang chờ frame từ Middleware...")
                        waiting_reported = True
                    time.sleep(0.01)
                    continue
                stream_meter.tick((time.monotonic() - _t0) * 1000.0)
                fps_stream.maybe_flush()

                _, frame = item
                waiting_reported = False
                self._publish(frame)
        except Exception as e:
            self.error.emit(str(e))
        finally:
            if reader is not None:
                reader.close()

    def stop(self):
        self._running = False
        self.wait(3000)
        self._mutex.lock()
        try:
            self._latest = None
        finally:
            self._mutex.unlock()
        self.status.emit("Stream đã dừng")


class ViewTransformer:
    """Perspective transform: image coords ↔ real-world floor-plane coords."""

    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32),
        )
        self.m_inv = cv2.getPerspectiveTransform(
            target.astype(np.float32), source.astype(np.float32),
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Image coords → real-world coords (cm)."""
        if points.size == 0:
            return points
        r = points.reshape(-1, 1, 2).astype(np.float32)
        t = cv2.perspectiveTransform(r, self.m)
        return t.reshape(-1, 2)

    def inverse_transform_points(self, points: np.ndarray) -> np.ndarray:
        """Real-world coords (cm) → image coords."""
        if points.size == 0:
            return points
        r = points.reshape(-1, 1, 2).astype(np.float32)
        t = cv2.perspectiveTransform(r, self.m_inv)
        return t.reshape(-1, 2)


class AIDetectionWorker(QThread):
    """Runs person detection and recognition grouped by desk region.

    Pipeline: YOLOv11 person detection → perspective-transform foot position
    to real floor-plane coords → in-zone-check → RetinaFace face detection
    → ArcFace R100 embedding extraction → similarity matching.
    """

    # Results and the exact source frame form one synchronized snapshot.
    detection_ready = Signal(list, np.ndarray)

    # ArcFace similarity threshold for a match (0..1, higher = stricter)
    FACE_MATCH_THRESHOLD = 0.55

    # Real desk dimensions (centimetres) — width left-right, depth front-back
    DESK_CM_W = 40.0
    DESK_CM_D = 80.0

    # 4 corners of the desk in real-world floor-plane (cm): TL, TR, BR, BL
    TARGET_CORNERS = np.array(
        [[0, 0], [DESK_CM_W, 0], [DESK_CM_W, DESK_CM_D], [0, DESK_CM_D]],
        dtype=np.float32,
    )
    _MODEL_LOCK = threading.Lock()
    _INFERENCE_LOCK = threading.Lock()
    _MODEL_BUNDLE = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._frame: np.ndarray | None = None
        self._seats: list[dict] = []
        self._students: dict[str, dict] = {}
        self._face_gallery: list[dict] = []
        self._gallery_embeddings = np.empty(
            (0, ArcFaceExtractor.EMBEDDING_DIM), dtype=np.float32
        )
        self._seat_lookup: dict[str, tuple[int, int]] = {}
        self._mutex = QMutex()

        # Ground-plane calibration state (rebuilt when seats change)
        self._ground_transform: ViewTransformer | None = None
        self._zone_ground_polys: dict = {}
        self._calibration_dirty = True

        # Detection mode: "perspective" (foot→ground polygon) or "centerpoint" (bbox center→image polygon)
        self._detection_mode = "perspective"

        # Loading ArcFace can take tens of seconds on this CPU. Keep one model
        # bundle alive across AI on/off cycles instead of rebuilding sessions.
        self._yolo, self._retinaface, self._arcface, self._gaze = self._shared_models()
        # Resolve a stable device hint once so every YOLO call uses the same
        # backend even if torch.cuda changes state later.
        try:
            import torch  # type: ignore
            self._yolo_device = 0 if torch.cuda.is_available() else "cpu"
        except Exception:
            self._yolo_device = "cpu"
        # Per-worker hysteresis state — keyed by student_id when known, else by desk-slot.
        # FPS instrumentation for YOLO / RetinaFace / ArcFace / Gaze / pipeline.
        self._fps = _FpsAggregator(FPS_LOG_PERIOD, "ai")
        self._fps_yolo = self._fps.get("yolo")
        self._fps_retina = self._fps.get("retina")
        self._fps_arc = self._fps.get("arc")
        self._fps_gaze = self._fps.get("gaze")
        self._fps_pipeline = self._fps.get("pipe")
        self._distraction = DistractionTracker(
            yaw_threshold_deg=float(os.getenv("GAZE_YAW_THRESHOLD", "60")),
            # Pitch (look up/down) intentionally disabled — a student looking
            # down at the book or up at the board is NOT "distracted".
            pitch_threshold_deg=float(os.getenv("GAZE_PITCH_THRESHOLD", "9999")),
            # Require >2 s of sustained off-axis gaze before the alert latches
            # so brief glances at neighbours do not trigger a warning.
            alert_after_sec=float(os.getenv("GAZE_ALERT_AFTER", "2.5")),
            clear_after_sec=float(os.getenv("GAZE_CLEAR_AFTER", "1.0")),
            ema_alpha=float(os.getenv("GAZE_EMA_ALPHA", "0.35")),
        )

    @classmethod
    def _shared_models(cls):
        with cls._MODEL_LOCK:
            if cls._MODEL_BUNDLE is None:
                # Import lazily so opening monitoring does not load PyTorch.
                from ultralytics import YOLO
                yolo_model = (
                    Path(__file__).resolve().parents[1] / "yolo11n.pt"
                )
                try:
                    gaze = GazeEstimator()
                except (FileNotFoundError, Exception) as e:
                    _debug(f"[GAZE] disabled: {e}")
                    gaze = None
                yolo = YOLO(str(yolo_model), verbose=False)
                # Move YOLO to CUDA when available — Ultralytics defaults to
                # CPU otherwise and a YOLOv11n forward pass costs ~6 ms CPU vs
                # ~2 ms on T4. Tolerate missing torch/cuda gracefully.
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        yolo.to("cuda:0")
                        _debug(f"[YOLO] moved to cuda:0 ({torch.cuda.get_device_name(0)})")
                except Exception as e:
                    _debug(f"[YOLO] CUDA unavailable, staying on CPU: {e}")
                cls._MODEL_BUNDLE = (
                    yolo,
                    RetinaFaceDetector(),
                    ArcFaceExtractor(),
                    gaze,
                )
            return cls._MODEL_BUNDLE

    def update_frame(self, frame: np.ndarray):
        self._mutex.lock()
        self._frame = frame.copy()
        self._mutex.unlock()

    def update_seats(self, seats: list[dict]):
        self.update_context(seats, list(self._students.values()))

    def set_detection_mode(self, mode: str):
        """Switch detection mode: 'perspective' or 'centerpoint'."""
        self._mutex.lock()
        self._detection_mode = mode
        self._mutex.unlock()

    def update_context(self, seats: list[dict], students: list[dict]):
        """Refresh seat map and known student face templates."""
        students_by_id = {str(s.get("id", "")): dict(s) for s in students if s.get("id")}
        seat_lookup: dict[str, tuple[int, int]] = {}
        for seat in seats:
            desk_num = int(seat.get("desk_num", 0) or 0)
            for slot in seat.get("slots", []) or []:
                sid = str(slot.get("student_id", "") or "")
                if sid:
                    seat_lookup[sid] = (desk_num, int(slot.get("slot_num", 1) or 1))

        gallery = self._build_face_gallery(list(students_by_id.values()))

        self._mutex.lock()
        self._seats = seats
        self._students = students_by_id
        self._seat_lookup = seat_lookup
        self._face_gallery = gallery
        self._gallery_embeddings = (
            np.stack([item["embedding"] for item in gallery]).astype(
                np.float32, copy=False
            )
            if gallery else np.empty(
                (0, ArcFaceExtractor.EMBEDDING_DIM), dtype=np.float32
            )
        )
        self._calibration_dirty = True
        self._mutex.unlock()

    def run(self):
        self._running = True
        while self._running:
            self._mutex.lock()
            # update_frame already publishes a private copy. Transfer that
            # reference to the inference thread instead of copying a full
            # camera frame for a second time.
            frame = self._frame
            self._frame = None  # latest-frame-wins; never process backlog
            seats = list(self._seats)
            students = dict(self._students)
            seat_lookup = dict(self._seat_lookup)
            gallery = list(self._face_gallery)
            cal_dirty = self._calibration_dirty
            self._calibration_dirty = False
            detection_mode = self._detection_mode
            self._mutex.unlock()

            if frame is None:
                time.sleep(0.05)
                continue

            _, results = self._detect(
                frame, seats, students, seat_lookup, gallery, cal_dirty,
                detection_mode,
            )
            self.detection_ready.emit(results, frame)
            time.sleep(AI_LOOP_DELAY)

    def _ensure_ground_transform(self, seats: list[dict], frame_w: int, frame_h: int, dirty: bool):
        """(Re)build the global image→ground homography from the first oriented zone found."""
        if not dirty and self._ground_transform is not None:
            return
        self._ground_transform = None
        self._zone_ground_polys.clear()
        for seat in seats:
            zone = seat.get("zone") if isinstance(seat.get("zone"), dict) else {}
            if zone and zone.get("type") == "oriented":
                corners, _ = self._zone_corners_and_aabb(zone, frame_w, frame_h)
                target = np.array(
                    [[0, 0], [self.DESK_CM_W, 0], [self.DESK_CM_W, self.DESK_CM_D], [0, self.DESK_CM_D]],
                    dtype=np.float32,
                )
                self._ground_transform = ViewTransformer(corners, target)
                _debug(
                    f"[GROUND] homography built from desk region "
                    f"corners=[({corners[0,0]:.0f},{corners[0,1]:.0f}) "
                    f"({corners[1,0]:.0f},{corners[1,1]:.0f}) "
                    f"({corners[2,0]:.0f},{corners[2,1]:.0f}) "
                    f"({corners[3,0]:.0f},{corners[3,1]:.0f})]"
                )
                return

    def _detect(self, frame: np.ndarray, seats: list[dict], students: dict[str, dict],
                seat_lookup: dict[str, tuple[int, int]], gallery: list[dict], cal_dirty: bool = False,
                detection_mode: str = "perspective"):
        h, w = frame.shape[:2]
        annotated = frame.copy()
        results: list[dict] = []

        # Ensure one global image→ground homography (rebuilt only when config changes)
        self._ensure_ground_transform(seats, w, h, cal_dirty)

        # 1. YOLOv11 person detection (once on full frame)
        _pipe_t0 = time.monotonic()
        _yolo_t0 = time.monotonic()
        with self._INFERENCE_LOCK:
            # device=0 pins inference to cuda:0 — without it Ultralytics can
            # silently fall back to CPU on certain stream/threading patterns.
            yolo_results = self._yolo(
                frame, classes=[0], verbose=False, device=self._yolo_device,
                conf=float(os.getenv("YOLO_PERSON_CONF", "0.15")),
            )
        self._fps_yolo.tick((time.monotonic() - _yolo_t0) * 1000.0)
        person_boxes: list[tuple[int, int, int, int]] = []
        if yolo_results and len(yolo_results) > 0:
            for box in yolo_results[0].boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                person_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        # 2. Pre-compute each person's foot point (image space + ground cm)
        person_records: list[dict] = []
        for bbox in person_boxes:
            fx_img, fy_img = self._foot_point(bbox, detection_mode)
            if self._ground_transform is not None:
                real = self._ground_transform.transform_points(
                    np.array([[fx_img, fy_img]], dtype=np.float32)
                )
                rx, ry = float(real[0, 0]), float(real[0, 1])
            else:
                rx, ry = float("nan"), float("nan")
            person_records.append({
                "bbox": bbox,
                "foot_img": (fx_img, fy_img),
                "foot_cm": (rx, ry),
                "consumed": False,
            })

        # 3. Build every desk polygon first. A person may fall inside multiple
        # overlapping regions, so assign it to the region where its point lies
        # deepest instead of whichever desk happens to be iterated first.
        zone_contexts: dict[int, dict] = {}
        for seat in seats:
            desk_num = int(seat.get("desk_num", 0) or 0)
            zone = seat.get("zone") if isinstance(seat.get("zone"), dict) else None
            slots = seat.get("slots", []) or []
            if not zone or not slots:
                continue
            corners_img, _ = self._zone_corners_and_aabb(zone, w, h)
            # Use the exact saved desk boundary. Expanding it here caused
            # people just outside a desk to be claimed by that desk.
            poly_img = corners_img.astype(np.float32)
            zone_key = ("parent", desk_num)
            if zone_key not in self._zone_ground_polys and self._ground_transform is not None:
                ground = self._ground_transform.transform_points(corners_img)
                self._zone_ground_polys[zone_key] = ground.astype(np.float32)
            zone_contexts[desk_num] = {
                "corners_img": corners_img,
                "poly_img": poly_img,
                "poly_ground": self._zone_ground_polys.get(zone_key),
            }

        for rec in person_records:
            candidates: list[
                tuple[float, float, float, float, float, int, str]
            ] = []
            for desk_num, context in zone_contexts.items():
                poly_ground = context["poly_ground"]
                use_ground = (
                    detection_mode == "perspective"
                    and poly_ground is not None
                    and rec["foot_cm"][0] == rec["foot_cm"][0]
                )
                point = rec["foot_cm"] if use_ground else rec["foot_img"]
                poly = poly_ground if use_ground else context["poly_img"]
                score = self._polygon_membership_score(poly, point)
                overlap = 1.0
                match_iou = 0.0
                if detection_mode == "centerpoint":
                    overlap = self._bbox_polygon_overlap_ratio(
                        rec["bbox"], context["corners_img"]
                    )
                    match_iou = self._bbox_polygon_iou(
                        rec["bbox"], context["corners_img"]
                    )
                if score >= 0.0 and overlap >= (
                    CENTERPOINT_MIN_OVERLAP
                    if detection_mode == "centerpoint"
                    else 0.0
                ):
                    # Overlap is only an admission threshold. Do not add it to
                    # the ownership score: a large desk region that surrounds
                    # a smaller neighbouring desk would otherwise always win
                    # because it covers more of the person's bbox. The signed
                    # depth is normalized by polygon area, so it favors the
                    # most specific region around the representative point.
                    polygon_area = max(
                        float(abs(cv2.contourArea(
                            np.asarray(poly, dtype=np.float32)
                        ))),
                        1.0,
                    )
                    # Region height is only a late tie-breaker. Making it the
                    # primary rule sends a person who is clearly concentrated
                    # in an upper desk to a lower overlapping desk.
                    region_top_y = float(
                        np.asarray(context["corners_img"])[:, 1].min()
                    )
                    candidates.append((
                        match_iou,
                        score,
                        overlap,
                        -polygon_area,
                        region_top_y,
                        desk_num,
                        (
                            "perspective_cm"
                            if use_ground else (
                                f"image_px overlap={overlap:.2f} "
                                f"iou={match_iou:.3f}"
                            )
                        ),
                    ))
            if candidates:
                _, score, _, _, _, owner_desk, owner_space = max(
                    candidates,
                    # Prefer the region whose area is most specifically matched
                    # by this bbox (IoU), then point depth and bbox coverage.
                    # Smaller regions and lower image position only break ties.
                    key=lambda item: (
                        item[0], item[1], item[2], item[3], item[4], -item[5]
                    ),
                )
                rec["owner_desk"] = owner_desk
                rec["owner_score"] = score
                rec["owner_space"] = owner_space
            else:
                rec["owner_desk"] = None
                rec["owner_score"] = None
                rec["owner_space"] = ""

        # 4. Run face detection and recognition for all owned persons through
        # one adaptive pipeline. GPU uses batches; CPU uses faster single runs.
        owned_records = [
            rec for rec in person_records if rec.get("owner_desk") is not None
        ]
        face_infos = self._recognize_faces_in_person_crops(
            frame, [rec["bbox"] for rec in owned_records], gallery
        )
        for rec, info in zip(owned_records, face_infos):
            rec["face_info"] = info
            rec["consumed"] = True

        # 5. Process each desk region (one parent polygon + many assigned slots).
        _debug(f"[DETECT] n_seats={len(seats)}  n_persons={len(person_records)}  mode={detection_mode}")
        for seat in seats:
            desk_num = int(seat.get("desk_num", 0) or 0)
            zone = seat.get("zone") if isinstance(seat.get("zone"), dict) else None
            slots = seat.get("slots", []) or []
            _debug(
                f"  seat desk={desk_num}  zone={'yes' if zone else 'NO'}  "
                f"n_slots={len(slots)}  slots_with_anchor="
                f"{sum(1 for s in slots if isinstance(s.get('anchor'), dict))}"
            )
            if not zone or not slots:
                _debug("    → skipped (no zone or no slots)")
                continue

            context = zone_contexts[desk_num]
            parent_corners_img = context["corners_img"]
            poly_img_member = context["poly_img"]
            poly_ground_member = context["poly_ground"]

            # Membership was resolved globally so overlapping regions cannot
            # claim the same person based on Redis/set iteration order.
            _debug(
                f"[ZONE B{desk_num}] mode={detection_mode}  frame={w}x{h}  "
                f"poly_img_corners={parent_corners_img.tolist()}  "
                f"poly_img_inflated={poly_img_member.tolist()}  "
                f"ground_poly={'yes' if poly_ground_member is not None else 'no'}  "
                f"n_persons={len(person_records)}"
            )
            in_zone: list[dict] = []
            for i, rec in enumerate(person_records):
                inside = rec.get("owner_desk") == desk_num
                _debug(
                    f"  [P{i}] bbox={rec['bbox']} owner={rec.get('owner_desk')} "
                    f"space={rec.get('owner_space')} score={rec.get('owner_score')} "
                    f"inside={inside}"
                )
                if inside:
                    in_zone.append(rec)
            _debug(f"  → {len(in_zone)} person(s) in zone B{desk_num}")

            # Face results were computed globally in batches above.
            _debug(
                f"  RECOG B{desk_num}: gallery_size={len(gallery)}  "
                f"threshold={self.FACE_MATCH_THRESHOLD}"
            )
            for i, rec in enumerate(in_zone):
                info = rec.get("face_info", {"face_found": False})
                rid = str(info.get("student_id", "") or "")
                rec_student = students.get(rid, {}) if rid else {}
                _debug(
                    f"    [P{i}] face_found={info.get('face_found')}  "
                    f"student_id={rid!r}  name={rec_student.get('name','')!r}  "
                    f"score={info.get('score')}  face_box={info.get('face_box')}"
                )

            # 3d. Build identity index: recognised_student_id → person record(s)
            id_to_persons: dict[str, list[dict]] = {}
            for rec in in_zone:
                rid = str(rec.get("face_info", {}).get("student_id", "") or "")
                if rid:
                    id_to_persons.setdefault(rid, []).append(rec)
            persons_matched_to_slot: set[int] = set()  # python id() set, not student_id
            assigned_ids = [str(s.get("student_id", "") or "") for s in slots]
            _debug(
                f"  SLOTS B{desk_num}: assigned_ids={assigned_ids}  "
                f"recognised_in_zone={list(id_to_persons.keys())}"
            )

            # 3e. Emit one result per slot — match by identity, not by position
            for slot in slots:
                slot_num = int(slot.get("slot_num", 1) or 1)
                assigned_id = str(slot.get("student_id", "") or "")
                assigned_student = students.get(assigned_id, {}) if assigned_id else {}

                # Find the recognised person whose student_id == this slot's assigned_id.
                matched = None
                if assigned_id and assigned_id in id_to_persons:
                    for cand in id_to_persons[assigned_id]:
                        if id(cand) not in persons_matched_to_slot:
                            matched = cand
                            persons_matched_to_slot.add(id(cand))
                            break

                if matched is None:
                    # No bbox carries this student's identity in the zone.
                    # The slot is empty (or the student is absent / unrecognised).
                    info: dict = {}
                    has_face = False
                    recognized_id = ""
                    face_box = None
                    person_bbox = None
                    person_real_pt = None
                    rec_score = None
                    present = False
                else:
                    info = matched.get("face_info", {}) or {}
                    has_face = bool(info.get("face_found"))
                    recognized_id = assigned_id   # by construction
                    face_box = info.get("face_box")
                    person_bbox = matched["bbox"]
                    person_real_pt = matched["foot_cm"] if matched["foot_cm"][0] == matched["foot_cm"][0] else None
                    rec_score = info.get("score")
                    present = True

                recognized_student = students.get(recognized_id, {}) if recognized_id else {}
                match_status, match_text, expected_desk, expected_slot = self._compare_position(
                    recognized_id, desk_num, slot_num, assigned_id, seat_lookup,
                    present, has_face, bool(gallery)
                )

                results.append({
                    "desk_num": desk_num,
                    "slot_num": slot_num,
                    "assigned_student_id": assigned_id,
                    "assigned_name": assigned_student.get("name", ""),
                    "assigned_code": assigned_student.get("student_code", ""),
                    "student_id": assigned_id,
                    "present": present,
                    "face_found": has_face,
                    "recognized_student_id": recognized_id,
                    "recognized_name": recognized_student.get("name", ""),
                    "recognized_code": recognized_student.get("student_code", ""),
                    "recognition_score": rec_score,
                    "best_candidate_id": info.get("best_candidate_id", ""),
                    "face_pose_ok": info.get("face_pose_ok"),
                    "face_pose": info.get("face_pose", "unknown"),
                    "face_yaw_score": info.get("face_yaw_score"),
                    "face_pitch_score": info.get("face_pitch_score"),
                    "face_roll_deg": info.get("face_roll_deg"),
                    "gaze_yaw_deg": info.get("gaze_yaw_deg"),
                    "gaze_pitch_deg": info.get("gaze_pitch_deg"),
                    "gaze_focused": info.get("gaze_focused"),
                    "gaze_alert": bool(info.get("gaze_alert")),
                    "match_status": match_status,
                    "match_text": match_text,
                    "expected_desk": expected_desk,
                    "expected_slot": expected_slot,
                    "face_box": face_box,
                    "person_bbox": person_bbox,
                    "person_real_cm": person_real_pt,
                })

            # 3f. Persons in zone NOT matched to any slot of this zone → "unassigned"
            # Either: face unknown (no student_id), OR recognised student not assigned here,
            # OR same student appeared more than once (duplicates).
            for rec in in_zone:
                if id(rec) in persons_matched_to_slot:
                    continue
                info = rec.get("face_info", {}) or {}
                rid = str(info.get("student_id", "") or "")
                recognized_student = students.get(rid, {}) if rid else {}
                # Decide a useful match_status for the extra
                if not info.get("face_found"):
                    status, text = "no_face", "Có người trong vùng, chưa thấy mặt"
                elif not rid:
                    status, text = "unknown", "Có mặt nhưng chưa nhận diện được"
                else:
                    # Recognised but not assigned to THIS desk
                    expected = seat_lookup.get(rid)
                    if expected and int(expected[0]) != desk_num:
                        status = "wrong"
                        text = f"Sai bàn — đúng là B{expected[0]}.{expected[1]}"
                    else:
                        status, text = "unassigned", "Nhận diện được nhưng không thuộc bàn này"
                results.append({
                    "desk_num": desk_num,
                    "slot_num": -1,
                    "assigned_student_id": "",
                    "assigned_name": "",
                    "assigned_code": "",
                    "student_id": "",
                    "present": True,
                    "face_found": bool(info.get("face_found")),
                    "recognized_student_id": rid,
                    "recognized_name": recognized_student.get("name", ""),
                    "recognized_code": recognized_student.get("student_code", ""),
                    "recognition_score": info.get("score"),
                    "best_candidate_id": info.get("best_candidate_id", ""),
                    "face_pose_ok": info.get("face_pose_ok"),
                    "face_pose": info.get("face_pose", "unknown"),
                    "face_yaw_score": info.get("face_yaw_score"),
                    "face_pitch_score": info.get("face_pitch_score"),
                    "face_roll_deg": info.get("face_roll_deg"),
                    "gaze_yaw_deg": info.get("gaze_yaw_deg"),
                    "gaze_pitch_deg": info.get("gaze_pitch_deg"),
                    "gaze_focused": info.get("gaze_focused"),
                    "gaze_alert": bool(info.get("gaze_alert")),
                    "match_status": status,
                    "match_text": text,
                    "expected_desk": None,
                    "expected_slot": None,
                    "face_box": info.get("face_box"),
                    "person_bbox": rec["bbox"],
                    "person_real_cm": rec["foot_cm"] if rec["foot_cm"][0] == rec["foot_cm"][0] else None,
                })

            # 3g. Visualisation — parent polygon once, then per-person overlays
            parent_color = (96, 165, 250)  # blue-400 for parent zone outline
            cv2.polylines(annotated, [parent_corners_img.astype(np.int32)],
                          True, parent_color, 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Zone {desk_num}",
                        (int(parent_corners_img[:, 0].min()) + 4,
                         max(int(parent_corners_img[:, 1].min()) - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, parent_color, 2)

            # Draw all detections in result-video test mode. Production/teacher
            # view can set AI_DRAW_ALL_DETECTIONS=0 to keep only confirmed seats.
            desk_results = [
                item for item in results
                if int(item.get("desk_num", 0) or 0) == desk_num
            ]
            for r in desk_results:
                status = str(r.get("match_status") or "")
                if not AI_DRAW_ALL_DETECTIONS and status != "correct":
                    continue
                col = self._status_color(status, r["present"])
                pb = r.get("person_bbox")
                if pb is not None:
                    px, py, pw, ph = pb
                    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), col, 2)
                    tag = (
                        f"B{desk_num}.{r['slot_num']}"
                        if r["slot_num"] != -1 else f"B{desk_num}.?"
                    ) + f" {status}"
                    cv2.putText(
                        annotated, tag, (px + 4, max(py - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2,
                    )
                    if r.get("recognized_name"):
                        cv2.putText(
                            annotated, r["recognized_name"][:18],
                            (px + 4, py + ph + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2,
                        )
                    if r.get("face_pose_ok") is False:
                        pose_text = {
                            "turned_left": "QUAY TRAI",
                            "turned_right": "QUAY PHAI",
                            "tilted": "NGHIENG DAU",
                        }.get(r.get("face_pose"), "KHONG NHIN THANG")
                        cv2.putText(
                            annotated, pose_text, (px + 4, py + ph + 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 165, 255), 2,
                        )
                    if r.get("gaze_alert"):
                        yaw_v = r.get("gaze_yaw_deg")
                        suffix = f" (yaw={yaw_v:.0f})" if yaw_v is not None else ""
                        cv2.putText(
                            annotated, f"KHONG TAP TRUNG{suffix}",
                            (px + 4, py + ph + 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 2,
                        )
                        cv2.rectangle(
                            annotated, (px, py), (px + pw, py + ph),
                            (0, 0, 255), 3,
                        )
                if r.get("face_box"):
                    fx, fy, fw, fh = r["face_box"]
                    cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), col, 1)
                    # Red gaze arrow — identical formula to
                    # gaze-estimation/utils/helpers.py:draw_gaze.
                    # Inputs in radians; bbox length is the face width.
                    yaw_deg = r.get("gaze_yaw_deg")
                    pitch_deg = r.get("gaze_pitch_deg")
                    if yaw_deg is not None and pitch_deg is not None:
                        yaw_rad = float(np.radians(yaw_deg))
                        pitch_rad = float(np.radians(pitch_deg))
                        cx = int(fx + fw / 2)
                        cy = int(fy + fh / 2)
                        length = int(fw)
                        dx = int(-length * np.sin(yaw_rad) * np.cos(pitch_rad))
                        dy = int(-length * np.sin(pitch_rad))
                        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.arrowedLine(
                            annotated, (cx, cy), (cx + dx, cy + dy),
                            (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.25,
                        )

            if AI_DRAW_ALL_DETECTIONS:
                for rec in person_records:
                    if rec.get("owner_desk") is not None:
                        continue
                    px, py, pw, ph = rec["bbox"]
                    col = (160, 160, 160)
                    cv2.rectangle(annotated, (px, py), (px + pw, py + ph), col, 1)
                    cv2.putText(
                        annotated, "person outside zone",
                        (px + 4, max(py - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1,
                    )

        # ── Global cm grid (drawn once from calibrated desk) ──
        if self._ground_transform is not None:
            cm_w, cm_d = int(self.DESK_CM_W), int(self.DESK_CM_D)
            for gx in range(0, cm_w + 1, 10):
                for gy in range(0, cm_d + 1, 10):
                    real_pt = np.array([[gx, gy]], dtype=np.float32)
                    img_pt = self._ground_transform.inverse_transform_points(real_pt)
                    ix, iy = int(img_pt[0, 0]), int(img_pt[0, 1])
                    if (gx, gy) == (0, 0):
                        gc = (255, 0, 0)        # blue  = origin
                    elif (gx, gy) == (cm_w, 0):
                        gc = (0, 255, 255)       # yellow
                    elif (gx, gy) == (cm_w, cm_d):
                        gc = (255, 255, 0)       # cyan
                    elif (gx, gy) == (0, cm_d):
                        gc = (0, 165, 255)       # orange
                    elif (gx, gy) == (cm_w // 2, cm_d // 2):
                        gc = (0, 255, 0)         # green = center
                    else:
                        gc = (180, 180, 180)     # gray grid points
                    if 0 <= ix < w and 0 <= iy < h:
                        cv2.circle(annotated, (ix, iy), 4, gc, -1)
                        if (gx, gy) in [(0, 0), (cm_w, 0), (cm_w, cm_d), (0, cm_d), (cm_w // 2, cm_d // 2)]:
                            cv2.putText(annotated, f"({gx},{gy})", (ix + 6, iy - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, gc, 1)

        # Global YOLO person boxes intentionally suppressed — only confirmed
        # "correct" students are drawn (handled per-zone above).

        aggregated = self._aggregate_desk_results(results, seats)
        for desk_result in aggregated:
            desk_result["source_width"] = w
            desk_result["source_height"] = h
        # Keep global YOLO boxes available to non-Qt clients. Desk slot results
        # intentionally contain only people assigned to a configured region,
        # while a web monitor also needs proof that detection is running for
        # people outside every region. Store this once to avoid duplicating the
        # same list in all desk payloads.
        if aggregated:
            aggregated[0]["all_person_boxes"] = [list(box) for box in person_boxes]
        self._fps_pipeline.tick((time.monotonic() - _pipe_t0) * 1000.0)
        self._fps.maybe_flush()

        # On-screen FPS — top-left corner. Reads the live rolling rate so the
        # number updates every ~1s, not only when the log line flushes.
        pipe_fps, pipe_ms = self._fps_pipeline.current(time.monotonic())
        fps_text = f"AI {pipe_fps:5.1f} FPS  {pipe_ms:5.1f} ms/frame"
        text_origin = (12, 32)
        (tw, th), _ = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        # Semi-transparent black box for legibility on any background.
        bg = annotated.copy()
        cv2.rectangle(
            bg,
            (text_origin[0] - 6, text_origin[1] - th - 6),
            (text_origin[0] + tw + 6, text_origin[1] + 6),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(bg, 0.55, annotated, 0.45, 0, annotated)
        cv2.putText(
            annotated, fps_text, text_origin,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (16, 255, 120), 2, cv2.LINE_AA,
        )
        return annotated, aggregated

    def _build_face_gallery(self, students: list[dict]) -> list[dict]:
        """Build face gallery — RetinaFace alignment → ArcFace embedding."""
        with self._INFERENCE_LOCK:
            return build_face_gallery(
                students, self._arcface, self._retinaface
            )

    # ── New helpers for parent-zone pipeline ────────────────────────────

    @staticmethod
    def _foot_point(bbox: tuple[int, int, int, int],
                    detection_mode: str) -> tuple[float, float]:
        """Return the bbox's representative point in image space.

        perspective : true foot               (px + w/2, py + h)
        centerpoint : bbox geometric centre   (px + w/2, py + h/2)
        """
        px, py, pw, ph = bbox
        cx = float(px + pw / 2)
        if detection_mode == "centerpoint":
            cy = float(py + ph / 2)
        else:
            cy = float(py + ph)
        return cx, cy

    @staticmethod
    def _inflate_polygon(poly: np.ndarray, ratio: float = 0.05) -> np.ndarray:
        """Inflate a polygon outward around its centroid by `ratio`."""
        if poly is None or len(poly) == 0:
            return poly
        c = poly.mean(axis=0)
        return ((poly - c) * (1.0 + ratio) + c).astype(np.float32)

    @staticmethod
    def _polygon_membership_score(
        poly: np.ndarray,
        point: tuple[float, float],
    ) -> float:
        """Return normalized signed depth inside a polygon.

        Positive means inside. Dividing the OpenCV edge distance by the
        polygon's characteristic size makes overlapping desks comparable even
        when one region is much larger than another.
        """
        contour = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
        signed_distance = float(
            cv2.pointPolygonTest(
                contour, (float(point[0]), float(point[1])), True
            )
        )
        area = max(float(abs(cv2.contourArea(contour))), 1.0)
        return signed_distance / float(np.sqrt(area))

    @staticmethod
    def _bbox_polygon_overlap_ratio(
        bbox: tuple[int, int, int, int],
        polygon: np.ndarray,
    ) -> float:
        """Fraction of a person's bbox area covered by the desk polygon."""
        x, y, w, h = bbox
        bbox_area = max(float(w * h), 1.0)
        bbox_poly = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32,
        )
        desk_poly = np.asarray(polygon, dtype=np.float32)
        try:
            intersection_area, _ = cv2.intersectConvexConvex(
                bbox_poly, desk_poly
            )
        except cv2.error:
            return 0.0
        return max(0.0, min(1.0, float(intersection_area) / bbox_area))

    @staticmethod
    def _bbox_polygon_iou(
        bbox: tuple[int, int, int, int],
        polygon: np.ndarray,
    ) -> float:
        """Intersection-over-union between a person bbox and desk polygon."""
        x, y, w, h = bbox
        bbox_area = max(float(w * h), 1.0)
        bbox_poly = np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
            dtype=np.float32,
        )
        desk_poly = np.asarray(polygon, dtype=np.float32)
        desk_area = max(float(abs(cv2.contourArea(desk_poly))), 1.0)
        try:
            intersection_area, _ = cv2.intersectConvexConvex(
                bbox_poly, desk_poly
            )
        except cv2.error:
            return 0.0
        union_area = bbox_area + desk_area - float(intersection_area)
        if union_area <= 0.0:
            return 0.0
        return max(0.0, min(1.0, float(intersection_area) / union_area))

    def _recognize_face_in_person_crop(
        self,
        frame: np.ndarray,
        person_bbox: tuple[int, int, int, int],
        gallery: list[dict],
    ) -> dict:
        """Crop person bbox → RetinaFace → ArcFace → gallery match.

        One person → at most one face (RetinaFace already keeps biggest).
        """
        px, py, pw, ph = person_bbox
        h, w = frame.shape[:2]
        x1 = max(0, int(px));      y1 = max(0, int(py))
        x2 = min(w, int(px + pw)); y2 = min(h, int(py + ph))
        if x2 <= x1 or y2 <= y1:
            return {"face_found": False}

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return {"face_found": False}

        retina_dets, _ = self._retinaface.detect(crop)
        if len(retina_dets) == 0:
            return {"face_found": False}

        det = retina_dets[0]
        fx1, fy1, fx2, fy2 = det["loc"]
        face_box = (int(x1 + fx1), int(y1 + fy1),
                    int(fx2 - fx1), int(fy2 - fy1))
        pose = self._estimate_face_orientation(det.get("landm", []))

        aligned = det.get("aligned_face")
        if aligned is None or not gallery:
            return {
                "face_found": True, "face_box": face_box, "score": None,
                "student_id": "", **pose,
            }

        embedding = self._arcface.extract_single(aligned)
        if embedding is None:
            return {
                "face_found": True, "face_box": face_box, "score": None,
                "student_id": "", **pose,
            }

        best_id = ""
        best_sim = -1.0
        for item in gallery:
            sim = ArcFaceExtractor.similarity(embedding, item["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_id = item["student_id"]

        if best_sim < self.FACE_MATCH_THRESHOLD:
            # Trả về best_candidate_id để clip_inference dùng threshold thấp hơn
            return {"face_found": True, "face_box": face_box,
                    "score": round(best_sim, 3), "student_id": "",
                    "best_candidate_id": best_id, **pose}
        return {"face_found": True, "face_box": face_box,
                "score": round(best_sim, 3), "student_id": best_id,
                "best_candidate_id": best_id, **pose}

    def _recognize_faces_in_person_crops(
        self,
        frame: np.ndarray,
        person_bboxes: list[tuple[int, int, int, int]],
        gallery: list[dict],
    ) -> list[dict]:
        """Recognize all person crops while preserving result order.

        Despite the legacy name we no longer feed person crops to RetinaFace
        one at a time — that costs ~200 ms per frame for a classroom. Instead
        we run RetinaFace **once** on the full frame and assign each face to
        the YOLO person whose bbox contains its centre. Same outputs, ~20x
        faster.
        """
        if not person_bboxes:
            return []

        frame_h, frame_w = frame.shape[:2]
        outputs: list[dict] = [{"face_found": False} for _ in person_bboxes]

        _retina_t0 = time.monotonic()
        with self._INFERENCE_LOCK:
            face_dets = self._retinaface.detect_all(frame)
        self._fps_retina.tick((time.monotonic() - _retina_t0) * 1000.0)

        # Map each detected face to the person bbox that contains its centre.
        # Falls back to highest IoU if no person contains the face centre
        # (e.g. RetinaFace caught a face that YOLO missed).
        assigned: list[tuple[int, dict] | None] = [None] * len(person_bboxes)
        for det in face_dets:
            fx1, fy1, fx2, fy2 = det["loc"]
            cx = (fx1 + fx2) / 2.0
            cy = (fy1 + fy2) / 2.0
            face_area = max(1.0, float((fx2 - fx1) * (fy2 - fy1)))
            best_idx = -1
            best_overlap = 0.0
            for i, (px, py, pw, ph) in enumerate(person_bboxes):
                if assigned[i] is not None:
                    continue
                if px <= cx <= px + pw and py <= cy <= py + ph:
                    # Tie-break by tightest containment so two stacked
                    # bboxes get the right face instead of the first one.
                    overlap = (pw * ph) / face_area
                    if best_idx < 0 or overlap < best_overlap:
                        best_idx = i
                        best_overlap = overlap
            if best_idx >= 0:
                assigned[best_idx] = (best_idx, det)

        embedding_faces: list[np.ndarray] = []
        embedding_output_indexes: list[int] = []
        gaze_crops: list[np.ndarray] = []
        gaze_output_indexes: list[int] = []

        for output_index, slot in enumerate(assigned):
            if slot is None:
                continue
            _, det = slot
            fx1, fy1, fx2, fy2 = det["loc"]
            face_box = (
                int(fx1), int(fy1),
                int(fx2 - fx1), int(fy2 - fy1),
            )
            pose = self._estimate_face_orientation(det.get("landm", []))
            outputs[output_index] = {
                "face_found": True,
                "face_box": face_box,
                "score": None,
                "student_id": "",
                **pose,
            }
            aligned = det.get("aligned_face")
            if aligned is not None and gallery:
                embedding_faces.append(aligned)
                embedding_output_indexes.append(output_index)
            # Match the reference `gaze-estimation/onnx_inference.py`: crop
            # SÁT bbox của RetinaFace (no padding). Earlier tests showed that
            # a 0.5 padding shifts predicted yaw by ~30°+ for off-centre faces
            # because the padded crop is no longer symmetric around the head
            # — the gaze model trained on tight face crops sees the eyes at
            # the wrong position. Override via GAZE_FACE_PAD if needed.
            if self._gaze is not None:
                fx, fy, fw, fh = face_box
                pad_ratio = float(os.getenv("GAZE_FACE_PAD", "0.0"))
                pad = int(pad_ratio * max(fw, fh))
                gx1 = max(0, fx - pad)
                gy1 = max(0, fy - pad)
                gx2 = min(frame_w, fx + fw + pad)
                gy2 = min(frame_h, fy + fh + pad)
                if gx2 > gx1 and gy2 > gy1:
                    gaze_crops.append(frame[gy1:gy2, gx1:gx2])
                    gaze_output_indexes.append(output_index)

        if self._gaze is not None and gaze_crops:
            _gaze_t0 = time.monotonic()
            try:
                with self._INFERENCE_LOCK:
                    gaze_results = self._gaze.estimate_batch(gaze_crops)
                self._fps_gaze.tick((time.monotonic() - _gaze_t0) * 1000.0)
            except Exception as e:
                _debug(f"[GAZE] inference failed: {e}")
                gaze_results = [(0.0, 0.0)] * len(gaze_crops)
            for output_index, (yaw_rad, pitch_rad) in zip(
                gaze_output_indexes, gaze_results
            ):
                # GazeEstimator now returns the reference units (radians).
                # Convert to degrees once at the boundary so the rest of the
                # pipeline (tracker thresholds, UI labels, logs) keeps the
                # human-readable degrees it has always used.
                yaw_deg = float(np.degrees(yaw_rad))
                pitch_deg = float(np.degrees(pitch_rad))
                outputs[output_index]["gaze_yaw_deg"] = round(yaw_deg, 1)
                outputs[output_index]["gaze_pitch_deg"] = round(pitch_deg, 1)

        _arc_t0 = time.monotonic()
        with self._INFERENCE_LOCK:
            embeddings = self._arcface.extract(embedding_faces)
        self._fps_arc.tick((time.monotonic() - _arc_t0) * 1000.0)
        if embeddings is None or not len(embeddings):
            return outputs

        gallery_embeddings = self._gallery_embeddings
        if len(gallery_embeddings) != len(gallery):
            gallery_embeddings = np.stack(
                [item["embedding"] for item in gallery]
            ).astype(np.float32, copy=False)

        distances = np.linalg.norm(
            embeddings[:, np.newaxis, :] -
            gallery_embeddings[np.newaxis, :, :],
            axis=2,
        )
        similarities = (
            np.tanh((1.23132175 - distances) * 6.602259425) + 1.0
        ) / 2.0
        best_gallery_indexes = np.argmax(similarities, axis=1)

        for row, output_index in enumerate(embedding_output_indexes):
            gallery_index = int(best_gallery_indexes[row])
            best_similarity = float(similarities[row, gallery_index])
            outputs[output_index]["score"] = round(best_similarity, 3)
            if best_similarity >= self.FACE_MATCH_THRESHOLD:
                outputs[output_index]["student_id"] = str(
                    gallery[gallery_index]["student_id"]
                )
        self._apply_distraction(outputs, person_bboxes)
        return outputs

    def _apply_distraction(
        self,
        outputs: list[dict],
        person_bboxes: list[tuple[int, int, int, int]],
    ) -> None:
        """Run hysteresis on gaze angles and stamp focus/alert flags per face."""
        if self._gaze is None:
            return
        now = time.monotonic()
        seen_keys: set[str] = set()
        for index, info in enumerate(outputs):
            if not info.get("face_found"):
                continue
            yaw = info.get("gaze_yaw_deg")
            pitch = info.get("gaze_pitch_deg")
            if yaw is None or pitch is None:
                continue
            sid = str(info.get("student_id") or "")
            if sid:
                key = f"sid:{sid}"
            else:
                # Fall back to bbox-center bucket so unknown faces still get
                # a stable short-term track without polluting the student map.
                bx, by, bw, bh = person_bboxes[index]
                key = f"bbox:{(bx + bw // 2) // 32}:{(by + bh // 2) // 32}"
            seen_keys.add(key)
            flags = self._distraction.update(key, float(yaw), float(pitch), now)
            info.update(flags)
            # Keep the RAW yaw/pitch in `gaze_yaw_deg` / `gaze_pitch_deg` so
            # the on-screen arrow points the right way — EMA-smoothing them
            # lags by 5-6 frames and the arrow then drifts (visible as the
            # arrow leaning left while the student is looking down).
            # `gaze_yaw_smooth` / `gaze_pitch_smooth` are still available via
            # `flags` if any downstream consumer wants a debounced number.
        # Reset bbox-only keys aggressively; keep student keys across frames.
        self._distraction.prune(
            {k for k in seen_keys}
            | {k for k in self._distraction._state if k.startswith("sid:")}
        )

    @staticmethod
    def _estimate_face_orientation(landmarks) -> dict:
        """Estimate side-turn and roll from RetinaFace's five landmarks.

        Gaze-estimation predicts eye direction, not head rotation. RetinaFace
        already provides the geometry needed for a cheap face-quality check.
        The score is intentionally advisory: recognition still runs so a
        briefly turned student is not marked absent.
        """
        try:
            pts = np.asarray(landmarks, dtype=np.float32).reshape(5, 2)
            left_eye, right_eye, nose, left_mouth, right_mouth = pts
            eye_vector = right_eye - left_eye
            eye_distance = float(np.linalg.norm(eye_vector))
            if eye_distance < 1.0:
                raise ValueError("invalid eye distance")
            eye_mid = (left_eye + right_eye) / 2.0
            mouth_mid = (left_mouth + right_mouth) / 2.0
            yaw_score = float((nose[0] - eye_mid[0]) / (eye_distance / 2.0))
            roll_deg = float(np.degrees(np.arctan2(eye_vector[1], eye_vector[0])))
            vertical = max(float(mouth_mid[1] - eye_mid[1]), 1.0)
            pitch_score = float((nose[1] - eye_mid[1]) / vertical)
            pose_ok = abs(yaw_score) <= 0.55 and abs(roll_deg) <= 25.0
            if abs(roll_deg) > 25.0:
                pose_label = "tilted"
            elif yaw_score < -0.55:
                pose_label = "turned_left"
            elif yaw_score > 0.55:
                pose_label = "turned_right"
            else:
                pose_label = "frontal"
            return {
                "face_pose_ok": pose_ok,
                "face_pose": pose_label,
                "face_yaw_score": round(yaw_score, 3),
                "face_pitch_score": round(pitch_score, 3),
                "face_roll_deg": round(roll_deg, 1),
            }
        except (TypeError, ValueError):
            return {
                "face_pose_ok": None,
                "face_pose": "unknown",
                "face_yaw_score": None,
                "face_pitch_score": None,
                "face_roll_deg": None,
            }

    @staticmethod
    def _aggregate_desk_results(flat_results: list[dict], seats: list[dict]) -> list[dict]:
        """Expose one result object per desk while preserving slot details."""
        by_desk: dict[int, list[dict]] = {}
        for result in flat_results:
            by_desk.setdefault(int(result.get("desk_num", 0)), []).append(result)

        aggregated = []
        for seat in seats:
            desk_num = int(seat.get("desk_num", 0) or 0)
            items = by_desk.get(desk_num, [])
            slot_results = [item for item in items if int(item.get("slot_num", -1)) >= 0]
            extra_results = [item for item in items if int(item.get("slot_num", -1)) < 0]
            present_count = sum(1 for item in items if item.get("present"))
            correct_count = sum(1 for item in slot_results if item.get("match_status") == "correct")
            wrong_count = sum(
                1 for item in items if item.get("match_status") in {"wrong", "unassigned"}
            )
            recognized_count = sum(1 for item in items if item.get("recognized_student_id"))
            aggregated.append({
                "desk_num": desk_num,
                "label": f"Bàn {desk_num}",
                "zone": seat.get("zone", {}),
                "capacity": len(seat.get("slots", [])),
                "present": present_count > 0,
                "present_count": present_count,
                "recognized_count": recognized_count,
                "correct_count": correct_count,
                "wrong_count": wrong_count,
                "slot_results": slot_results,
                "extra_results": extra_results,
                "persons": [
                    {
                        "student_id": item.get("recognized_student_id", ""),
                        "name": item.get("recognized_name", ""),
                        "bbox": item.get("person_bbox"),
                        "face_pose": item.get("face_pose", "unknown"),
                        "face_pose_ok": item.get("face_pose_ok"),
                        "gaze_yaw_deg": item.get("gaze_yaw_deg"),
                        "gaze_pitch_deg": item.get("gaze_pitch_deg"),
                        "gaze_focused": item.get("gaze_focused"),
                        "gaze_alert": bool(item.get("gaze_alert")),
                    }
                    for item in items if item.get("present")
                ],
                "distracted_count": sum(
                    1 for item in items if item.get("gaze_alert")
                ),
            })
        return aggregated

    def _recognize_face_in_zone(
        self,
        frame: np.ndarray,
        zx: int, zy: int, zw: int, zh: int,
        gallery: list[dict],
    ) -> dict:
        """Crop zone ROI → RetinaFace detect → ArcFace embedding → gallery match."""
        roi = frame[max(0, zy):max(0, zy + zh), max(0, zx):max(0, zx + zw)]
        if roi is None or roi.size == 0:
            return {"face_found": False}

        # RetinaFace detection on zone ROI (preprocess_batch → ONNX → postProcess_batch)
        retina_dets, croped_images = self._retinaface.detect(roi)

        if len(retina_dets) == 0:
            return {"face_found": False}

        # Take the first/best detection
        det = retina_dets[0]
        x1, y1, x2, y2 = det["loc"]
        # Offset local ROI coordinates back to full frame
        absolute_box = (int(zx + x1), int(zy + y1), int(x2 - x1), int(y2 - y1))

        # ArcFace embedding
        aligned_face = det.get("aligned_face")
        if aligned_face is None or not gallery:
            return {"face_found": True, "face_box": absolute_box, "score": None}

        embedding = self._arcface.extract_single(aligned_face)
        if embedding is None:
            return {"face_found": True, "face_box": absolute_box, "score": None}

        # Match against gallery
        best_student_id = ""
        best_similarity = -1.0
        for item in gallery:
            sim = ArcFaceExtractor.similarity(embedding, item["embedding"])
            if sim > best_similarity:
                best_similarity = sim
                best_student_id = item["student_id"]

        if best_similarity < self.FACE_MATCH_THRESHOLD:
            return {
                "face_found": True,
                "face_box": absolute_box,
                "score": round(best_similarity, 3),
                "student_id": "",
            }

        return {
            "face_found": True,
            "face_box": absolute_box,
            "score": round(best_similarity, 3),
            "student_id": best_student_id,
        }

    def _compare_position(self, recognized_id: str, desk_num: int, slot_num: int, assigned_id: str,
                          seat_lookup: dict[str, tuple[int, int]], present: bool, has_face: bool,
                          has_gallery: bool):
        if not present:
            return "empty", "Trống / vắng", None, None
        if not has_face:
            return "no_face", "Có người, chưa thấy mặt", None, None
        if not has_gallery:
            return "no_gallery", "Chưa có ảnh mẫu khuôn mặt", None, None
        if not recognized_id:
            return "unknown", "Có mặt nhưng chưa nhận diện được", None, None

        expected = seat_lookup.get(recognized_id)
        if expected is None:
            return "unassigned", "Nhận diện được nhưng chưa gán chỗ", None, None
        exp_desk, exp_slot = expected
        if int(exp_desk) == int(desk_num) and int(exp_slot) == int(slot_num):
            return "correct", "Đúng vị trí", exp_desk, exp_slot

        # Also flag when the current slot expects a different assigned student.
        if assigned_id and recognized_id != assigned_id:
            return "wrong", f"Sai chỗ - đúng là B{exp_desk}.{exp_slot}", exp_desk, exp_slot
        return "wrong", f"Sai vị trí - đúng là B{exp_desk}.{exp_slot}", exp_desk, exp_slot

    @staticmethod
    def _status_color(match_status: str, present: bool):
        if match_status == "correct":
            return (16, 185, 129)  # emerald
        if match_status in {"wrong", "unassigned"}:
            return (225, 29, 72)  # rose
        if present:
            return (245, 158, 11)  # amber
        return (225, 29, 72)

    @staticmethod
    def _zone_corners_and_aabb(zone: dict, frame_w: int, frame_h: int):
        """Return (4_corners_float32, axis_aligned_rect) for any zone type.

        corners shape (4, 2) in order: top-left, top-right, bottom-right, bottom-left.
        aabb = (x, y, w, h) ints — safe for ROI cropping.
        """
        import math
        zone_type = zone.get("type", "normal")
        if zone_type == "oriented":
            cx = float(zone["cx"]) * frame_w
            cy = float(zone["cy"]) * frame_h
            bw = float(zone["w"]) * frame_w
            bh = float(zone["h"]) * frame_h
            angle = float(zone.get("angle", 0))
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            hw, hh = bw / 2.0, bh / 2.0
            raw = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts = (raw @ rot.T + [cx, cy])
        else:
            zx = int(zone.get("x", 0) * frame_w)
            zy = int(zone.get("y", 0) * frame_h)
            zw = int(zone.get("w", 0.1) * frame_w)
            zh = int(zone.get("h", 0.15) * frame_h)
            pts = np.array([[zx, zy], [zx + zw, zy], [zx + zw, zy + zh], [zx, zy + zh]], dtype=np.float32)

        x_min = int(pts[:, 0].min())
        y_min = int(pts[:, 1].min())
        rw = int(pts[:, 0].max() - x_min)
        rh = int(pts[:, 1].max() - y_min)
        return pts, (x_min, y_min, rw, rh)

    @staticmethod
    def _draw_oriented_zone_cv(display, zone, frame_w, frame_h, color, thickness):
        """Draw an oriented (rotated) rectangular zone on an OpenCV image."""
        import math
        cx = int(zone["cx"] * frame_w)
        cy = int(zone["cy"] * frame_h)
        bw = int(zone["w"] * frame_w)
        bh = int(zone["h"] * frame_h)
        angle = zone.get("angle", 0)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        hw, hh = bw / 2.0, bh / 2.0
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float32)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = (corners @ rot.T + [cx, cy]).astype(np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=thickness)

    def stop(self):
        self._running = False
        self.wait(3000)
