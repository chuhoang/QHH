"""AI monitoring with one camera region per physical desk."""

import os

import numpy as np
import cv2

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSplitter, QGroupBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor, QCursor

from db import redis_client as db
from workers.camera_worker import CameraWorker, AIDetectionWorker


ROLE_DESK = Qt.ItemDataRole.UserRole
ROLE_SLOT = Qt.ItemDataRole.UserRole + 1


class ZoneCanvas(QLabel):
    """Shows camera frames and lets the user draft a zone before saving it.

    Supports two draw modes:
      - normal:  drag rect → {type: "normal", x, y, w, h}
      - oriented: click 4 corner points → {type: "oriented", cx, cy, w, h, angle}
    """

    zone_drafted = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setObjectName("cam_label")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMouseTracking(True)

        self._frame: np.ndarray | None = None
        # Exact camera frame used by the latest AI inference. Detection
        # coordinates must be painted on this frame, not on a newer live frame.
        self._detection_frame: np.ndarray | None = None
        self._seats: list[dict] = []
        self._results: list[dict] = []
        self._students: dict[str, dict] = {}
        self._class_id = ""
        self._camera_id = ""

        self._mapping_mode = False
        self._draw_mode = "normal"  # or "oriented"
        self._pending_desk: int | None = None
        self._drawing = False
        self._draw_start = None
        self._draw_rect = None
        self._draft_zone: dict | None = None

        # Oriented bbox: 4-click state
        self._oriented_points: list[tuple[int, int]] = []
        self._oriented_mouse_pos: tuple[int, int] | None = None

    def set_context(self, class_id: str, camera_id: str):
        self._class_id = class_id
        self._camera_id = camera_id
        self._detection_frame = None
        self._results = []
        self._reload_data()
        self._refresh()

    def set_mapping_mode(self, active: bool, desk_num: int | None = None):
        self._mapping_mode = active
        self._pending_desk = desk_num
        self._draw_rect = None
        self._draft_zone = None
        self._oriented_points.clear()
        self._oriented_mouse_pos = None
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor if active else Qt.CursorShape.ArrowCursor))
        self._refresh()

    def set_draw_mode(self, mode: str):
        self._draw_mode = mode
        self._draw_rect = None
        self._draft_zone = None
        self._oriented_points.clear()
        self._oriented_mouse_pos = None
        self._refresh()

    def pending_zone(self) -> dict | None:
        return self._draft_zone

    def clear_draft(self):
        self._draw_rect = None
        self._draft_zone = None
        self.set_mapping_mode(False)
        self._refresh()

    def reload_data(self):
        self._reload_data()
        self._refresh()

    def update_frame(self, frame: np.ndarray):
        self._frame = self._prepare_display_frame(frame)
        # Once an AI snapshot exists, repeatedly repainting that same snapshot
        # for every incoming live frame wastes X11/SSH bandwidth. Keep feeding
        # AI with the newest frame, but repaint only when a new result arrives.
        if self._detection_frame is None or not self._results or self._mapping_mode:
            self._refresh()

    def update_results(self, results: list[dict], source_frame: np.ndarray | None = None):
        self._results = results
        self._detection_frame = (
            self._prepare_display_frame(source_frame)
            if results and source_frame is not None
            else None
        )
        self._refresh()

    @staticmethod
    def _prepare_display_frame(frame: np.ndarray) -> np.ndarray:
        max_w = max(640, int(os.getenv("LIVE_VIEW_MAX_WIDTH", "960")))
        h, w = frame.shape[:2]
        if w <= max_w:
            return frame
        scale = max_w / w
        return cv2.resize(
            frame,
            (max_w, max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    def _reload_data(self):
        self._seats = db.monitor_seats(self._camera_id, self._class_id)
        self._students = {s["id"]: s for s in db.list_students()}

    def mousePressEvent(self, event):
        if not self._mapping_mode:
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._draw_mode == "oriented":
            pos = (int(event.position().x()), int(event.position().y()))
            self._oriented_points.append(pos)
            self._draft_zone = None
            if len(self._oriented_points) == 4:
                self._compute_oriented_zone()
            else:
                self._refresh()
            return

        # Normal mode
        self._drawing = True
        self._draw_start = event.position().toPoint()
        self._draw_rect = None
        self._draft_zone = None

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()

        if self._draw_mode == "oriented" and self._oriented_points:
            self._oriented_mouse_pos = (pos.x(), pos.y())
            self._refresh()
            return

        if not self._drawing or not self._draw_start:
            return
        from PySide6.QtCore import QRect
        self._draw_rect = QRect(self._draw_start, pos).normalized()
        self._refresh()

    def mouseReleaseEvent(self, event):
        if self._draw_mode == "oriented":
            return  # 4-click mode: points added on press, not drag

        if not self._drawing:
            return
        self._drawing = False

        # Normal: convert rect to normalized zone
        if not self._draw_rect or self._frame is None:
            return
        pix_rect = self._get_pixmap_rect()
        if not pix_rect:
            return
        px, py, pw, ph = pix_rect
        rect = self._draw_rect
        nx = (rect.x() - px) / pw
        ny = (rect.y() - py) / ph
        nw = rect.width() / pw
        nh = rect.height() / ph
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        nw = max(0.001, min(nw, 1.0 - nx))
        nh = max(0.001, min(nh, 1.0 - ny))
        self._draft_zone = {"type": "normal", "x": round(nx, 3), "y": round(ny, 3), "w": round(nw, 3), "h": round(nh, 3)}
        self.zone_drafted.emit(self._draft_zone)
        self._refresh()

    def _compute_oriented_zone(self):
        """Convert 4 clicked points to oriented zone dict using cv2.minAreaRect."""
        if len(self._oriented_points) != 4:
            return
        if self._frame is None:
            self._oriented_points.clear()
            self._oriented_mouse_pos = None
            return
        pix_rect = self._get_pixmap_rect()
        if not pix_rect:
            return
        px, py, pw, ph = pix_rect
        h_img, w_img = self._frame.shape[:2]

        # Convert 4 widget points → pixel coordinates on the original image
        pts = []
        for wx, wy in self._oriented_points:
            ix = (wx - px) / pw * w_img
            iy = (wy - py) / ph * h_img
            pts.append([ix, iy])
        pts = np.array(pts, dtype=np.float32)

        # Minimum-area rotated rectangle from the 4 points
        (cx, cy), (bw, bh), angle = cv2.minAreaRect(pts)

        # Normalise to 0..1 relative to image dimensions
        cx_n = max(0.0, min(1.0, cx / w_img))
        cy_n = max(0.0, min(1.0, cy / h_img))
        bw_n = max(0.01, min(bw / w_img, 1.0))
        bh_n = max(0.01, min(bh / h_img, 1.0))

        self._draft_zone = {
            "type": "oriented",
            "cx": round(cx_n, 3), "cy": round(cy_n, 3),
            "w": round(bw_n, 3), "h": round(bh_n, 3),
            "angle": round(angle, 1),
        }
        self._oriented_points.clear()
        self._oriented_mouse_pos = None
        self.zone_drafted.emit(self._draft_zone)
        self._refresh()

    def _get_pixmap_rect(self) -> tuple | None:
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        x = (lw - pw) // 2
        y = (lh - ph) // 2
        return x, y, pw, ph

    @staticmethod
    def _cv2_draw_oriented_zone(display, zone, w, h, color, thickness):
        """Draw oriented (rotated) zone on OpenCV image."""
        import math
        cx = int(zone["cx"] * w)
        cy = int(zone["cy"] * h)
        bw = int(zone["w"] * w)
        bh = int(zone["h"] * h)
        angle = zone.get("angle", 0)
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        # 4 corners relative to center, rotated
        hw, hh = bw / 2, bh / 2
        corners = np.array([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        ], dtype=np.float32)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = (corners @ rot.T + [cx, cy]).astype(np.int32)
        cv2.polylines(display, [pts], isClosed=True, color=color, thickness=thickness)

    def _refresh(self):
        # AI boxes and their source image are one atomic snapshot. Overlaying
        # coordinates from an older inference on the newest live frame makes
        # boxes visibly drift whenever a person moves or inference is slow.
        src = (
            self._detection_frame
            if self._detection_frame is not None and self._results and not self._mapping_mode
            else self._frame
        )
        if src is None:
            return
        display = src.copy()
        h, w = display.shape[:2]
        results_by_desk = {int(r.get("desk_num", 0)): r for r in self._results}

        for seat in self._seats:
            desk = int(seat.get("desk_num", 0))
            zone = seat.get("zone", {}) if isinstance(seat.get("zone"), dict) else {}
            if not zone:
                continue
            zone_type = zone.get("type", "normal")
            res = results_by_desk.get(desk)
            present_count = int(res.get("present_count", 0)) if res else -1
            wrong_count = int(res.get("wrong_count", 0)) if res else 0
            correct_count = int(res.get("correct_count", 0)) if res else 0
            if res is None:
                color = (14, 165, 233)
            elif wrong_count:
                color = (225, 29, 72)
            elif correct_count:
                color = (16, 185, 129)
            elif present_count:
                color = (245, 158, 11)
            else:
                color = (100, 116, 139)
            thickness = 3 if self._pending_desk == desk else 2

            if zone_type == "oriented":
                lx = int(zone.get("cx", 0.5) * w)
                ly = int(zone.get("cy", 0.5) * h)
                self._cv2_draw_oriented_zone(display, zone, w, h, color, thickness)
            else:
                zx = int(zone.get("x", 0) * w)
                zy = int(zone.get("y", 0) * h)
                zw = int(zone.get("w", 0.1) * w)
                zh = int(zone.get("h", 0.15) * h)
                cv2.rectangle(display, (zx, zy), (zx + zw, zy + zh), color, thickness)
                lx, ly = zx, zy

            cv2.putText(display, f"BAN {desk}", (lx + 4, max(ly - 6, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if res is not None:
                slots = len(seat.get("slots", []))
                cv2.putText(display, f"{present_count}/{slots} nguoi",
                            (lx + 4, ly + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

                source_w = max(1, int(res.get("source_width", w)))
                source_h = max(1, int(res.get("source_height", h)))
                sx, sy = w / source_w, h / source_h
                detections = [
                    *res.get("slot_results", []),
                    *res.get("extra_results", []),
                ]
                for detection in detections:
                    bbox = detection.get("person_bbox")
                    if not bbox:
                        continue
                    if detection.get("match_status") != "correct":
                        continue
                    bx, by, bw, bh = bbox
                    x1, y1 = round(bx * sx), round(by * sy)
                    x2, y2 = round((bx + bw) * sx), round((by + bh) * sy)
                    box_color = (16, 185, 129)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                    name = detection.get("recognized_name", "") or ""
                    if name:
                        cv2.putText(
                            display, name[:18], (x1 + 3, max(16, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2,
                        )
                    if detection.get("face_pose_ok") is False:
                        pose = {
                            "turned_left": "QUAY TRAI",
                            "turned_right": "QUAY PHAI",
                            "tilted": "NGHIENG DAU",
                        }.get(detection.get("face_pose"), "KHONG NHIN THANG")
                        cv2.putText(
                            display, pose, (x1 + 3, min(h - 5, y2 + 17)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2,
                        )
                    if detection.get("gaze_alert"):
                        yaw_v = detection.get("gaze_yaw_deg")
                        suffix = f" ({yaw_v:.0f}°)" if yaw_v is not None else ""
                        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(
                            display, f"KHONG TAP TRUNG{suffix}",
                            (x1 + 3, min(h - 5, y2 + 34)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2,
                        )

        # ── Draft zone overlay (drawn while the user is dragging/clicking) ──
        pix_rect = self._get_pixmap_rect()
        if pix_rect and self._mapping_mode:
            px_, py_, pw_, ph_ = pix_rect

            # Normal mode: dashed rect preview during drag
            if self._draw_rect and self._draw_mode == "normal":
                ix1 = int((self._draw_rect.x() - px_) / pw_ * w)
                iy1 = int((self._draw_rect.y() - py_) / ph_ * h)
                ix2 = int((self._draw_rect.x() + self._draw_rect.width() - px_) / pw_ * w)
                iy2 = int((self._draw_rect.y() + self._draw_rect.height() - py_) / ph_ * h)
                overlay = display.copy()
                cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (245, 158, 11), -1)
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                cv2.rectangle(display, (ix1, iy1), (ix2, iy2), (245, 158, 11), 2)

            # Oriented mode: 4-click preview
            if self._draw_mode == "oriented" and self._oriented_points:
                # Convert each stored point to image coords
                img_pts = []
                for wx, wy in self._oriented_points:
                    ix = int((wx - px_) / pw_ * w)
                    iy = int((wy - py_) / ph_ * h)
                    img_pts.append((ix, iy))
                    cv2.circle(display, (ix, iy), 7, (16, 185, 129), -1)
                    cv2.putText(display, str(len(img_pts)), (ix + 10, iy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (16, 185, 129), 2)

                # Lines between consecutive clicked points
                for i in range(1, len(img_pts)):
                    cv2.line(display, img_pts[i - 1], img_pts[i], (16, 185, 129), 2)

                # Preview line from last point to current mouse position
                if self._oriented_mouse_pos and len(self._oriented_points) < 4:
                    mx = int((self._oriented_mouse_pos[0] - px_) / pw_ * w)
                    my = int((self._oriented_mouse_pos[1] - py_) / ph_ * h)
                    cv2.line(display, img_pts[-1], (mx, my), (245, 158, 11), 1, cv2.LINE_AA)
                    cv2.circle(display, (mx, my), 5, (245, 158, 11), -1)

        # Draft zone (confirmed — emitted but not yet saved)
        if self._draft_zone:
            ztype = self._draft_zone.get("type", "normal")
            color = (245, 158, 11)
            if ztype == "oriented":
                self._cv2_draw_oriented_zone(display, self._draft_zone, w, h, color, 2)
            else:
                zx = int(self._draft_zone.get("x", 0) * w)
                zy = int(self._draft_zone.get("y", 0) * h)
                zw = int(self._draft_zone.get("w", 0.1) * w)
                zh = int(self._draft_zone.get("h", 0.15) * h)
                overlay = display.copy()
                cv2.rectangle(overlay, (zx, zy), (zx + zw, zy + zh), color, -1)
                cv2.addWeighted(overlay, 0.15, display, 0.85, 0, display)
                cv2.rectangle(display, (zx, zy), (zx + zw, zy + zh), color, 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pix)


class MonitorTab(QWidget):
    def __init__(self):
        super().__init__()
        self._camera_worker: CameraWorker | None = None
        self._ai_worker: AIDetectionWorker | None = None
        self._ai_active = False
        self._detection_mode = "centerpoint"
        self._last_camera_sequence = -1
        self._frame_timer = QTimer(self)
        display_fps = max(5, min(20, int(os.getenv("LIVE_VIEW_FPS", "12"))))
        self._frame_timer.setInterval(round(1000 / display_fps))
        self._frame_timer.timeout.connect(self._poll_camera_frame)
        # Re-pull seats + students from Redis on this cadence so a student
        # being moved between classes shows up in monitoring without restart.
        try:
            context_refresh_ms = max(
                1000, int(float(os.getenv("AI_CONTEXT_REFRESH_SEC", "5")) * 1000)
            )
        except ValueError:
            context_refresh_ms = 5000
        self._context_timer = QTimer(self)
        self._context_timer.setInterval(context_refresh_ms)
        self._context_timer.timeout.connect(self._refresh_ai_context)
        self._build_ui()

    def _refresh_ai_context(self):
        if not self._ai_worker or not self._ai_active:
            return
        cid = self.class_combo.currentData() or ""
        cam_id = self.cam_combo.currentData() or ""
        if not cid or not cam_id:
            return
        try:
            seats = db.monitor_seats(cam_id, cid)
            students = self._students_for_ai(cid, seats)
            self._ai_worker.update_context(seats, students)
            self._reload_status_tree(cid)
        except Exception as exc:
            print(f"[ai] context refresh failed: {exc}", flush=True)

    def _build_ui(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(12, 12, 6, 12)
        left_lay.setSpacing(10)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Lớp:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(160)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        ctrl.addWidget(self.class_combo)
        ctrl.addWidget(QLabel("Camera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(180)
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        ctrl.addWidget(self.cam_combo)
        self.btn_connect = QPushButton("▶  Kết nối")
        self.btn_connect.setObjectName("btn_add")
        self.btn_connect.clicked.connect(self._start_camera)
        ctrl.addWidget(self.btn_connect)
        self.btn_disconnect = QPushButton("◼  Ngắt")
        self.btn_disconnect.clicked.connect(self._stop_camera)
        ctrl.addWidget(self.btn_disconnect)
        ctrl.addStretch()
        left_lay.addLayout(ctrl)

        self.canvas = ZoneCanvas()
        self.canvas.zone_drafted.connect(self._on_zone_drafted)
        left_lay.addWidget(self.canvas)

        ai_row = QHBoxLayout()
        self.btn_ai = QPushButton("🤖  Bật AI kiểm tra")
        self.btn_ai.setObjectName("btn_ai")
        self.btn_ai.setCheckable(True)
        self.btn_ai.clicked.connect(self._toggle_ai)
        ai_row.addWidget(self.btn_ai)
        self.btn_detection_mode = QPushButton("📍  Tâm hộp")
        self.btn_detection_mode.setCheckable(True)
        self.btn_detection_mode.setChecked(True)
        self.btn_detection_mode.setToolTip("Chuyển đổi giữa Perspective (điểm chân) và Centerpoint (tâm hộp)")
        self.btn_detection_mode.clicked.connect(self._toggle_detection_mode)
        ai_row.addWidget(self.btn_detection_mode)
        ai_row.addStretch()
        self.lbl_ai_status = QLabel("AI: Chưa chạy")
        self.lbl_ai_status.setObjectName("status_err")
        ai_row.addWidget(self.lbl_ai_status)
        left_lay.addLayout(ai_row)
        splitter.addWidget(left)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(6, 12, 12, 12)
        right_lay.setSpacing(10)

        title = QLabel("TRẠNG THÁI CHỖ NGỒI THEO BÀN")
        title.setObjectName("section_title")
        right_lay.addWidget(title)

        self.status_tree = QTreeWidget()
        self.status_tree.setHeaderLabels(["Bàn / Chỗ", "Học sinh gán chỗ", "Hiện diện", "Nhận diện mặt", "Đúng vị trí?"])
        self.status_tree.setRootIsDecorated(True)
        self.status_tree.setAlternatingRowColors(True)
        self.status_tree.header().setStretchLastSection(False)
        self.status_tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.status_tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.status_tree.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.status_tree.header().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.status_tree.header().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        right_lay.addWidget(self.status_tree)

        map_box = QGroupBox("Mapping vùng camera ↔ bàn học")
        map_lay = QVBoxLayout(map_box)
        map_hint = QLabel(
            "Chọn một bàn rồi khoanh toàn bộ khu vực bàn trên camera. "
            "Vùng được lưu riêng theo cặp camera–lớp và áp dụng cho tất cả chỗ của bàn."
        )
        map_hint.setWordWrap(True)
        map_lay.addWidget(map_hint)

        desk_row = QHBoxLayout()
        desk_row.addWidget(QLabel("Bàn:"))
        self.desk_combo = QComboBox()
        self.desk_combo.setMinimumWidth(80)
        desk_row.addWidget(self.desk_combo)
        desk_row.addStretch()
        map_lay.addLayout(desk_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Kiểu vùng:"))
        self.btn_mode_normal = QPushButton("▣  Chữ nhật")
        self.btn_mode_normal.setObjectName("btn_add")
        self.btn_mode_normal.setCheckable(True)
        self.btn_mode_normal.setChecked(True)
        self.btn_mode_oriented = QPushButton("◆  Xoay")
        self.btn_mode_oriented.setCheckable(True)
        self.btn_mode_normal.clicked.connect(lambda: self._set_zone_mode("normal"))
        self.btn_mode_oriented.clicked.connect(lambda: self._set_zone_mode("oriented"))
        mode_row.addWidget(self.btn_mode_normal)
        mode_row.addWidget(self.btn_mode_oriented)
        mode_row.addStretch()
        map_lay.addLayout(mode_row)

        draw_row = QHBoxLayout()
        self.btn_draw_zone = QPushButton("✏  Vẽ vùng")
        self.btn_draw_zone.setObjectName("btn_edit")
        self.btn_save_zone = QPushButton("💾  Lưu vùng")
        self.btn_save_zone.setObjectName("btn_add")
        self.btn_save_zone.setEnabled(False)
        self.btn_cancel_zone = QPushButton("Hủy vẽ")
        self.btn_draw_zone.clicked.connect(self._start_draw_zone)
        self.btn_save_zone.clicked.connect(self._save_drawn_zone)
        self.btn_cancel_zone.clicked.connect(self._cancel_draw_zone)
        self.btn_remove_zone = QPushButton("⌫  Xóa vùng")
        self.btn_remove_zone.setObjectName("btn_del")
        self.btn_remove_zone.clicked.connect(self._remove_saved_zone)
        draw_row.addWidget(self.btn_draw_zone)
        draw_row.addWidget(self.btn_save_zone)
        draw_row.addWidget(self.btn_cancel_zone)
        draw_row.addWidget(self.btn_remove_zone)
        draw_row.addStretch()
        map_lay.addLayout(draw_row)
        right_lay.addWidget(map_box)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        right_lay.addWidget(self.lbl_summary)
        splitter.addWidget(right)
        splitter.setSizes([760, 420])
        outer.addWidget(splitter)

        self._load_combos()

    def _load_combos(self):
        old_class = self.class_combo.currentData()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("-- Chọn lớp --", "")
        for c in db.list_classrooms():
            self.class_combo.addItem(c["name"], c["id"])
        self.class_combo.blockSignals(False)
        if old_class:
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == old_class:
                    self.class_combo.setCurrentIndex(i)
                    break
        self._on_class_changed()

    def _on_class_changed(self, *_):
        cid = self.class_combo.currentData() or ""
        if cid:
            db.ensure_classroom_seats(cid)
        self._reload_cameras(cid)
        self._on_camera_changed()
        self.btn_save_zone.setEnabled(False)

    def _reload_cameras(self, class_id: str):
        current = self.cam_combo.currentData()
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        for cam in db.list_cameras():
            if not class_id or class_id in cam.get("class_ids", []) or cam.get("class_id") == class_id:
                self.cam_combo.addItem(cam["name"], cam["id"])
        if current:
            for i in range(self.cam_combo.count()):
                if self.cam_combo.itemData(i) == current:
                    self.cam_combo.setCurrentIndex(i)
                    break
        self.cam_combo.blockSignals(False)

    def _on_camera_changed(self, *_):
        cid = self.class_combo.currentData() or ""
        cam_id = self.cam_combo.currentData() or ""
        self.canvas.set_context(cid, cam_id)
        self._reload_desks(cid)
        self._reload_status_tree(cid)
        if self._ai_worker:
            seats = db.monitor_seats(cam_id, cid)
            self._ai_worker.update_context(seats, self._students_for_ai(cid, seats) if cid else [])
        self.btn_save_zone.setEnabled(False)

    def _reload_desks(self, class_id: str):
        current = self.desk_combo.currentData()
        self.desk_combo.blockSignals(True)
        self.desk_combo.clear()
        if class_id:
            cam_id = self.cam_combo.currentData() or ""
            for seat in db.monitor_seats(cam_id, class_id):
                desk = int(seat.get("desk_num", 0))
                label = f"Bàn {desk}"
                if seat.get("zone"):
                    label += "  ✓ đã khoanh"
                self.desk_combo.addItem(label, desk)
        self.desk_combo.blockSignals(False)
        if current:
            for i in range(self.desk_combo.count()):
                if self.desk_combo.itemData(i) == current:
                    self.desk_combo.setCurrentIndex(i)
                    break

    def _reload_status_tree(self, class_id: str):
        self.status_tree.clear()
        if not class_id:
            return
        students = {s["id"]: s for s in db.list_students()}
        cam_id = self.cam_combo.currentData() or ""
        for seat in db.monitor_seats(cam_id, class_id):
            desk = int(seat.get("desk_num", 0))
            slots = seat.get("slots", [])
            assigned = sum(1 for slot in slots if slot.get("student_id"))
            has_zone = bool(seat.get("zone"))
            parent = QTreeWidgetItem([
                f"Bàn {desk} ({len(slots)} chỗ)",
                f"{assigned}/{len(slots)} đã gán",
                "Đã có vùng AI" if has_zone else "Chưa có vùng AI",
                "",
                "",
            ])
            parent.setData(0, ROLE_DESK, desk)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            self.status_tree.addTopLevelItem(parent)
            for slot in slots:
                slot_num = int(slot.get("slot_num", 1))
                sid = slot.get("student_id", "")
                student = students.get(sid, {})
                student_text = f"{student.get('student_code','')} - {student.get('name','')}" if student else "-- trống --"
                presence = "⏳ Chờ AI" if has_zone else "Chưa có vùng AI"
                face_status = "—" if has_zone else "Chưa cấu hình vùng bàn"
                position_status = "—"
                child = QTreeWidgetItem([f"  Chỗ {slot_num}", student_text, presence, face_status, position_status])
                child.setData(0, ROLE_DESK, desk)
                child.setData(0, ROLE_SLOT, slot_num)
                child.setForeground(2, QColor("#64748b"))
                child.setForeground(3, QColor("#64748b"))
                child.setForeground(4, QColor("#64748b"))
                parent.addChild(child)
            parent.setExpanded(True)

    def _students_for_ai(self, class_id: str, seats: list[dict]) -> list[dict]:
        """Return students that currently BELONG to ``class_id``.

        A seat slot can still reference a student who has been moved to another
        class. Such stale references must not be re-added, otherwise the AI
        gallery would keep recognising them at the old class forever.
        """
        students_by_id = {s["id"]: s for s in db.list_students(class_id)}
        target = str(class_id or "")
        for seat in seats:
            for slot in seat.get("slots", []) or []:
                sid = str(slot.get("student_id", "") or "")
                if not sid or sid in students_by_id:
                    continue
                student = db.get_student(sid)
                if student and str(student.get("class_id") or "") == target:
                    students_by_id[sid] = student
        return list(students_by_id.values())

    def _start_camera(self):
        cam_id = self.cam_combo.currentData()
        if not cam_id:
            return
        cam = db.get_camera(cam_id)
        if not cam:
            return
        self._stop_camera()
        self._last_camera_sequence = -1
        self._camera_worker = CameraWorker(cam_id)
        self._camera_worker.error.connect(lambda e: self.lbl_ai_status.setText(f"Lỗi: {e}"))
        self._camera_worker.status.connect(self.lbl_ai_status.setText)
        self._camera_worker.start()
        self._frame_timer.start()

    def _stop_camera(self):
        self._frame_timer.stop()
        if self._camera_worker:
            self._camera_worker.stop()
            self._camera_worker = None
        self._last_camera_sequence = -1
        self._stop_ai()

    def _poll_camera_frame(self):
        if not self._camera_worker:
            return
        sequence, frame = self._camera_worker.latest_frame(self._last_camera_sequence)
        if frame is None:
            return
        self._last_camera_sequence = sequence
        self.canvas.update_frame(frame)
        if self._ai_active and self._ai_worker:
            self._ai_worker.update_frame(frame)

    def _toggle_ai(self, checked: bool):
        if checked:
            self._start_ai()
        else:
            self._stop_ai()

    def _start_ai(self):
        cid = self.class_combo.currentData() or ""
        cam_id = self.cam_combo.currentData() or ""
        seats = db.monitor_seats(cam_id, cid)
        has_zone = any(seat.get("zone") for seat in seats)
        if not has_zone:
            self.btn_ai.blockSignals(True)
            self.btn_ai.setChecked(False)
            self.btn_ai.blockSignals(False)
            self.lbl_ai_status.setText("AI: Chưa có vùng bàn nào cho camera này")
            return
        if self._ai_worker:
            self._ai_worker.stop()
        self._ai_worker = AIDetectionWorker()
        self._ai_worker.set_detection_mode(self._detection_mode)
        self._ai_worker.update_context(seats, self._students_for_ai(cid, seats))
        self._ai_worker.detection_ready.connect(self._on_detection)
        self._ai_worker.start()
        self._ai_active = True
        self._context_timer.start()
        self.lbl_ai_status.setText("AI: Đang chạy 🟢")
        self.lbl_ai_status.setObjectName("status_ok")
        self.lbl_ai_status.style().unpolish(self.lbl_ai_status)
        self.lbl_ai_status.style().polish(self.lbl_ai_status)

    def _stop_ai(self):
        self._context_timer.stop()
        if self._ai_worker:
            # Disconnect first so queued detection signals don't re-populate canvas
            self._ai_worker.detection_ready.disconnect(self._on_detection)
            self._ai_worker.stop()
            self._ai_worker = None
        self._ai_active = False
        self.btn_ai.blockSignals(True)
        self.btn_ai.setChecked(False)
        self.btn_ai.blockSignals(False)
        self.lbl_ai_status.setText("AI: Chưa chạy")
        # Clear detection overlay on canvas + status tree
        self.canvas.update_results([])
        self._update_status_tree([])

    def _on_detection(self, results: list[dict], source_frame: np.ndarray):
        self.canvas.update_results(results, source_frame)
        self._update_status_tree(results)

    def _find_status_child(self, desk: int, slot_num: int) -> QTreeWidgetItem | None:
        for i in range(self.status_tree.topLevelItemCount()):
            parent = self.status_tree.topLevelItem(i)
            if parent.data(0, ROLE_DESK) != desk:
                continue
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.data(0, ROLE_SLOT) == slot_num:
                    return child
        return None

    def _update_status_tree(self, results: list[dict]):
        # Reset all status rows when AI stops (empty results)
        if not results:
            for i in range(self.status_tree.topLevelItemCount()):
                parent = self.status_tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    child = parent.child(j)
                    child.setText(2, "⏳ Chờ AI")
                    child.setText(3, "—")
                    child.setText(4, "—")
                    child.setForeground(2, QColor("#64748b"))
                    child.setForeground(3, QColor("#64748b"))
                    child.setForeground(4, QColor("#64748b"))
            self.lbl_summary.setText("")
            return

        flat_results = []
        for desk_result in results:
            flat_results.extend(desk_result.get("slot_results", []))
            flat_results.extend(desk_result.get("extra_results", []))

        present_count = 0       # persons matched into a real slot
        checked_count = 0       # number of slot rows updated
        correct_count = 0
        wrong_count = 0
        recognized_count = 0
        unassigned_count = 0    # extras in zone with no slot to bind to
        total_in_zone = 0       # total persons inside any parent zone
        # Per-desk aggregates so each "Bàn" row can show its own summary.
        per_desk: dict[int, dict] = {}
        def _desk(d: int) -> dict:
            return per_desk.setdefault(d, {
                "in_zone": 0, "matched": 0, "unassigned": 0,
                "correct": 0, "wrong": 0, "slot_count": 0,
            })
        for res in flat_results:
            desk = int(res.get("desk_num", 0))
            slot_num = int(res.get("slot_num", 1))
            present = bool(res.get("present", False))
            d = _desk(desk)
            if present:
                total_in_zone += 1
                d["in_zone"] += 1
            if slot_num == -1:
                # Extra person inside a zone but no slot available
                if present:
                    unassigned_count += 1
                    d["unassigned"] += 1
                continue
            child = self._find_status_child(desk, slot_num)
            if not child:
                continue
            checked_count += 1
            d["slot_count"] += 1
            if present:
                present_count += 1
                d["matched"] += 1

            recognized_name = res.get("recognized_name", "") or ""
            recognized_code = res.get("recognized_code", "") or ""
            score = res.get("recognition_score")
            match_status = res.get("match_status", "") or ""
            match_text = res.get("match_text", "") or ""

            if recognized_name:
                recognized_count += 1
                face_text = f"👤 {recognized_code} - {recognized_name}" if recognized_code else f"👤 {recognized_name}"
                if score is not None:
                    face_text += f" ({float(score):.2f})"
            elif match_status == "no_gallery":
                face_text = "⚠ Chưa có ảnh mẫu"
            elif match_status == "no_face":
                face_text = "⚠ Chưa thấy mặt"
            elif match_status == "unknown":
                face_text = "❔ Có mặt nhưng chưa khớp HS"
            elif match_status == "empty":
                face_text = "—"
            else:
                face_text = "—"

            if res.get("face_pose_ok") is False:
                pose_labels = {
                    "turned_left": "quay trái",
                    "turned_right": "quay phải",
                    "tilted": "nghiêng đầu",
                }
                pose_text = pose_labels.get(res.get("face_pose"), "không nhìn thẳng")
                face_text += f"  ⚠ {pose_text}"

            if res.get("gaze_alert"):
                yaw_v = res.get("gaze_yaw_deg")
                if yaw_v is not None:
                    face_text += f"  🚨 KHÔNG TẬP TRUNG (yaw={yaw_v:.0f}°)"
                else:
                    face_text += "  🚨 KHÔNG TẬP TRUNG"

            # Real-world position (from perspective transform)
            person_real = res.get("person_real_cm")
            if present and person_real:
                face_text += f"  📐 ({person_real[0]:.0f},{person_real[1]:.0f})cm"

            if match_status == "correct":
                correct_count += 1
                d["correct"] += 1
                position_color = QColor("#059669")
            elif match_status in {"wrong", "unassigned"}:
                wrong_count += 1
                d["wrong"] += 1
                position_color = QColor("#e11d48")
            elif present:
                position_color = QColor("#d97706")
            else:
                position_color = QColor("#64748b")

            child.setText(2, "✅ Có người" if present else "❌ Trống / vắng")
            child.setText(3, face_text)
            child.setText(4, match_text or "—")
            child.setForeground(2, QColor("#059669") if present else QColor("#e11d48"))
            child.setForeground(3, QColor("#0f172a") if recognized_name else QColor("#64748b"))
            child.setForeground(4, position_color)

        # Update each Bàn parent row with its own per-desk summary (column 2 = "Hiện diện")
        for i in range(self.status_tree.topLevelItemCount()):
            parent = self.status_tree.topLevelItem(i)
            desk = parent.data(0, ROLE_DESK)
            d = per_desk.get(int(desk) if desk is not None else -999)
            if not d:
                continue
            extra = f" (+{d['unassigned']} thừa)" if d["unassigned"] else ""
            parent.setText(2, f"{d['in_zone']} người trong vùng{extra}")
            parent.setText(3, f"{d['correct']}/{d['slot_count']} đúng")
            if d["wrong"]:
                parent.setText(4, f"{d['wrong']} sai")
                parent.setForeground(4, QColor("#e11d48"))
            else:
                parent.setText(4, "—")
                parent.setForeground(4, QColor("#64748b"))
            parent.setForeground(2,
                QColor("#059669") if d["in_zone"] > 0 else QColor("#64748b"))
            parent.setForeground(3,
                QColor("#059669") if d["correct"] == d["slot_count"] and d["slot_count"] > 0
                else QColor("#d97706") if d["correct"] > 0
                else QColor("#64748b"))

        extra_txt = f"  •  Thừa trong vùng: {unassigned_count}" if unassigned_count else ""
        self.lbl_summary.setText(
            f"📊 Trong vùng: {total_in_zone} người  •  "
            f"Khớp chỗ: {present_count}/{checked_count}  •  "
            f"Nhận diện được: {recognized_count}/{checked_count}  •  "
            f"Đúng vị trí: {correct_count}  •  Sai vị trí: {wrong_count}"
            f"{extra_txt}"
        )

    def _toggle_detection_mode(self, checked: bool):
        if checked:
            self._detection_mode = "centerpoint"
            self.btn_detection_mode.setText("📍  Tâm hộp")
        else:
            self._detection_mode = "perspective"
            self.btn_detection_mode.setText("📐  Perspective")
        if self._ai_worker:
            self._ai_worker.set_detection_mode(self._detection_mode)

    def _set_zone_mode(self, mode: str):
        """Switch zone drawing between normal (rect) and oriented (rotated)."""
        other = "oriented" if mode == "normal" else "normal"
        getattr(self, f"btn_mode_{mode}").setChecked(True)
        getattr(self, f"btn_mode_{other}").setChecked(False)
        self.canvas.set_draw_mode(mode)
        mode_label = "Chữ nhật" if mode == "normal" else "Xoay"
        self.lbl_ai_status.setText(f"🖊  Kiểu vùng: {mode_label}")

    def _start_draw_zone(self):
        desk = self.desk_combo.currentData()
        cam_id = self.cam_combo.currentData()
        if desk is None or not cam_id:
            return
        self.canvas.set_mapping_mode(True, int(desk))
        self.btn_save_zone.setEnabled(False)
        self.lbl_ai_status.setText(f"🖊 Đang khoanh toàn bộ bàn {desk}; vẽ xong bấm Lưu vùng")

    def _on_zone_drafted(self, zone: dict):
        self.btn_save_zone.setEnabled(True)
        self.lbl_ai_status.setText("Đã vẽ vùng tạm. Bấm Lưu vùng để cập nhật danh sách và overlay.")

    def _save_drawn_zone(self):
        cid = self.class_combo.currentData() or ""
        cam_id = self.cam_combo.currentData() or ""
        desk = self.desk_combo.currentData()
        zone = self.canvas.pending_zone()
        if not cid or not cam_id or desk is None or not zone:
            self.lbl_ai_status.setText("Chưa có vùng tạm để lưu")
            return
        db.set_desk_region(cam_id, cid, int(desk), zone)
        self.canvas.set_mapping_mode(False)
        self.canvas.reload_data()
        self._reload_desks(cid)
        self._reload_status_tree(cid)
        if self._ai_worker:
            seats = db.monitor_seats(cam_id, cid)
            self._ai_worker.update_context(seats, self._students_for_ai(cid, seats) if cid else [])
        self.btn_save_zone.setEnabled(False)
        self.lbl_ai_status.setText(f"✅ Đã lưu vùng cho toàn bộ bàn {desk}")

    def _cancel_draw_zone(self):
        self.canvas.clear_draft()
        self.btn_save_zone.setEnabled(False)
        self.lbl_ai_status.setText("Đã hủy vùng đang vẽ")

    def _remove_saved_zone(self):
        """Remove the selected desk region for the current camera-class pair."""
        cid = self.class_combo.currentData() or ""
        cam_id = self.cam_combo.currentData() or ""
        desk = self.desk_combo.currentData()
        if not cid or not cam_id or desk is None:
            self.lbl_ai_status.setText("Chọn camera và bàn cần xóa vùng")
            return
        db.clear_desk_region(cam_id, cid, int(desk))
        self.canvas.reload_data()
        self._reload_desks(cid)
        self._reload_status_tree(cid)
        if self._ai_worker:
            seats = db.monitor_seats(cam_id, cid)
            self._ai_worker.update_context(seats, self._students_for_ai(cid, seats) if cid else [])
        self.lbl_ai_status.setText(f"✅ Đã xóa vùng bàn {desk}")

    def refresh(self):
        self._load_combos()

    def closeEvent(self, event):
        self._stop_camera()
        super().closeEvent(event)
