"""Camera management tab with live Middleware SHM preview."""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QDialog, QFormLayout, QLineEdit, QComboBox,
    QMessageBox, QHeaderView, QAbstractItemView, QSplitter,
    QSpinBox, QCheckBox, QGroupBox, QGridLayout, QSizePolicy,
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap
import numpy as np
import cv2

from db import redis_client as db
from workers.camera_worker import CameraWorker


class CameraDialog(QDialog):
    def __init__(self, parent=None, data: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Thêm camera" if not data else "Sửa camera")
        self.setMinimumWidth(720)
        self.data = data or {}

        outer = QVBoxLayout(self)
        outer.setSpacing(14)
        outer.setContentsMargins(24, 24, 24, 24)

        identity_box = QGroupBox("Thông tin camera")
        identity = QGridLayout(identity_box)
        self.name_edit = QLineEdit(self.data.get("name", ""))
        self.name_edit.setPlaceholderText("Ví dụ: Camera cửa lớp")
        self.location_edit = QLineEdit(self.data.get("location", ""))
        self.location_edit.setPlaceholderText("Vị trí lắp đặt")
        self.brand_edit = QComboBox()
        self.brand_edit.setEditable(True)
        self.brand_edit.addItems(["Dahua", "Hikvision", "KBVision", "Ezviz", "Imou", "Khác"])
        self.brand_edit.setCurrentText(self.data.get("brand", "Dahua") or "Dahua")
        self.model_edit = QLineEdit(self.data.get("model", ""))
        identity.addWidget(QLabel("Tên camera *"), 0, 0)
        identity.addWidget(self.name_edit, 0, 1)
        identity.addWidget(QLabel("Vị trí"), 0, 2)
        identity.addWidget(self.location_edit, 0, 3)
        identity.addWidget(QLabel("Hãng"), 1, 0)
        identity.addWidget(self.brand_edit, 1, 1)
        identity.addWidget(QLabel("Model"), 1, 2)
        identity.addWidget(self.model_edit, 1, 3)
        outer.addWidget(identity_box)

        connection_box = QGroupBox("Kết nối thiết bị")
        connection = QGridLayout(connection_box)
        self.ip_edit = QLineEdit(self.data.get("ipAddress", ""))
        self.ip_edit.setPlaceholderText("192.168.1.100")
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(int(self.data.get("port", 554) or 554))
        self.onvif_port_spin = QSpinBox()
        self.onvif_port_spin.setRange(1, 65535)
        self.onvif_port_spin.setValue(int(self.data.get("onvifPort", 80) or 80))
        self.username_edit = QLineEdit(self.data.get("username", ""))
        self.password_edit = QLineEdit(self.data.get("password", ""))
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.ws_port_spin = QSpinBox()
        self.ws_port_spin.setRange(0, 65535)
        self.ws_port_spin.setSpecialValueText("Không dùng")
        self.ws_port_spin.setValue(int(self.data.get("wsPort", 0) or 0))
        connection.addWidget(QLabel("Địa chỉ IP *"), 0, 0)
        connection.addWidget(self.ip_edit, 0, 1)
        connection.addWidget(QLabel("Cổng RTSP"), 0, 2)
        connection.addWidget(self.port_spin, 0, 3)
        connection.addWidget(QLabel("Tài khoản"), 1, 0)
        connection.addWidget(self.username_edit, 1, 1)
        connection.addWidget(QLabel("Mật khẩu"), 1, 2)
        connection.addWidget(self.password_edit, 1, 3)
        connection.addWidget(QLabel("Cổng ONVIF"), 2, 0)
        connection.addWidget(self.onvif_port_spin, 2, 1)
        connection.addWidget(QLabel("Cổng WebSocket"), 2, 2)
        connection.addWidget(self.ws_port_spin, 2, 3)
        outer.addWidget(connection_box)

        stream_box = QGroupBox("Luồng hình ảnh và lớp học")
        stream = QFormLayout(stream_box)
        self.rtsp_path_edit = QLineEdit(
            self.data.get("rtspPath", self.data.get("notes", ""))
            or "/cam/realmonitor?channel=1&subtype=0"
        )
        self.rtsp_path_edit.setPlaceholderText("/cam/realmonitor?channel=1&subtype=0")
        self.stream_url_edit = QLineEdit(self.data.get("streamUrl", ""))
        self.stream_url_edit.setPlaceholderText("/cctv-ws/9999 (nếu có)")
        self.class_combo = QComboBox()
        self.class_combo.addItem("-- Không gắn lớp --", "")
        for c in db.list_classrooms():
            self.class_combo.addItem(c["name"], c["id"])
        cid = self.data.get("class_id", "")
        for i in range(self.class_combo.count()):
            if self.class_combo.itemData(i) == cid:
                self.class_combo.setCurrentIndex(i)
                break
        self.active_check = QCheckBox("Camera đang hoạt động")
        self.active_check.setChecked(str(self.data.get("isActive", "1")).lower() in {"1", "true", "yes"})
        stream.addRow("Đường dẫn RTSP *:", self.rtsp_path_edit)
        stream.addRow("Đường dẫn WebSocket:", self.stream_url_edit)
        stream.addRow("Lớp học:", self.class_combo)
        stream.addRow("", self.active_check)
        outer.addWidget(stream_box)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("Lưu")
        self.btn_ok.setObjectName("btn_add")
        self.btn_cancel = QPushButton("Hủy")
        self.btn_ok.clicked.connect(self._validate)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        outer.addLayout(btns)

    def _validate(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên camera.")
            return
        if not self.ip_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập địa chỉ IP camera.")
            return
        if not self.rtsp_path_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập đường dẫn RTSP.")
            return
        self.accept()

    def result_data(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "location": self.location_edit.text().strip(),
            "brand": self.brand_edit.currentText().strip(),
            "model": self.model_edit.text().strip(),
            "ipAddress": self.ip_edit.text().strip(),
            "port": self.port_spin.value(),
            "onvifPort": self.onvif_port_spin.value(),
            "username": self.username_edit.text().strip(),
            "password": self.password_edit.text(),
            "wsPort": self.ws_port_spin.value() or "",
            "streamUrl": self.stream_url_edit.text().strip(),
            "rtspPath": self.rtsp_path_edit.text().strip(),
            "notes": self.rtsp_path_edit.text().strip(),
            "isActive": self.active_check.isChecked(),
            "class_id": self.class_combo.currentData() or "",
        }


class CameraPreviewWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setObjectName("cam_label")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Chọn camera và nhấn 'Xem thử'")
        self.setMinimumSize(QSize(480, 270))
        # Ignore pixmap sizeHint so a new frame cannot progressively enlarge
        # its QLabel/QSplitter (the visible "slow zoom" over X11).
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self._worker: CameraWorker | None = None
        self._last_sequence = -1
        self._display_timer = QTimer(self)
        display_fps = max(5, min(20, int(os.getenv("LIVE_VIEW_FPS", "12"))))
        self._display_timer.setInterval(round(1000 / display_fps))
        self._display_timer.timeout.connect(self._poll_frame)

    def start(self, camera_id: str):
        self.stop()
        self._last_sequence = -1
        self._set_status("Đang kết nối Middleware SHM...")
        self._worker = CameraWorker(camera_id)
        self._worker.error.connect(lambda e: self._set_status(f"Lỗi kết nối:\n{e}"))
        self._worker.status.connect(self._set_status)
        self._worker.start()
        self._display_timer.start()

    def stop(self):
        self._display_timer.stop()
        if self._worker:
            self._worker.stop()
            self._worker = None
        self._last_sequence = -1
        self._set_status("Đã dừng stream")

    def _set_status(self, text: str):
        self.clear()
        self.setText(text)

    def _poll_frame(self):
        if not self._worker:
            return
        sequence, frame = self._worker.latest_frame(self._last_sequence)
        if frame is None:
            return
        self._last_sequence = sequence
        self._show_frame(frame)

    def _show_frame(self, frame: np.ndarray):
        # Downscale before converting to a QImage. Sending a multi-megapixel
        # image through X11 for every frame is expensive and adds no visible
        # detail inside this preview widget.
        max_w = max(480, int(os.getenv("LIVE_VIEW_MAX_WIDTH", "960")))
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(
                frame, (max_w, max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.setPixmap(pix)

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)


class CamerasTab(QWidget):
    def __init__(self):
        super().__init__()
        self._preview: CameraPreviewWidget | None = None
        self._build_ui()
        self.load_data()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setSpacing(12)
        vlay.setContentsMargins(16, 16, 16, 16)

        hdr = QHBoxLayout()
        title = QLabel("QUẢN LÝ CAMERA GIÁM SÁT")
        title.setObjectName("section_title")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_count = QLabel("")
        hdr.addWidget(self.lbl_count)
        vlay.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Table
        table_w = QWidget()
        tlay = QVBoxLayout(table_w)
        tlay.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Tên camera", "Thiết bị", "Hãng / model", "Vị trí", "Lớp", "Trạng thái", "ID"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnHidden(6, True)
        tlay.addWidget(self.table)
        splitter.addWidget(table_w)

        # Preview
        prev_w = QWidget()
        play = QVBoxLayout(prev_w)
        play.setContentsMargins(0, 0, 0, 0)
        play.addWidget(QLabel("Xem trước camera:"))
        self._preview = CameraPreviewWidget()
        play.addWidget(self._preview)

        preview_btns = QHBoxLayout()
        self.btn_preview = QPushButton("▶  Xem thử")
        self.btn_preview.setObjectName("btn_add")
        self.btn_stop = QPushButton("◼  Dừng")
        self.btn_preview.clicked.connect(self._start_preview)
        self.btn_stop.clicked.connect(self._preview.stop)
        preview_btns.addWidget(self.btn_preview)
        preview_btns.addWidget(self.btn_stop)
        preview_btns.addStretch()
        play.addLayout(preview_btns)
        splitter.addWidget(prev_w)

        splitter.setSizes([450, 500])
        vlay.addWidget(splitter)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("＋  Thêm camera")
        self.btn_add.setObjectName("btn_add")
        self.btn_edit = QPushButton("✎  Sửa")
        self.btn_edit.setObjectName("btn_edit")
        self.btn_del = QPushButton("✕  Xóa")
        self.btn_del.setObjectName("btn_del")
        self.btn_refresh = QPushButton("↻  Làm mới")

        self.btn_add.clicked.connect(self.add_camera)
        self.btn_edit.clicked.connect(self.edit_camera)
        self.btn_del.clicked.connect(self.delete_camera)
        self.btn_refresh.clicked.connect(self.load_data)

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_del)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_refresh)
        vlay.addLayout(btn_row)

    def load_data(self):
        cameras = db.list_cameras()
        classes = {c["id"]: c["name"] for c in db.list_classrooms()}
        self.table.setRowCount(len(cameras))
        for r, cam in enumerate(cameras):
            self.table.setItem(r, 0, QTableWidgetItem(cam.get("name", "")))
            address = cam.get("ipAddress", "")
            if address:
                address += f":{cam.get('port', '554')}"
            self.table.setItem(r, 1, QTableWidgetItem(address))
            brand_model = " / ".join(
                value for value in [cam.get("brand", ""), cam.get("model", "")] if value
            )
            self.table.setItem(r, 2, QTableWidgetItem(brand_model or "—"))
            self.table.setItem(r, 3, QTableWidgetItem(cam.get("location", "") or "—"))
            cls_name = classes.get(cam.get("class_id", ""), "—")
            self.table.setItem(r, 4, QTableWidgetItem(cls_name))
            active = str(cam.get("isActive", "1")).lower() in {"1", "true", "yes"}
            self.table.setItem(r, 5, QTableWidgetItem("● Hoạt động" if active else "○ Tạm dừng"))
            self.table.setItem(r, 6, QTableWidgetItem(cam.get("id", "")))
        self.lbl_count.setText(f"{len(cameras)} camera")

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 6).text()

    def _start_preview(self):
        cam_id = self._selected_id()
        if not cam_id:
            QMessageBox.information(self, "Chọn camera", "Vui lòng chọn camera cần xem.")
            return
        if db.get_camera(cam_id):
            self._preview.start(cam_id)

    def add_camera(self):
        dlg = CameraDialog(self)
        if dlg.exec():
            d = dlg.result_data()
            db.create_camera(d)
            self.load_data()

    def edit_camera(self):
        cam_id = self._selected_id()
        if not cam_id:
            QMessageBox.information(self, "Chọn camera", "Vui lòng chọn camera cần sửa.")
            return
        current = db.get_camera(cam_id)
        dlg = CameraDialog(self, current)
        if dlg.exec():
            d = dlg.result_data()
            db.update_camera(cam_id, d)
            self.load_data()

    def delete_camera(self):
        cam_id = self._selected_id()
        if not cam_id:
            QMessageBox.information(self, "Chọn camera", "Vui lòng chọn camera cần xóa.")
            return
        cam = db.get_camera(cam_id)
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Xóa camera '{cam.get('name', '')}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            db.delete_camera(cam_id)
            self.load_data()

    def closeEvent(self, event):
        if self._preview:
            self._preview.stop()
        super().closeEvent(event)
