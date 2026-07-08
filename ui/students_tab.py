"""Students CRUD tab with student face image management."""

from __future__ import annotations

from pathlib import Path
import time

import cv2

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QDialog, QFormLayout, QLineEdit, QComboBox,
    QMessageBox, QHeaderView, QAbstractItemView, QFileDialog,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

from db import redis_client as db


APP_DATA_DIR = Path.home() / ".classroom_manager"
FACE_DIR = APP_DATA_DIR / "faces"


class StudentDialog(QDialog):
    def __init__(self, parent=None, data: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Học sinh" if not data else "Sửa học sinh")
        self.setMinimumWidth(380)
        self.data = data or {}

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        self.name_edit = QLineEdit(self.data.get("name", ""))
        self.code_edit = QLineEdit(self.data.get("student_code", ""))
        self.class_combo = QComboBox()
        self._load_classes()

        layout.addRow("Họ tên:", self.name_edit)
        layout.addRow("Mã học sinh:", self.code_edit)
        layout.addRow("Lớp học:", self.class_combo)

        btns = QHBoxLayout()
        self.btn_ok = QPushButton("Lưu")
        self.btn_ok.setObjectName("btn_add")
        self.btn_cancel = QPushButton("Hủy")
        self.btn_ok.clicked.connect(self._validate)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        layout.addRow(btns)

    def _load_classes(self):
        self.class_combo.addItem("-- Chưa phân lớp --", "")
        for c in db.list_classrooms():
            self.class_combo.addItem(c["name"], c["id"])
        # select current
        cid = self.data.get("class_id", "")
        for i in range(self.class_combo.count()):
            if self.class_combo.itemData(i) == cid:
                self.class_combo.setCurrentIndex(i)
                break

    def _validate(self):
        if not self.name_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập họ tên.")
            return
        if not self.code_edit.text().strip():
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập mã học sinh.")
            return
        self.accept()

    def result_data(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "student_code": self.code_edit.text().strip(),
            "class_id": self.class_combo.currentData() or "",
        }


class FaceDialog(QDialog):
    """Dialog for attaching/removing one face image for a student.

    The selected image is validated with OpenCV's built-in Haar face detector.
    When a face is found, the largest face is cropped and stored as a normalized
    JPEG. If no face is found, the user can still save the original image.
    """

    def __init__(self, student: dict, parent=None):
        super().__init__(parent)
        self.student = student
        self._selected_image: str | None = None
        self._processed_face = None
        self._saved_path: str = student.get("face_image", "") or ""
        self._removed = False

        self.setWindowTitle("Thêm khuôn mặt học sinh")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        info = QLabel(
            f"Học sinh: {student.get('student_code', '')} - {student.get('name', '')}\n"
            "Chọn ảnh rõ mặt, nhìn thẳng để hệ thống crop và lưu làm ảnh nhận diện."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.preview = QLabel("Chưa có ảnh khuôn mặt")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(360, 260)
        self.preview.setStyleSheet(
            "background:#f1f5f9; border:1px dashed #94a3b8; border-radius:12px; color:#64748b;"
        )
        layout.addWidget(self.preview)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        action_row = QHBoxLayout()
        self.btn_choose = QPushButton("🖼  Chọn ảnh")
        self.btn_choose.setObjectName("btn_edit")
        self.btn_remove = QPushButton("✕  Xóa khuôn mặt")
        self.btn_remove.setObjectName("btn_del")
        self.btn_choose.clicked.connect(self._choose_image)
        self.btn_remove.clicked.connect(self._remove_face)
        action_row.addWidget(self.btn_choose)
        action_row.addWidget(self.btn_remove)
        action_row.addStretch()
        layout.addLayout(action_row)

        btns = QHBoxLayout()
        self.btn_cancel = QPushButton("Hủy")
        self.btn_save = QPushButton("💾  Lưu khuôn mặt")
        self.btn_save.setObjectName("btn_add")
        self.btn_save.clicked.connect(self._save_face)
        self.btn_cancel.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_save)
        layout.addLayout(btns)

        self._load_current_preview()

    def saved_path(self) -> str:
        return self._saved_path

    def removed(self) -> bool:
        return self._removed

    def _load_current_preview(self):
        path = self.student.get("face_image", "") or ""
        if path and Path(path).exists():
            image = cv2.imread(path)
            if image is not None:
                self._show_image(image)
                self.lbl_status.setText(f"Đang có ảnh khuôn mặt: {path}")
                return
        self.lbl_status.setText("Chưa lưu khuôn mặt cho học sinh này.")

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh khuôn mặt",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            QMessageBox.warning(self, "Không đọc được ảnh", "File ảnh không hợp lệ hoặc không thể mở.")
            return

        self._selected_image = path
        face = self._extract_largest_face(image)
        if face is None:
            reply = QMessageBox.question(
                self,
                "Không tìm thấy khuôn mặt",
                "Không phát hiện khuôn mặt rõ trong ảnh. Bạn có muốn vẫn lưu ảnh gốc không?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            face = image
            self.lbl_status.setText("Không detect được khuôn mặt; sẽ lưu ảnh gốc.")
        else:
            self.lbl_status.setText("Đã detect khuôn mặt; ảnh sẽ được crop và lưu.")

        self._processed_face = self._normalize_face_image(face)
        self._show_image(self._processed_face)
        self._removed = False

    def _remove_face(self):
        self._processed_face = None
        self._selected_image = None
        self._saved_path = ""
        self._removed = True
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Đã chọn xóa khuôn mặt")
        self.lbl_status.setText("Bấm Lưu khuôn mặt để xác nhận xóa khỏi học sinh này.")

    def _save_face(self):
        if self._removed:
            self.accept()
            return

        if self._processed_face is None:
            QMessageBox.information(self, "Chưa chọn ảnh", "Vui lòng chọn ảnh khuôn mặt trước khi lưu.")
            return

        FACE_DIR.mkdir(parents=True, exist_ok=True)
        sid = self.student.get("id", "student") or "student"
        safe_sid = "".join(ch for ch in sid if ch.isalnum() or ch in "-_") or "student"
        dest = FACE_DIR / f"{safe_sid}_{int(time.time())}.jpg"
        ok = cv2.imwrite(str(dest), self._processed_face)
        if not ok:
            QMessageBox.warning(self, "Lỗi lưu ảnh", "Không thể ghi file ảnh khuôn mặt.")
            return
        self._saved_path = str(dest)
        self._removed = False
        self.accept()

    @staticmethod
    def _extract_largest_face(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        margin_x = int(w * 0.25)
        margin_y = int(h * 0.30)
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _normalize_face_image(image):
        if image is None or image.size == 0:
            return image
        max_side = 420
        h, w = image.shape[:2]
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return image

    def _show_image(self, image):
        if image is None:
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setText("")
        self.preview.setPixmap(pix)


class StudentsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self.load_data()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setSpacing(12)
        vlay.setContentsMargins(16, 16, 16, 16)

        # Header
        hdr = QHBoxLayout()
        title = QLabel("DANH SÁCH HỌC SINH")
        title.setObjectName("section_title")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_count = QLabel("")
        hdr.addWidget(self.lbl_count)
        vlay.addLayout(hdr)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Mã HS", "Họ tên", "Lớp", "Khuôn mặt", "ID"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setColumnHidden(4, True)
        vlay.addWidget(self.table)

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("＋  Thêm học sinh")
        self.btn_add.setObjectName("btn_add")
        self.btn_edit = QPushButton("✎  Sửa")
        self.btn_edit.setObjectName("btn_edit")
        self.btn_face = QPushButton("📷  Khuôn mặt")
        self.btn_face.setObjectName("btn_ai")
        self.btn_del = QPushButton("✕  Xóa")
        self.btn_del.setObjectName("btn_del")
        self.btn_refresh = QPushButton("↻  Làm mới")

        self.btn_add.clicked.connect(self.add_student)
        self.btn_edit.clicked.connect(self.edit_student)
        self.btn_face.clicked.connect(self.manage_face)
        self.btn_del.clicked.connect(self.delete_student)
        self.btn_refresh.clicked.connect(self.load_data)

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_face)
        btn_row.addWidget(self.btn_del)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_refresh)
        vlay.addLayout(btn_row)

    def load_data(self):
        students = db.list_students()
        classes = {c["id"]: c["name"] for c in db.list_classrooms()}
        self.table.setRowCount(len(students))
        face_count = 0
        for r, s in enumerate(students):
            self.table.setItem(r, 0, QTableWidgetItem(s.get("student_code", "")))
            self.table.setItem(r, 1, QTableWidgetItem(s.get("name", "")))
            cls_name = classes.get(s.get("class_id", ""), "—")
            self.table.setItem(r, 2, QTableWidgetItem(cls_name))

            face_path = s.get("face_image", "") or ""
            has_face = bool(face_path and Path(face_path).exists())
            if has_face:
                face_count += 1
            face_item = QTableWidgetItem("✅ Đã có" if has_face else "— Chưa có")
            if face_path:
                face_item.setToolTip(face_path)
            self.table.setItem(r, 3, face_item)
            self.table.setItem(r, 4, QTableWidgetItem(s.get("id", "")))
        self.lbl_count.setText(f"{len(students)} học sinh • {face_count} có khuôn mặt")

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 4)
        return item.text() if item else None

    def add_student(self):
        dlg = StudentDialog(self)
        if dlg.exec():
            d = dlg.result_data()
            db.create_student(d["name"], d["student_code"], d["class_id"])
            self.load_data()

    def edit_student(self):
        sid = self._selected_id()
        if not sid:
            QMessageBox.information(self, "Chọn học sinh", "Vui lòng chọn học sinh cần sửa.")
            return
        current = db.get_student(sid)
        dlg = StudentDialog(self, current)
        if dlg.exec():
            d = dlg.result_data()
            db.update_student(sid, d["name"], d["student_code"], d["class_id"])
            self.load_data()

    def manage_face(self):
        sid = self._selected_id()
        if not sid:
            QMessageBox.information(self, "Chọn học sinh", "Vui lòng chọn học sinh cần thêm khuôn mặt.")
            return
        student = db.get_student(sid)
        if not student:
            QMessageBox.warning(self, "Lỗi", "Không tìm thấy thông tin học sinh.")
            return

        old_path = student.get("face_image", "") or ""
        dlg = FaceDialog(student, self)
        if not dlg.exec():
            return

        if dlg.removed():
            db.clear_student_face(sid)
            self._safe_remove_face_file(old_path)
            QMessageBox.information(self, "Đã xóa", "Đã xóa khuôn mặt của học sinh.")
        else:
            new_path = dlg.saved_path()
            db.set_student_face(sid, new_path)
            if old_path and old_path != new_path:
                self._safe_remove_face_file(old_path)
            QMessageBox.information(self, "Đã lưu", "Đã thêm/cập nhật khuôn mặt cho học sinh.")
        self.load_data()

    def delete_student(self):
        sid = self._selected_id()
        if not sid:
            QMessageBox.information(self, "Chọn học sinh", "Vui lòng chọn học sinh cần xóa.")
            return
        s = db.get_student(sid)
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Xóa học sinh '{s.get('name', '')}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            face_path = s.get("face_image", "") or ""
            db.delete_student(sid)
            self._safe_remove_face_file(face_path)
            self.load_data()

    @staticmethod
    def _safe_remove_face_file(path: str):
        try:
            if path and Path(path).exists() and FACE_DIR in Path(path).resolve().parents:
                Path(path).unlink()
        except Exception:
            pass
