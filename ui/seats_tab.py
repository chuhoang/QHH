"""Seat slot management tab.

A desk can have multiple physical slots. Student assignments live at slot
level; each camera has one AI region for the whole desk.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QDialog,
    QFormLayout, QComboBox, QMessageBox, QSplitter, QSpinBox, QListWidget, QHeaderView,
    QListWidgetItem, QTreeWidget, QTreeWidgetItem, QAbstractItemView,
)
from PySide6.QtCore import Qt, Signal, QMimeData
from PySide6.QtGui import QDrag, QPainter, QColor, QFont, QPen

from db import redis_client as db


ROLE_DESK = Qt.ItemDataRole.UserRole
ROLE_SLOT = Qt.ItemDataRole.UserRole + 1


class DraggableStudentList(QListWidget):
    """Student list whose items can be dragged onto a slot row."""

    def __init__(self):
        super().__init__()
        self.setDragEnabled(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    def startDrag(self, supported_actions):
        item = self.currentItem()
        if not item:
            return
        student_id = item.data(Qt.ItemDataRole.UserRole)
        if not student_id:
            return
        mime = QMimeData()
        mime.setText(item.text())
        mime.setData("application/x-student-id", str(student_id).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class SlotTree(QTreeWidget):
    student_dropped = Signal(int, int, str)  # desk_num, slot_num, student_id

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setHeaderLabels(["Bàn / Chỗ", "Học sinh", "Vùng bàn"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-student-id"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-student-id"):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if not item or not event.mimeData().hasFormat("application/x-student-id"):
            return super().dropEvent(event)
        desk = item.data(0, ROLE_DESK)
        slot = item.data(0, ROLE_SLOT)
        if desk is None or slot is None:
            return
        student_id = bytes(event.mimeData().data("application/x-student-id")).decode("utf-8")
        self.student_dropped.emit(int(desk), int(slot), student_id)
        event.acceptProposedAction()

    def selected_slot(self) -> tuple[int, int] | None:
        item = self.currentItem()
        if not item:
            return None
        desk = item.data(0, ROLE_DESK)
        slot = item.data(0, ROLE_SLOT)
        if desk is None:
            return None
        if slot is None:
            # Parent desk selected: use first slot for operations that need a slot.
            slot = 1
        return int(desk), int(slot)

    def selected_desk(self) -> int | None:
        item = self.currentItem()
        if not item:
            return None
        desk = item.data(0, ROLE_DESK)
        return int(desk) if desk is not None else None


class CapacityDialog(QDialog):
    def __init__(self, parent=None, desk_num: int = 1, current_capacity: int = 1):
        super().__init__(parent)
        self.setWindowTitle(f"Số chỗ ngồi bàn {desk_num}")
        self.setMinimumWidth(340)
        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        self.capacity_spin = QSpinBox()
        self.capacity_spin.setRange(1, 12)
        self.capacity_spin.setValue(max(1, int(current_capacity)))
        layout.addRow("Số chỗ ngồi:", self.capacity_spin)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Hủy")
        btn_ok = QPushButton("Lưu")
        btn_ok.setObjectName("btn_add")
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        layout.addRow(btns)

    def capacity(self) -> int:
        return int(self.capacity_spin.value())


class SlotAssignDialog(QDialog):
    def __init__(self, parent=None, class_id: str = "", desk_num: int = 1, slot_num: int = 1):
        super().__init__(parent)
        self.setWindowTitle(f"Gán học sinh - Bàn {desk_num}, chỗ {slot_num}")
        self.setMinimumWidth(420)
        self.class_id = class_id
        self.desk_num = desk_num
        self.slot_num = slot_num
        current = db.get_seat_slot(class_id, desk_num, slot_num)

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        self.student_combo = QComboBox()
        self.student_combo.addItem("-- Chưa gán học sinh --", "")
        for s in db.list_students(class_id):
            self.student_combo.addItem(f"{s.get('student_code','')} - {s.get('name','')}", s["id"])
        sid = current.get("student_id", "")
        for i in range(self.student_combo.count()):
            if self.student_combo.itemData(i) == sid:
                self.student_combo.setCurrentIndex(i)
                break
        layout.addRow("Học sinh:", self.student_combo)

        btns = QHBoxLayout()
        btn_cancel = QPushButton("Hủy")
        btn_ok = QPushButton("Lưu")
        btn_ok.setObjectName("btn_add")
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self.accept)
        btns.addStretch()
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)
        layout.addRow(btns)

    def selected_student_id(self) -> str:
        return self.student_combo.currentData() or ""


class SeatsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._current_class_id = ""
        self._students: dict[str, dict] = {}
        self._build_ui()
        self._load_classes()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setSpacing(12)
        vlay.setContentsMargins(16, 16, 16, 16)

        hdr = QHBoxLayout()
        title = QLabel("CẤU HÌNH CHỖ NGỒI")
        title.setObjectName("section_title")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(QLabel("Lớp học:"))
        self.class_combo = QComboBox()
        self.class_combo.setMinimumWidth(200)
        self.class_combo.currentIndexChanged.connect(self._on_class_changed)
        hdr.addWidget(self.class_combo)
        vlay.addLayout(hdr)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(8)
        left_lay.addWidget(QLabel("Học sinh trong lớp (kéo thả vào chỗ ngồi):"))
        self.student_list = DraggableStudentList()
        left_lay.addWidget(self.student_list)
        splitter.addWidget(left)

        mid = QWidget()
        mid_lay = QVBoxLayout(mid)
        mid_lay.setContentsMargins(0, 0, 0, 0)
        mid_lay.setSpacing(8)
        mid_lay.addWidget(QLabel("Danh sách chỗ ngồi nhóm theo bàn:"))
        self.slot_tree = SlotTree()
        self.slot_tree.student_dropped.connect(self.assign_student_by_drop)
        self.slot_tree.itemDoubleClicked.connect(lambda *_: self.edit_slot_student())
        mid_lay.addWidget(self.slot_tree)
        splitter.addWidget(mid)

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(8)
        right_lay.addWidget(QLabel("Sơ đồ trực quan:"))
        self.seat_map = SeatMapWidget()
        right_lay.addWidget(self.seat_map)
        splitter.addWidget(right)

        splitter.setSizes([260, 540, 360])
        vlay.addWidget(splitter)

        btn_row = QHBoxLayout()
        self.btn_capacity = QPushButton("⚙  Số chỗ / bàn")
        self.btn_capacity.setObjectName("btn_add")
        self.btn_edit = QPushButton("✎  Gán / đổi học sinh")
        self.btn_edit.setObjectName("btn_edit")
        self.btn_clear_student = QPushButton("✕  Bỏ học sinh khỏi chỗ")
        self.btn_clear_student.setObjectName("btn_del")
        self.btn_refresh = QPushButton("↻  Làm mới")

        self.btn_capacity.clicked.connect(self.configure_capacity)
        self.btn_edit.clicked.connect(self.edit_slot_student)
        self.btn_clear_student.clicked.connect(self.clear_slot_student)
        self.btn_refresh.clicked.connect(self._reload_all)

        btn_row.addWidget(self.btn_capacity)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_clear_student)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_refresh)
        vlay.addLayout(btn_row)

        hint = QLabel(
            "Quy trình: chọn lớp → cấu hình số chỗ cho từng bàn → kéo học sinh vào từng chỗ → "
            "sang màn AI Giám sát để khoanh một vùng cho toàn bộ bàn theo từng camera."
        )
        hint.setWordWrap(True)
        vlay.addWidget(hint)

    def _load_classes(self):
        current = self.class_combo.currentData()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItem("-- Chọn lớp --", "")
        for c in db.list_classrooms():
            self.class_combo.addItem(c["name"], c["id"])
        self.class_combo.blockSignals(False)
        if current:
            for i in range(self.class_combo.count()):
                if self.class_combo.itemData(i) == current:
                    self.class_combo.setCurrentIndex(i)
                    break
        self._on_class_changed()

    def _on_class_changed(self, *_):
        self._current_class_id = self.class_combo.currentData() or ""
        if self._current_class_id:
            db.ensure_classroom_seats(self._current_class_id)
        self._reload_all()

    def _reload_all(self):
        self._load_students()
        self._reload_slots()

    def _load_students(self):
        self.student_list.clear()
        cid = self._current_class_id
        self._students = {s["id"]: s for s in db.list_students(cid) if cid}
        for s in self._students.values():
            label = f"{s.get('student_code','')} - {s.get('name','')}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, s["id"])
            self.student_list.addItem(item)
        if cid and not self._students:
            self.student_list.addItem("Chưa có học sinh trong lớp này")
        elif not cid:
            self.student_list.addItem("Chọn lớp để xem học sinh")

    def _reload_slots(self):
        cid = self._current_class_id
        self.slot_tree.clear()
        if not cid:
            self.seat_map.update_seats([], 0, {})
            return

        seats = db.list_seats(cid)
        classroom = db.get_classroom(cid)
        num_desks = int(classroom.get("num_desks", 0)) if classroom else 0

        for seat in seats:
            desk_num = int(seat.get("desk_num", 0))
            slots = seat.get("slots", [])
            assigned = sum(1 for slot in slots if slot.get("student_id"))
            parent = QTreeWidgetItem([f"Bàn {desk_num} ({len(slots)} chỗ)", f"{assigned}/{len(slots)} đã gán", ""])
            parent.setData(0, ROLE_DESK, desk_num)
            parent.setFirstColumnSpanned(False)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setForeground(0, QColor("#0f172a"))
            self.slot_tree.addTopLevelItem(parent)

            for slot in slots:
                slot_num = int(slot.get("slot_num", 1))
                sid = slot.get("student_id", "")
                student = self._students.get(sid) or db.get_student(sid) if sid else {}
                student_text = f"{student.get('student_code','')} - {student.get('name','')}" if student else "-- trống --"
                zone_text = "Theo vùng chung của bàn"
                child = QTreeWidgetItem([f"  Chỗ {slot_num}", student_text, zone_text])
                child.setData(0, ROLE_DESK, desk_num)
                child.setData(0, ROLE_SLOT, slot_num)
                child.setForeground(2, QColor("#64748b"))
                parent.addChild(child)
            parent.setExpanded(True)

        self.seat_map.update_seats(seats, num_desks, self._students)

    def configure_capacity(self):
        cid = self._current_class_id
        desk = self.slot_tree.selected_desk()
        if not cid or desk is None:
            QMessageBox.information(self, "Chọn bàn", "Vui lòng chọn một bàn hoặc chỗ thuộc bàn cần cấu hình.")
            return
        seat = db.get_or_create_seat(cid, desk)
        dlg = CapacityDialog(self, desk, int(seat.get("capacity", len(seat.get("slots", [])) or 1)))
        if dlg.exec():
            db.set_seat_capacity(cid, desk, dlg.capacity())
            self._reload_slots()

    def edit_slot_student(self):
        cid = self._current_class_id
        selected = self.slot_tree.selected_slot()
        if not cid or not selected:
            QMessageBox.information(self, "Chọn chỗ", "Vui lòng chọn chỗ ngồi cần gán học sinh.")
            return
        desk, slot = selected
        dlg = SlotAssignDialog(self, cid, desk, slot)
        if dlg.exec():
            sid = dlg.selected_student_id()
            if sid:
                db.assign_student_to_slot(cid, desk, slot, sid)
            else:
                db.clear_slot_student(cid, desk, slot)
            self._reload_all()

    def assign_student_by_drop(self, desk_num: int, slot_num: int, student_id: str):
        cid = self._current_class_id
        if not cid:
            return
        student = db.get_student(student_id)
        if student.get("class_id", "") != cid:
            QMessageBox.warning(self, "Sai lớp", "Chỉ được kéo thả học sinh thuộc lớp đang chọn.")
            return
        db.assign_student_to_slot(cid, desk_num, slot_num, student_id)
        self._reload_all()

    def clear_slot_student(self):
        cid = self._current_class_id
        selected = self.slot_tree.selected_slot()
        if not cid or not selected:
            QMessageBox.information(self, "Chọn chỗ", "Vui lòng chọn chỗ cần bỏ học sinh.")
            return
        desk, slot = selected
        db.clear_slot_student(cid, desk, slot)
        self._reload_all()

    def refresh_classes(self):
        old_data = self.class_combo.currentData()
        self._load_classes()
        for i in range(self.class_combo.count()):
            if self.class_combo.itemData(i) == old_data:
                self.class_combo.setCurrentIndex(i)
                break


class SeatMapWidget(QWidget):
    """Simple visual grid showing desks and slot assignment progress."""

    def __init__(self):
        super().__init__()
        self._seats: list[dict] = []
        self._num_desks = 0
        self._students: dict = {}
        self.setMinimumSize(300, 320)

    def update_seats(self, seats: list[dict], num_desks: int, students: dict | None = None):
        self._seats = seats
        self._num_desks = num_desks
        self._students = students or {}
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))

        if self._num_desks == 0:
            p.setPen(QColor("#64748b"))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Chọn lớp để xem sơ đồ")
            return

        seats_by_desk = {int(s.get("desk_num", 0)): s for s in self._seats}
        cols = 4
        rows = (self._num_desks + cols - 1) // cols
        margin = 12
        gap = 10
        cell_w = max(70, (self.width() - 2 * margin - gap * (cols - 1)) // cols)
        cell_h = max(82, min(130, (self.height() - 2 * margin - gap * max(rows - 1, 0)) // max(rows, 1)))

        for desk in range(1, self._num_desks + 1):
            col = (desk - 1) % cols
            row = (desk - 1) // cols
            x = margin + col * (cell_w + gap)
            y = margin + row * (cell_h + gap)
            w = cell_w
            h = cell_h

            seat = seats_by_desk.get(desk, {})
            slots = seat.get("slots", []) if seat else []
            capacity = max(1, len(slots))
            assigned = sum(1 for slot in slots if slot.get("student_id"))

            p.setBrush(QColor("#f8fafc"))
            p.setPen(QPen(QColor("#cbd5e1"), 1))
            p.drawRoundedRect(x, y, w, h, 10, 10)

            p.setFont(QFont("JetBrains Mono", 9, QFont.Weight.Bold))
            p.setPen(QColor("#0f172a"))
            p.drawText(x + 8, y + 18, f"Bàn {desk}")

            p.setFont(QFont("JetBrains Mono", 8))
            p.setPen(QColor("#475569"))
            p.drawText(x + 8, y + 36, f"{assigned}/{capacity} học sinh")
            p.drawText(x + 8, y + 52, "Vùng AI theo camera")

            if slots:
                slot_w = max(12, (w - 16 - (capacity - 1) * 4) // capacity)
                sy = y + h - 22
                for idx, slot in enumerate(slots):
                    sx = x + 8 + idx * (slot_w + 4)
                    has_student = bool(slot.get("student_id"))
                    color = QColor("#bbf7d0") if has_student else QColor("#e2e8f0")
                    border = QColor("#059669") if has_student else QColor("#94a3b8")
                    p.setBrush(color)
                    p.setPen(QPen(border, 1))
                    p.drawRoundedRect(sx, sy, slot_w, 14, 4, 4)
