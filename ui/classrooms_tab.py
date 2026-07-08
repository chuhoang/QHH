"""Classrooms CRUD tab."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QPushButton, QLabel, QDialog, QFormLayout, QLineEdit, QSpinBox,
    QMessageBox, QHeaderView, QAbstractItemView,
)

from db import redis_client as db


class ClassroomDialog(QDialog):
    def __init__(self, parent=None, data: dict | None = None):
        super().__init__(parent)
        self.setWindowTitle("Thêm lớp học" if not data else "Sửa lớp học")
        self.setMinimumWidth(340)
        self.data = data or {}

        layout = QFormLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        self.name_edit = QLineEdit(self.data.get("name", ""))
        self.desks_spin = QSpinBox()
        self.desks_spin.setRange(1, 100)
        self.desks_spin.setValue(int(self.data.get("num_desks", 20)))

        layout.addRow("Tên lớp:", self.name_edit)
        layout.addRow("Số bàn:", self.desks_spin)

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

    def _validate(self):
        if not self.name_edit.text().strip():
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên lớp.")
            return
        self.accept()

    def result_data(self) -> dict:
        return {
            "name": self.name_edit.text().strip(),
            "num_desks": self.desks_spin.value(),
        }


class ClassroomsTab(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()
        self.load_data()

    def _build_ui(self):
        vlay = QVBoxLayout(self)
        vlay.setSpacing(12)
        vlay.setContentsMargins(16, 16, 16, 16)

        hdr = QHBoxLayout()
        title = QLabel("QUẢN LÝ LỚP HỌC")
        title.setObjectName("section_title")
        hdr.addWidget(title)
        hdr.addStretch()
        self.lbl_count = QLabel("")
        hdr.addWidget(self.lbl_count)
        vlay.addLayout(hdr)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Tên lớp", "Số bàn", "Học sinh", "ID"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnHidden(3, True)
        vlay.addWidget(self.table)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("＋  Thêm lớp")
        self.btn_add.setObjectName("btn_add")
        self.btn_edit = QPushButton("✎  Sửa")
        self.btn_edit.setObjectName("btn_edit")
        self.btn_del = QPushButton("✕  Xóa")
        self.btn_del.setObjectName("btn_del")
        self.btn_refresh = QPushButton("↻  Làm mới")

        self.btn_add.clicked.connect(self.add_classroom)
        self.btn_edit.clicked.connect(self.edit_classroom)
        self.btn_del.clicked.connect(self.delete_classroom)
        self.btn_refresh.clicked.connect(self.load_data)

        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_edit)
        btn_row.addWidget(self.btn_del)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_refresh)
        vlay.addLayout(btn_row)

    def load_data(self):
        classrooms = db.list_classrooms()
        students = db.list_students()
        # count students per class
        cnt: dict[str, int] = {}
        for s in students:
            cid = s.get("class_id", "")
            if cid:
                cnt[cid] = cnt.get(cid, 0) + 1

        self.table.setRowCount(len(classrooms))
        for r, c in enumerate(classrooms):
            self.table.setItem(r, 0, QTableWidgetItem(c.get("name", "")))
            self.table.setItem(r, 1, QTableWidgetItem(c.get("num_desks", "0")))
            self.table.setItem(r, 2, QTableWidgetItem(str(cnt.get(c["id"], 0))))
            self.table.setItem(r, 3, QTableWidgetItem(c.get("id", "")))
        self.lbl_count.setText(f"{len(classrooms)} lớp")

    def _selected_id(self) -> str | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        return self.table.item(row, 3).text()

    def add_classroom(self):
        dlg = ClassroomDialog(self)
        if dlg.exec():
            d = dlg.result_data()
            db.create_classroom(d["name"], d["num_desks"])
            self.load_data()

    def edit_classroom(self):
        cid = self._selected_id()
        if not cid:
            QMessageBox.information(self, "Chọn lớp", "Vui lòng chọn lớp cần sửa.")
            return
        current = db.get_classroom(cid)
        dlg = ClassroomDialog(self, current)
        if dlg.exec():
            d = dlg.result_data()
            db.update_classroom(cid, d["name"], d["num_desks"])
            self.load_data()

    def delete_classroom(self):
        cid = self._selected_id()
        if not cid:
            QMessageBox.information(self, "Chọn lớp", "Vui lòng chọn lớp cần xóa.")
            return
        c = db.get_classroom(cid)
        reply = QMessageBox.question(
            self, "Xác nhận xóa",
            f"Xóa lớp '{c.get('name', '')}'?\nCác vị trí ngồi của lớp này cũng sẽ bị xóa.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            # Remove seats of this classroom
            for seat in db.list_seats(cid):
                db.delete_seat(cid, int(seat["desk_num"]))
            db.delete_classroom(cid)
            self.load_data()
