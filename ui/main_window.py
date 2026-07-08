"""Main application window."""

from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QStatusBar, QLabel,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from db import redis_client as db
from ui.students_tab import StudentsTab
from ui.classrooms_tab import ClassroomsTab
from ui.seats_tab import SeatsTab
from ui.cameras_tab import CamerasTab
from ui.monitor_tab import MonitorTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎓 Classroom Manager – AI Seat Tracking")
        self.resize(1280, 800)
        self._build_ui()
        self._check_redis()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)

        self.students_tab = StudentsTab()
        self.classrooms_tab = ClassroomsTab()
        self.seats_tab = SeatsTab()
        self.cameras_tab = CamerasTab()
        self.monitor_tab = MonitorTab()

        self.tabs.addTab(self.students_tab,   "👤  Học sinh")
        self.tabs.addTab(self.classrooms_tab, "🏫  Lớp học")
        self.tabs.addTab(self.seats_tab,      "💺  Vị trí ngồi")
        self.tabs.addTab(self.cameras_tab,    "📷  Camera")
        self.tabs.addTab(self.monitor_tab,    "🤖  AI Giám sát")

        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_redis = QLabel()
        self.status_bar.addPermanentWidget(self.lbl_redis)

        # Periodic redis ping
        self._ping_timer = QTimer(self)
        self._ping_timer.timeout.connect(self._check_redis)
        self._ping_timer.start(10000)

    def _check_redis(self):
        ok = db.ping()
        if ok:
            self.lbl_redis.setText("⬤ Redis: Kết nối thành công")
            self.lbl_redis.setStyleSheet("color: #059669; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_redis.setText("⬤ Redis: Mất kết nối")
            self.lbl_redis.setStyleSheet("color: #e11d48; font-size: 11px; font-weight: bold;")

    def _on_tab_changed(self, idx: int):
        # Reload combos that depend on other tabs' data
        if self.tabs.widget(idx) is self.seats_tab:
            self.seats_tab.refresh_classes()
        elif self.tabs.widget(idx) is self.monitor_tab:
            self.monitor_tab.refresh()
        elif self.tabs.widget(idx) is self.students_tab:
            self.students_tab.load_data()
        elif self.tabs.widget(idx) is self.classrooms_tab:
            self.classrooms_tab.load_data()

    def closeEvent(self, event):
        self.cameras_tab.closeEvent(event)
        self.monitor_tab.closeEvent(event)
        super().closeEvent(event)
