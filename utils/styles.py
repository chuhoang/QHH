"""Light Tailwind-inspired stylesheet for the classroom manager."""

STYLE = """
/* ═══════════════════════ BASE ═══════════════════════ */
QWidget {
    background-color: #f8fafc;              /* slate-50 */
    color: #0f172a;                         /* slate-900 */
    font-family: 'JetBrains Mono', 'Courier New', monospace;
    font-size: 13px;
    selection-background-color: #a7f3d0;    /* emerald-200 */
    selection-color: #064e3b;
}

QMainWindow {
    background-color: #f8fafc;
}

/* ═══════════════════════ TABS ═══════════════════════ */
QTabWidget::pane {
    border: 1px solid #e2e8f0;              /* slate-200 */
    background: #ffffff;
    border-radius: 12px;
    top: -1px;
}

QTabBar::tab {
    background: #f1f5f9;                    /* slate-100 */
    color: #64748b;                         /* slate-500 */
    padding: 10px 22px;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: bold;
    letter-spacing: 0.5px;
    font-size: 11px;
    text-transform: uppercase;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background: #ffffff;
    color: #059669;                         /* emerald-600 */
    border-top: 3px solid #10b981;          /* emerald-500 */
    border-left: 1px solid #d1fae5;
    border-right: 1px solid #d1fae5;
}

QTabBar::tab:hover:!selected {
    background: #e0f2fe;                    /* sky-100 */
    color: #0369a1;                         /* sky-700 */
}

/* ═══════════════════════ TABLES ══════════════════════ */
QTableWidget {
    background-color: #ffffff;
    color: #0f172a;
    gridline-color: #e2e8f0;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    alternate-background-color: #f8fafc;
    selection-background-color: #d1fae5;
    selection-color: #064e3b;
}

QTableWidget::item {
    padding: 7px 10px;
    border: none;
}

QTableWidget::item:selected {
    background-color: #d1fae5;              /* emerald-100 */
    color: #065f46;                         /* emerald-800 */
}

QHeaderView::section {
    background-color: #f1f5f9;
    color: #0f766e;                         /* teal-700 */
    padding: 9px 12px;
    border: none;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ═══════════════════════ INPUTS ══════════════════════ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;              /* slate-300 */
    border-radius: 10px;
    padding: 8px 12px;
    color: #0f172a;
    font-size: 13px;
}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color: #94a3b8;                  /* slate-400 */
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #10b981;
    background-color: #ecfdf5;              /* emerald-50 */
}

QComboBox::drop-down {
    border: none;
    width: 28px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #059669;
    margin-right: 8px;
}

QComboBox QAbstractItemView {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #a7f3d0;
    selection-background-color: #d1fae5;
    selection-color: #065f46;
    outline: none;
}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #f1f5f9;
    border: none;
    width: 20px;
}

/* ═══════════════════════ BUTTONS ═════════════════════ */
QPushButton {
    background-color: #ffffff;
    color: #334155;                         /* slate-700 */
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 12px;
    letter-spacing: 0.2px;
}

QPushButton:hover {
    background-color: #f8fafc;
    color: #0f172a;
    border-color: #94a3b8;
}

QPushButton:pressed {
    background-color: #e2e8f0;
}

QPushButton#btn_add {
    background-color: #ecfdf5;              /* emerald-50 */
    color: #047857;                         /* emerald-700 */
    border: 1px solid #6ee7b7;              /* emerald-300 */
}

QPushButton#btn_add:hover {
    background-color: #d1fae5;
    border-color: #10b981;
}

QPushButton#btn_edit {
    background-color: #eff6ff;              /* blue-50 */
    color: #1d4ed8;                         /* blue-700 */
    border: 1px solid #93c5fd;              /* blue-300 */
}

QPushButton#btn_edit:hover {
    background-color: #dbeafe;
    border-color: #3b82f6;
}

QPushButton#btn_del {
    background-color: #fff1f2;              /* rose-50 */
    color: #be123c;                         /* rose-700 */
    border: 1px solid #fda4af;              /* rose-300 */
}

QPushButton#btn_del:hover {
    background-color: #ffe4e6;
    border-color: #f43f5e;
}

QPushButton#btn_ai {
    background-color: #f5f3ff;              /* violet-50 */
    color: #6d28d9;                         /* violet-700 */
    border: 1px solid #c4b5fd;              /* violet-300 */
    padding: 10px 24px;
    font-size: 13px;
}

QPushButton#btn_ai:hover {
    background-color: #ede9fe;
    border-color: #8b5cf6;
}

QPushButton#btn_ai:checked {
    background-color: #7c3aed;              /* violet-600 */
    color: #ffffff;
    border-color: #6d28d9;
}

/* ═══════════════════════ LABELS ══════════════════════ */
QLabel#section_title {
    color: #047857;
    font-size: 15px;
    font-weight: bold;
    letter-spacing: 1.2px;
    padding: 4px 0;
}

QLabel#status_ok {
    color: #059669;
    font-size: 11px;
    font-weight: bold;
}

QLabel#status_err {
    color: #e11d48;
    font-size: 11px;
    font-weight: bold;
}

QLabel#cam_label {
    background: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
}

/* ═══════════════════════ GROUP BOX ═══════════════════ */
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    margin-top: 14px;
    padding-top: 12px;
    font-size: 11px;
    color: #64748b;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #0f766e;
    background-color: #ffffff;
}

/* ═══════════════════════ SCROLLBAR ═══════════════════ */
QScrollBar:vertical {
    background: #f8fafc;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #cbd5e1;
    border-radius: 5px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

/* ═══════════════════════ MISC ════════════════════════ */
QSplitter::handle {
    background: #e2e8f0;
    width: 2px;
}

QFrame#divider {
    background: #e2e8f0;
    max-height: 1px;
}

QDialog {
    background: #ffffff;
}

QMessageBox {
    background: #ffffff;
}

QStatusBar {
    background: #ffffff;
    color: #475569;
    border-top: 1px solid #e2e8f0;
}
"""
