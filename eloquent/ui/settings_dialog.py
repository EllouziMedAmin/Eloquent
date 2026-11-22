"""Eloquent settings & personal dictionary dialogs (minimalist monochrome obsidian theme)."""

import ctypes
import sys

from PySide6.QtCore import QEasingCurve, QPoint, QPointF, QRectF, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QPalette, QPen
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QKeySequenceEdit, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSizePolicy, QStackedWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from .. import APP_NAME
from ..audio import AUDIO_SOURCES, list_input_devices, list_loopback_devices
from ..config import (DEFAULT_LLM_MODEL, ISLAND_SIZES, LLM_MODELS, SPEECH_LANGUAGES, is_arabic_speech,
                      normalize_island_size, normalize_speech_language)
from ..hotkeys import parse_hotkey
from ..textutil import format_custom_vocabulary, parse_custom_vocabulary


def _hex_to_colorref(hex_color: str) -> int:
    c = QColor(hex_color)
    return (c.blue() << 16) | (c.green() << 8) | c.red()


def apply_dark_titlebar(widget: QWidget, bg_hex: str = "#0d0d10", border_hex: str = "#27272a"):
    """Apply seamless dark obsidian title bar & border on Windows 10/11 (no-op elsewhere)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(widget.winId())
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi
        dark = ctypes.c_int(1)
        # DWMWA_USE_IMMERSIVE_DARK_MODE (20 on Win11 / newer Win10, 19 on older Win10)
        if dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(dark), ctypes.sizeof(dark)) != 0:
            dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(dark), ctypes.sizeof(dark))
        # Win11 seamless obsidian caption (35) & subtle hairline border (34)
        cap = ctypes.c_int(_hex_to_colorref(bg_hex))
        dwm.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(cap), ctypes.sizeof(cap))
        bdr = ctypes.c_int(_hex_to_colorref(border_hex))
        dwm.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(bdr), ctypes.sizeof(bdr))
    except Exception:
        pass


class _ComboBox(QComboBox):
    """Minimalist QComboBox with a crisp vector monochrome dropdown chevron."""

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        col = QColor("#a1a1aa") if self.isEnabled() else QColor("#52525b")
        pen = QPen(col, 1.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        p.setPen(pen)
        cx = self.width() - 14.0
        cy = self.height() / 2.0
        p.drawPolyline([QPointF(cx - 3.5, cy - 1.5), QPointF(cx, cy + 2.0), QPointF(cx + 3.5, cy - 1.5)])


class ToggleSwitch(QCheckBox):
    """Minimalist Apple-style monochrome pill toggle switch (100% QCheckBox-compatible API)."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(36, 20)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(140)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_anim)
        self.setChecked(checked)
        self.toggled.connect(self._animate_to)

    def _on_anim(self, val):
        self._pos = float(val)
        self.update()

    def _animate_to(self, checked: bool):
        target = 1.0 if checked else 0.0
        if not self.isVisible():
            self._anim.stop()
            self._pos = target
            self.update()
            return
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(target)
        self._anim.start()

    def setChecked(self, checked: bool):
        super().setChecked(checked)
        if not self.isVisible():
            self._pos = 1.0 if checked else 0.0
            self.update()

    def hitButton(self, pos: QPoint) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = float(self.width()), float(self.height())
        r = h / 2.0
        enabled = self.isEnabled()
        hovered = self.underMouse() and enabled
        t = self._pos

        if not enabled:
            track = QColor(255, 255, 255, 14)
            border = QColor(255, 255, 255, 16)
            knob = QColor(113, 113, 122)
        else:
            # Interpolate smoothly from frosted dark zinc (off) to crisp white (on)
            tr = int(38 + (245 - 38) * t)
            tg = int(38 + (245 - 38) * t)
            tb = int(42 + (247 - 42) * t)
            if hovered and t < 0.5:
                tr, tg, tb = tr + 14, tg + 14, tb + 14
            track = QColor(tr, tg, tb)
            border = QColor(255, 255, 255, int(36 + (160 - 36) * t))
            kr = int(176 + (9 - 176) * t)
            kg = int(176 + (9 - 176) * t)
            kb = int(184 + (11 - 184) * t)
            knob = QColor(kr, kg, kb)

        p.setPen(border)
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0.5, 0.5, w - 1.0, h - 1.0), r - 0.5, r - 0.5)

        d = h - 6.0
        kx = 3.0 + t * (w - d - 6.0)
        ky = 3.0
        p.setPen(Qt.NoPen)
        p.setBrush(knob)
        p.drawEllipse(QRectF(kx, ky, d, d))


class _SwitchRow(QWidget):
    """Clickable macOS-style setting row: label + optional hint on left, ToggleSwitch on right."""

    def __init__(self, title: str, switch: ToggleSwitch, hint: str = "", parent=None):
        super().__init__(parent)
        self._switch = switch
        self.setObjectName("switchRow")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(12)

        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        self._title_lbl = _label(title, "rowTitle", wrap=True)
        col.addWidget(self._title_lbl)
        if hint:
            self._hint_lbl = _label(hint, "hint", wrap=True)
            col.addWidget(self._hint_lbl)
        row.addLayout(col, 1)
        row.addWidget(switch, 0, Qt.AlignVCenter)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton and self._switch.isEnabled() and self.isEnabled():
            if not self._switch.geometry().contains(e.position().toPoint()):
                self._switch.toggle()
        super().mouseReleaseEvent(e)


DIALOG_STYLE = """
QDialog, SettingsDialog, VocabularyDialog { background: #0d0d10; color: #f5f5f7; }
QWidget#body, QWidget#tabPage, QWidget#switchRow { background: transparent; }
QScrollArea { background: transparent; border: none; }
QLabel { color: #d4d4d8; font: 9.5pt "Segoe UI"; background: transparent; }
QLabel#title { color: #f5f5f7; font: 600 13.5pt "Segoe UI"; }
QLabel#rowTitle { color: #f5f5f7; font: 9.5pt "Segoe UI"; }
QLabel#rowTitle:disabled { color: #636366; }
QLabel#intro { color: #a1a1aa; font: 9.5pt "Segoe UI"; }
QLabel#section { color: #71717a; font: 600 8pt "Segoe UI"; letter-spacing: 1px; }
QLabel#hint { color: #8e8e93; font: 8.5pt "Segoe UI"; }
QLabel#hint:disabled { color: #52525b; }
QLabel#hint a, QLabel a { color: #d4d4d8; text-decoration: none; }
QFrame#card {
    background: #151518;
    border: 1px solid rgba(255, 255, 255, 16);
    border-radius: 12px;
}
QFrame#divider { background: rgba(255, 255, 255, 12); border: none; }
QFrame#tabBar {
    background: #151518;
    border: 1px solid rgba(255, 255, 255, 14);
    border-radius: 10px;
}
QPushButton#segTab {
    color: #8e8e93;
    background: transparent;
    border: none;
    border-radius: 7px;
    padding: 5px 10px;
    font: 500 9pt "Segoe UI";
}
QPushButton#segTab:hover { color: #e4e4e7; background: rgba(255, 255, 255, 8); }
QPushButton#segTab:checked {
    color: #ffffff;
    background: rgba(255, 255, 255, 22);
    font-weight: 600;
}
QLineEdit, QComboBox, QKeySequenceEdit {
    background: #1b1b1f;
    color: #f5f5f7;
    border: 1px solid rgba(255, 255, 255, 18);
    border-radius: 8px;
    padding: 5px 10px;
    font: 9.5pt "Segoe UI";
    selection-background-color: #3f3f46;
}
QLineEdit:focus, QComboBox:focus, QKeySequenceEdit:focus {
    border: 1px solid rgba(255, 255, 255, 72);
    background: #1f1f24;
}
QLineEdit:disabled, QComboBox:disabled, QKeySequenceEdit:disabled {
    color: #52525b;
    background: #141417;
    border-color: rgba(255, 255, 255, 8);
}
QTableWidget {
    background: #121215;
    color: #f5f5f7;
    border: 1px solid rgba(255, 255, 255, 14);
    border-radius: 8px;
    gridline-color: rgba(255, 255, 255, 10);
    font: 9.5pt "Segoe UI";
    selection-background-color: rgba(255, 255, 255, 18);
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background: #19191d;
    color: #8e8e93;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 14);
    padding: 5px 8px;
    font: 600 8.5pt "Segoe UI";
}
QComboBox { padding-right: 26px; }
QComboBox QLineEdit { background: transparent; border: none; padding: 0; color: #f5f5f7; }
QComboBox QLineEdit:focus { background: transparent; border: none; }
QComboBox QLineEdit:disabled { background: transparent; border: none; color: #52525b; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow { image: none; width: 0px; height: 0px; }
QComboBox QAbstractItemView {
    background: #16161a;
    color: #f5f5f7;
    border: 1px solid rgba(255, 255, 255, 22);
    border-radius: 8px;
    padding: 4px;
    selection-background-color: rgba(255, 255, 255, 20);
    selection-color: #ffffff;
    outline: 0;
}
QPushButton {
    color: #e4e4e7;
    background: rgba(255, 255, 255, 10);
    border: 1px solid rgba(255, 255, 255, 16);
    border-radius: 8px;
    padding: 6px 14px;
    font: 9.5pt "Segoe UI";
}
QPushButton:hover { background: rgba(255, 255, 255, 22); color: #ffffff; }
QPushButton:pressed { background: rgba(255, 255, 255, 14); }
QPushButton:disabled { color: #52525b; background: rgba(255, 255, 255, 5); border-color: rgba(255, 255, 255, 8); }
QPushButton#primary {
    background: #f5f5f7;
    border: 1px solid #f5f5f7;
    color: #09090b;
    font-weight: 600;
}
QPushButton#primary:hover { background: #ffffff; border-color: #ffffff; color: #000000; }
QPushButton#small { padding: 5px 11px; font-size: 9pt; }
QPushButton#rowDel {
    color: #8e8e93;
    background: transparent;
    border: none;
    border-radius: 6px;
    padding: 2px;
    font: 9pt "Segoe UI Symbol";
}
QPushButton#rowDel:hover { background: rgba(255, 255, 255, 16); color: #ffffff; }
QScrollBar:vertical { background: transparent; width: 6px; margin: 2px 0; }
QScrollBar::handle:vertical { background: rgba(255, 255, 255, 28); border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: rgba(255, 255, 255, 48); }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QMessageBox { background: #0d0d10; color: #f5f5f7; }
QMessageBox QLabel { color: #f5f5f7; }
"""


def _label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    pal = lb.palette()
    pal.setColor(QPalette.Link, QColor("#d4d4d8"))
    pal.setColor(QPalette.LinkVisited, QColor("#d4d4d8"))
    lb.setPalette(pal)
    return lb


def _divider() -> QFrame:
    d = QFrame(objectName="divider")
    d.setFixedHeight(1)
    return d


def _show_warning(parent: QWidget, text: str):
    box = QMessageBox(QMessageBox.Warning, APP_NAME, text, QMessageBox.Ok, parent)
    box.setStyleSheet(DIALOG_STYLE)
    box.setWindowFlag(Qt.WindowStaysOnTopHint, True)
    apply_dark_titlebar(box)
    box.exec()


class VocabularyDialog(QDialog):
    """Guided editor with separate Names & Terms and Misheard -> Corrected lists."""

    def __init__(self, vocab_str: str = "", parent=None):
        super().__init__(parent)
        self._styled_chrome = False
        self.setWindowTitle(f"{APP_NAME} - Personal Dictionary")
        self.setMinimumSize(400, 380)
        self.resize(480, 490)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setStyleSheet(DIALOG_STYLE)

        terms, pairs = parse_custom_vocabulary(vocab_str)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        root.addWidget(_label("Personal Dictionary", "title"))
        root.addWidget(_label(
            "Items below are only passed to the AI when your speech sounds similar, so normal sentences stay untouched.",
            "hint", wrap=True,
        ))

        # --- Section 1: Names & Terms ---
        root.addWidget(_label("NAMES & TERMS", "section"))
        names_card = QFrame(objectName="card")
        nc_lay = QVBoxLayout(names_card)
        nc_lay.setContentsMargins(12, 10, 12, 10)
        nc_lay.setSpacing(8)
        nc_lay.addWidget(_label(
            "Add your name, brands, or domain terms so they are spelled cleanly whenever mentioned.",
            "hint", wrap=True,
        ))

        n_row = QHBoxLayout()
        n_row.setSpacing(6)
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Name or term (e.g. John, Axbuilt, Kubernetes)")
        self.name_input.returnPressed.connect(self.add_name)
        self.btn_add_name = QPushButton("Add")
        self.btn_add_name.setObjectName("small")
        self.btn_add_name.setAutoDefault(False)
        self.btn_add_name.clicked.connect(self.add_name)
        n_row.addWidget(self.name_input, 1)
        n_row.addWidget(self.btn_add_name)
        nc_lay.addLayout(n_row)

        self.names_table = QTableWidget(0, 2)
        self.names_table.setHorizontalHeaderLabels(["Name / Term", ""])
        self.names_table.verticalHeader().setVisible(False)
        self.names_table.setSelectionMode(QAbstractItemView.NoSelection)
        nh = self.names_table.horizontalHeader()
        nh.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        nh.setSectionResizeMode(0, QHeaderView.Stretch)
        nh.setSectionResizeMode(1, QHeaderView.Fixed)
        self.names_table.setColumnWidth(1, 36)
        self.names_table.setMinimumHeight(96)
        nc_lay.addWidget(self.names_table)
        root.addWidget(names_card, 1)

        # --- Section 2: Misheard -> Corrected ---
        root.addWidget(_label("WORD CORRECTIONS", "section"))
        pairs_card = QFrame(objectName="card")
        pc_lay = QVBoxLayout(pairs_card)
        pc_lay.setContentsMargins(12, 10, 12, 10)
        pc_lay.setSpacing(8)
        pc_lay.addWidget(_label(
            "Map recurring ASR mistakes to their right spelling (right-clicking a mistake in the transcript also adds here, and rules work offline too).",
            "hint", wrap=True,
        ))

        p_row = QHBoxLayout()
        p_row.setSpacing(6)
        self.wrong_input = QLineEdit()
        self.wrong_input.setPlaceholderText("Misheard (e.g. axe built)")
        self.wrong_input.returnPressed.connect(lambda: self.right_input.setFocus())
        self.right_input = QLineEdit()
        self.right_input.setPlaceholderText("Corrected (e.g. Axbuilt)")
        self.right_input.returnPressed.connect(self.add_pair)
        self.btn_add_pair = QPushButton("Add")
        self.btn_add_pair.setObjectName("small")
        self.btn_add_pair.setAutoDefault(False)
        self.btn_add_pair.clicked.connect(self.add_pair)
        p_row.addWidget(self.wrong_input, 1)
        p_row.addWidget(_label("→", "hint"))
        p_row.addWidget(self.right_input, 1)
        p_row.addWidget(self.btn_add_pair)
        pc_lay.addLayout(p_row)

        self.pairs_table = QTableWidget(0, 3)
        self.pairs_table.setHorizontalHeaderLabels(["Misheard", "Corrected", ""])
        self.pairs_table.verticalHeader().setVisible(False)
        self.pairs_table.setSelectionMode(QAbstractItemView.NoSelection)
        ph = self.pairs_table.horizontalHeader()
        ph.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        ph.setSectionResizeMode(0, QHeaderView.Stretch)
        ph.setSectionResizeMode(1, QHeaderView.Stretch)
        ph.setSectionResizeMode(2, QHeaderView.Fixed)
        self.pairs_table.setColumnWidth(2, 36)
        self.pairs_table.setMinimumHeight(106)
        pc_lay.addWidget(self.pairs_table)
        root.addWidget(pairs_card, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        done = QPushButton("Done")
        done.setObjectName("primary")
        done.setAutoDefault(False)
        done.setDefault(False)
        done.clicked.connect(self.accept)
        footer.addWidget(cancel)
        footer.addWidget(done)
        root.addLayout(footer)

        for t in terms:
            self._insert_name_row(t)
        for w, r in pairs:
            self._insert_pair_row(w, r)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._styled_chrome:
            self._styled_chrome = True
            apply_dark_titlebar(self)

    def _make_delete_btn(self, table: QTableWidget, del_col: int) -> QPushButton:
        btn = QPushButton("✕")
        btn.setObjectName("rowDel")
        btn.setAutoDefault(False)
        btn.setFixedSize(26, 26)
        btn.setToolTip("Remove")
        btn.clicked.connect(lambda _c=False, b=btn, tbl=table, col=del_col: self._remove_row_for_btn(tbl, b, col))
        return btn

    @staticmethod
    def _remove_row_for_btn(table: QTableWidget, btn: QPushButton, del_col: int):
        for r in range(table.rowCount()):
            if table.cellWidget(r, del_col) is btn:
                table.removeRow(r)
                return

    def _insert_name_row(self, text: str):
        val = (text or "").strip()
        if not val:
            return
        for r in range(self.names_table.rowCount()):
            item = self.names_table.item(r, 0)
            if item and item.text().strip().lower() == val.lower():
                return
        row = self.names_table.rowCount()
        self.names_table.insertRow(row)
        self.names_table.setItem(row, 0, QTableWidgetItem(val))
        self.names_table.setCellWidget(row, 1, self._make_delete_btn(self.names_table, 1))
        self.names_table.setRowHeight(row, 30)

    def _insert_pair_row(self, wrong: str, right: str):
        w, r_txt = (wrong or "").strip(), (right or "").strip()
        if not w or not r_txt or w.lower() == r_txt.lower():
            return
        for r in range(self.pairs_table.rowCount()):
            it_w = self.pairs_table.item(r, 0)
            if it_w and it_w.text().strip().lower() == w.lower():
                self.pairs_table.item(r, 0).setText(w)
                self.pairs_table.item(r, 1).setText(r_txt)
                return
        row = self.pairs_table.rowCount()
        self.pairs_table.insertRow(row)
        self.pairs_table.setItem(row, 0, QTableWidgetItem(w))
        self.pairs_table.setItem(row, 1, QTableWidgetItem(r_txt))
        self.pairs_table.setCellWidget(row, 2, self._make_delete_btn(self.pairs_table, 2))
        self.pairs_table.setRowHeight(row, 30)

    def add_name(self, _checked: bool = False):
        raw = self.name_input.text().strip()
        if not raw:
            return
        terms, pairs = parse_custom_vocabulary(raw)
        for t in terms:
            self._insert_name_row(t)
        for w, r in pairs:
            self._insert_pair_row(w, r)
        self.name_input.clear()

    def add_pair(self, _checked: bool = False):
        w = self.wrong_input.text().strip()
        r = self.right_input.text().strip()
        if not w or not r or w.lower() == r.lower():
            return
        self._insert_pair_row(w, r)
        self.wrong_input.clear()
        self.right_input.clear()
        self.wrong_input.setFocus()

    def vocabulary_string(self) -> str:
        terms = []
        for r in range(self.names_table.rowCount()):
            it = self.names_table.item(r, 0)
            if it and it.text().strip():
                terms.append(it.text().strip())
        extra_name = self.name_input.text().strip()
        if extra_name:
            extra_terms, extra_pairs = parse_custom_vocabulary(extra_name)
            terms.extend(extra_terms)
        else:
            extra_pairs = []

        pairs = list(extra_pairs)
        for r in range(self.pairs_table.rowCount()):
            it_w = self.pairs_table.item(r, 0)
            it_r = self.pairs_table.item(r, 1)
            if it_w and it_r and it_w.text().strip() and it_r.text().strip():
                pairs.append((it_w.text().strip(), it_r.text().strip()))
        ew, er = self.wrong_input.text().strip(), self.right_input.text().strip()
        if ew and er:
            pairs.append((ew, er))
        return format_custom_vocabulary(terms, pairs)


class SettingsDialog(QDialog):
    """Minimalist Apple-style monochrome tabbed settings window."""

    def __init__(self, settings, parent=None, first_run=False, speech_busy=False):
        super().__init__(parent)
        self.s = settings
        self._speech_busy = speech_busy
        self._positioned = False
        self.setWindowTitle(f"{APP_NAME} Settings")
        self.setMinimumSize(420, 380)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(DIALOG_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        root.addWidget(_label("Settings", "title"))
        if first_run:
            root.addWidget(_label(
                "Welcome to Eloquent! Speech recognition runs locally on your CPU. "
                "Optionally add a free Gemini API key to polish the transcript "
                "(fix misheard words and punctuation).", "intro", wrap=True))

        # Apple-style segmented tab bar
        tab_bar = QFrame(objectName="tabBar")
        tb_lay = QHBoxLayout(tab_bar)
        tb_lay.setContentsMargins(4, 4, 4, 4)
        tb_lay.setSpacing(4)
        self._tab_buttons: list[QPushButton] = []
        for idx, title in enumerate(("Audio && Speech", "AI && Polishing", "Shortcuts")):
            btn = QPushButton(title)
            btn.setObjectName("segTab")
            btn.setCheckable(True)
            btn.setAutoDefault(False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda _c=False, i=idx: self._select_tab(i))
            tb_lay.addWidget(btn, 1)
            self._tab_buttons.append(btn)
        root.addWidget(tab_bar)

        # Build pages (each wrapped in its own smooth scroll area for small screens)
        self._stack = QStackedWidget()
        self._stack.addWidget(self._wrap_scroll(self._page_audio()))
        self._stack.addWidget(self._wrap_scroll(self._page_ai()))
        self._stack.addWidget(self._wrap_scroll(self._page_shortcuts()))
        root.addWidget(self._stack, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        footer.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.setAutoDefault(False)
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.setDefault(True)
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        root.addLayout(footer)

        self._select_tab(1 if first_run else 0)
        self._sync_llm_enabled()

        screen = (parent.screen() if parent else None) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        max_w = max(420, avail.width() - 40) if avail else 600
        max_h = max(380, int(avail.height() * 0.78)) if avail else 560
        default_h = 580 if first_run else 560
        self.resize(min(490, max_w), max(380, min(default_h, max_h)))

    def _select_tab(self, index: int):
        self._stack.setCurrentIndex(index)
        for i, btn in enumerate(self._tab_buttons):
            btn.setChecked(i == index)

    @staticmethod
    def _wrap_scroll(page: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        return scroll

    def showEvent(self, event):
        super().showEvent(event)
        if self._positioned:
            return
        self._positioned = True
        apply_dark_titlebar(self)
        screen = self.screen() or QGuiApplication.primaryScreen()
        if not screen:
            return
        avail = screen.availableGeometry().adjusted(8, 8, -8, -8)
        fg = self.frameGeometry()
        extra_h = max(0, fg.height() - self.height())
        if fg.height() > avail.height():
            self.resize(self.width(), max(self.minimumHeight(), avail.height() - extra_h))
            fg = self.frameGeometry()
        if fg.width() > avail.width():
            self.resize(max(self.minimumWidth(), avail.width()), self.height())
            fg = self.frameGeometry()
        dx = max(0, avail.left() - fg.left()) + min(0, avail.right() - fg.right())
        dy = max(0, avail.top() - fg.top()) + min(0, avail.bottom() - fg.bottom())
        if dx or dy:
            self.move(self.pos() + QPoint(dx, dy))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _card():
        card = QFrame(objectName="card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(13, 9, 13, 9)
        lay.setSpacing(7)
        return card, lay

    @staticmethod
    def _field(title: str, widget: QWidget, hint: str = "") -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(_label(title, "rowTitle"))
        col.addWidget(widget)
        if hint:
            col.addWidget(_label(hint, "hint", wrap=True))
        return col

    def _section(self, title: str, card: QFrame) -> QVBoxLayout:
        sec = QVBoxLayout()
        sec.setSpacing(6)
        sec.addWidget(_label(title.upper(), "section"))
        sec.addWidget(card)
        return sec

    # ------------------------------------------------------------------ #
    def _page_audio(self) -> QWidget:
        s = self.s
        page = QWidget(objectName="tabPage")
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 2, 4, 2)
        pl.setSpacing(12)

        # Dynamic Island size selector (steady fixed width preset)
        card_ui, ulay = self._card()
        sz_row = QHBoxLayout()
        sz_row.setSpacing(10)
        sz_col = QVBoxLayout()
        sz_col.setSpacing(1)
        sz_col.addWidget(_label("Island size", "rowTitle"))
        sz_col.addWidget(_label("Fixed steady width while speaking (no resizing mid-sentence).", "hint", wrap=True))
        sz_row.addLayout(sz_col, 1)

        seg_box = QFrame(objectName="tabBar")
        seg_lay = QHBoxLayout(seg_box)
        seg_lay.setContentsMargins(3, 3, 3, 3)
        seg_lay.setSpacing(3)
        self._island_size = normalize_island_size(getattr(s, "island_size", "medium"))
        self._size_buttons: dict[str, QPushButton] = {}
        for key, label in ISLAND_SIZES:
            b = QPushButton(label)
            b.setObjectName("segTab")
            b.setCheckable(True)
            b.setAutoDefault(False)
            b.setCursor(Qt.PointingHandCursor)
            b.setChecked(key == self._island_size)
            b.clicked.connect(lambda _c=False, k=key: self._select_island_size(k))
            seg_lay.addWidget(b)
            self._size_buttons[key] = b
        sz_row.addWidget(seg_box, 0, Qt.AlignVCenter)
        ulay.addLayout(sz_row)
        pl.addLayout(self._section("Dynamic Island", card_ui))

        # Speech & capture card
        card_cap, lay = self._card()
        self.speech_lang = _ComboBox()
        for key, label in SPEECH_LANGUAGES:
            self.speech_lang.addItem(label, key)
        cur_lang = normalize_speech_language(getattr(s, "speech_language", "parakeet"))
        self.speech_lang.setCurrentIndex(max(0, self.speech_lang.findData(cur_lang)))
        self.speech_lang.setEnabled(not self._speech_busy)
        self.speech_lang.currentIndexChanged.connect(self._sync_speech_language)
        self.speech_lang_hint = _label("", "hint", wrap=True)
        lay.addLayout(self._field("Default speech model", self.speech_lang))
        lay.addWidget(self.speech_lang_hint)
        lay.addWidget(_divider())

        self.source = _ComboBox()
        for key, label in AUDIO_SOURCES:
            self.source.addItem(label, key)
        self.source.setCurrentIndex(max(0, self.source.findData(s.audio_source)))
        self.source.currentIndexChanged.connect(self._sync_audio_source)
        self.source_hint = _label("", "hint", wrap=True)
        lay.addLayout(self._field("Audio source", self.source))
        lay.addWidget(self.source_hint)

        self.mic = _ComboBox()
        self.mic.addItem("System default", None)
        for i, name in list_input_devices():
            self.mic.addItem(name, i)
        self.mic.setCurrentIndex(max(0, self.mic.findData(s.device)))
        lay.addLayout(self._field("Microphone", self.mic))

        self.loopback = _ComboBox()
        self.loopback.addItem("Default playback speakers", None)
        for _, name, is_def in list_loopback_devices():
            self.loopback.addItem(f"{name} (default)" if is_def else name, name)
        if s.loopback_device and self.loopback.findData(s.loopback_device) < 0:
            self.loopback.addItem(s.loopback_device, s.loopback_device)
        self.loopback.setCurrentIndex(max(0, self.loopback.findData(s.loopback_device)))
        lay.addLayout(self._field("System output (loopback)", self.loopback))
        pl.addLayout(self._section("Speech & audio input", card_cap))

        # Speakers & storage card
        card_spk, slay = self._card()
        self.diarize = ToggleSwitch(s.diarize)
        slay.addWidget(_SwitchRow(
            "Identify speakers",
            self.diarize,
            "Labels Speaker 1, Speaker 2, … on 1 background thread without slowing live captions.",
        ))

        self.max_speakers = _ComboBox()
        max_opts = [(0, "Auto (detect automatically)")] + [(k, f"{k} speakers") for k in (2, 3, 4, 5, 6, 8)]
        cur_max = int(getattr(s, "max_speakers", 0) or 0)
        if cur_max >= 2 and all(v != cur_max for v, _ in max_opts):
            max_opts.append((cur_max, f"{cur_max} speakers"))
            max_opts.sort(key=lambda kv: kv[0])
        for val, lbl in max_opts:
            self.max_speakers.addItem(lbl, val)
        idx_max = self.max_speakers.findData(cur_max if cur_max >= 2 else 0)
        self.max_speakers.setCurrentIndex(max(0, idx_max))
        self.max_speakers.setEnabled(s.diarize)
        self.diarize.toggled.connect(self.max_speakers.setEnabled)
        slay.addLayout(self._field(
            "Maximum speakers",
            self.max_speakers,
            "Leave on Auto for unknown groups, or pick a fixed limit for noisy debates. Applies live.",
        ))
        slay.addWidget(_divider())

        self.audio = ToggleSwitch(s.save_audio)
        slay.addWidget(_SwitchRow(
            "Save audio recording",
            self.audio,
            "Saves audio.wav in the session folder (~115 MB per hour).",
        ))
        slay.addWidget(_divider())

        row = QHBoxLayout()
        row.setSpacing(6)
        self.folder = QLineEdit(str(s.resolved_sessions_dir()))
        browse = QPushButton("Browse…")
        browse.setObjectName("small")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        f_col = QVBoxLayout()
        f_col.setSpacing(4)
        f_col.addWidget(_label("Sessions folder", "rowTitle"))
        f_col.addLayout(row)
        slay.addLayout(f_col)
        pl.addLayout(self._section("Speakers & storage", card_spk))

        pl.addStretch(1)
        self._sync_speech_language()
        self._sync_audio_source()
        return page

    def _page_ai(self) -> QWidget:
        s = self.s
        page = QWidget(objectName="tabPage")
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 2, 4, 2)
        pl.setSpacing(12)

        card, lay = self._card()
        self.cleanup = ToggleSwitch(s.cleanup)
        self.cleanup.toggled.connect(self._sync_llm_enabled)
        lay.addWidget(_SwitchRow(
            "Polish transcript with AI",
            self.cleanup,
            "Fixes punctuation, grammar, and misheard words in real time.",
        ))
        lay.addWidget(_divider())

        key_row = QHBoxLayout()
        key_row.setSpacing(6)
        self.key = QLineEdit(s.gemini_api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your key (or leave empty to use .env)")
        self.btn_show = QPushButton("Show")
        self.btn_show.setObjectName("small")
        self.btn_show.setAutoDefault(False)
        self.btn_show.setCheckable(True)
        self.btn_show.toggled.connect(self._toggle_key)
        key_row.addWidget(self.key, 1)
        key_row.addWidget(self.btn_show)
        key_box = QWidget()
        key_col = QVBoxLayout(key_box)
        key_col.setContentsMargins(0, 0, 0, 0)
        key_col.setSpacing(4)
        key_col.addWidget(_label("Gemini API key", "rowTitle"))
        key_col.addLayout(key_row)
        link = _label(
            '<a href="https://aistudio.google.com/apikey" style="text-decoration:none">'
            '<span style="color:#d4d4d8">Get a free key from Google AI Studio ↗</span></a>',
            "hint",
        )
        link.setOpenExternalLinks(True)
        key_col.addWidget(link)
        self.key_box = key_box
        lay.addWidget(key_box)

        self.model = _ComboBox()
        self.model.setEditable(True)
        self.model.setInsertPolicy(QComboBox.NoInsert)
        self._model_label_to_id: dict[str, str] = {}
        for mid, mlabel in LLM_MODELS:
            self.model.addItem(mlabel, mid)
            self._model_label_to_id[mlabel] = mid
        cur_m = (s.llm_model or DEFAULT_LLM_MODEL).strip()
        idx_m = self.model.findData(cur_m)
        if idx_m >= 0:
            self.model.setCurrentIndex(idx_m)
        else:
            self.model.setEditText(cur_m)
        self.model.currentTextChanged.connect(self._sync_model_hint)
        self.model_hint = _label("", "hint", wrap=True)
        self.model_box = QWidget()
        mb = QVBoxLayout(self.model_box)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.setSpacing(4)
        mb.addLayout(self._field("Model", self.model))
        mb.addWidget(self.model_hint)
        lay.addWidget(self.model_box)
        self._sync_model_hint()
        lay.addWidget(_divider())

        self.fillers = ToggleSwitch(s.keep_fillers)
        self._fillers_row = _SwitchRow(
            "Keep filler words",
            self.fillers,
            "Preserve words like \"uh\" and \"um\" during AI cleanup.",
        )
        lay.addWidget(self._fillers_row)
        pl.addLayout(self._section("Gemini & polishing", card))

        # Personal dictionary card
        v_card, v_lay = self._card()
        self._vocab_value = getattr(s, "custom_vocabulary", "")
        self.vocab_summary = _label("", "hint", wrap=True)
        self.btn_vocab = QPushButton("Manage…")
        self.btn_vocab.setObjectName("small")
        self.btn_vocab.setAutoDefault(False)
        self.btn_vocab.clicked.connect(self._open_vocab_dialog)
        v_row = QHBoxLayout()
        v_col = QVBoxLayout()
        v_col.setSpacing(2)
        v_col.addWidget(_label("Personal dictionary", "rowTitle"))
        v_col.addWidget(self.vocab_summary)
        v_row.addLayout(v_col, 1)
        v_row.addWidget(self.btn_vocab, 0, Qt.AlignVCenter)
        v_lay.addLayout(v_row)
        self._sync_vocab_summary()
        pl.addLayout(self._section("Vocabulary & corrections", v_card))

        pl.addStretch(1)
        return page

    @staticmethod
    def _hotkey_row(title: str, value: str, hint: str) -> tuple[QKeySequenceEdit, QWidget]:
        edit = QKeySequenceEdit()
        edit.setMaximumSequenceLength(1)
        if value:
            edit.setKeySequence(QKeySequence(value))
        row = QHBoxLayout()
        row.setSpacing(6)
        clear = QPushButton("Clear")
        clear.setObjectName("small")
        clear.setAutoDefault(False)
        clear.clicked.connect(edit.clear)
        row.addWidget(edit, 1)
        row.addWidget(clear)
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(4)
        col.addWidget(_label(title, "rowTitle"))
        col.addLayout(row)
        col.addWidget(_label(hint, "hint", wrap=True))
        return edit, box

    def _page_shortcuts(self) -> QWidget:
        page = QWidget(objectName="tabPage")
        pl = QVBoxLayout(page)
        pl.setContentsMargins(0, 2, 4, 2)
        pl.setSpacing(12)

        hk_card, lay = self._card()
        self.hotkey, rec_box = self._hotkey_row(
            "Start / stop recording (notes & meetings)",
            self.s.hotkey,
            "Click and press keys (e.g. Ctrl+Shift+Space). Leave empty to disable.",
        )
        lay.addWidget(rec_box)
        lay.addWidget(_divider())

        self.pause_hotkey, pause_box = self._hotkey_row(
            "Pause / resume recording (meetings)",
            getattr(self.s, "pause_hotkey", "ctrl+shift+p"),
            "Pause during breaks and resume into the same session & audio file.",
        )
        lay.addWidget(pause_box)
        lay.addWidget(_divider())

        self.dictate_hotkey, dict_box = self._hotkey_row(
            "Dictate & paste at cursor",
            self.s.dictate_hotkey,
            "Press once in any text box to talk, press again to clean and paste.",
        )
        lay.addWidget(dict_box)
        lay.addWidget(_divider())

        self.compose_hotkey, comp_box = self._hotkey_row(
            "Ask AI & paste at cursor",
            self.s.compose_hotkey,
            "Speak a prompt in any app, then press again to generate and paste.",
        )
        if hasattr(self.compose_hotkey, "keySequenceChanged"):
            self.compose_hotkey.keySequenceChanged.connect(self._sync_llm_enabled)
        elif hasattr(self.compose_hotkey, "textChanged"):
            self.compose_hotkey.textChanged.connect(self._sync_llm_enabled)
        lay.addWidget(comp_box)
        lay.addWidget(_divider())

        self.switch_model_hotkey, sw_box = self._hotkey_row(
            "Switch speech model (Parakeet ↔ Arabic)",
            self.s.switch_model_hotkey,
            "Toggle live without changing your startup default.",
        )
        lay.addWidget(sw_box)
        lay.addWidget(_label(
            f"Launch Eloquent from anywhere: {self.s.launch_hotkey.title()} (configured by setup.bat).",
            "hint", wrap=True,
        ))
        pl.addLayout(self._section("Global hotkeys", hk_card))

        # Dictation & Ask AI behavior card
        beh_card, blay = self._card()
        self.dictation_cleanup = ToggleSwitch(self.s.dictation_cleanup)
        self.dictation_cleanup.toggled.connect(self._sync_llm_enabled)
        blay.addWidget(_SwitchRow("Clean dictated text with AI", self.dictation_cleanup))
        blay.addWidget(_divider())

        self.compose_selection = ToggleSwitch(self.s.compose_selection)
        blay.addWidget(_SwitchRow("Read highlighted text for Ask AI", self.compose_selection))
        blay.addWidget(_divider())

        self.restore_clipboard = ToggleSwitch(self.s.restore_clipboard)
        blay.addWidget(_SwitchRow("Restore previous clipboard after pasting", self.restore_clipboard))
        blay.addWidget(_divider())

        self.dictation_sound = ToggleSwitch(self.s.dictation_sound)
        blay.addWidget(_SwitchRow("Play soft sound cues", self.dictation_sound))
        pl.addLayout(self._section("Dictation & Ask AI", beh_card))

        pl.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    def _select_island_size(self, key: str):
        self._island_size = normalize_island_size(key)
        for k, btn in self._size_buttons.items():
            btn.setChecked(k == self._island_size)

    def selected_island_size(self) -> str:
        return self._island_size

    def _sync_speech_language(self):
        lang = self.speech_lang.currentData() or "parakeet"
        sw = (self._hotkey_text(self.switch_model_hotkey) if hasattr(self, "switch_model_hotkey")
              else self.s.switch_model_hotkey) or "the menu"
        if self._speech_busy:
            self.speech_lang_hint.setText("Stop the active recording before changing the speech model.")
        elif is_arabic_speech(lang):
            self.speech_lang_hint.setText(
                f"Starts Eloquent in Arabic (+ FR/EN cleanup with Gemini). Switch anytime with {sw}."
            )
        else:
            self.speech_lang_hint.setText(
                f"Default model covering 25 European languages (fastest on CPU, word-level speaker splitting). Switch anytime with {sw}."
            )

    def _sync_audio_source(self):
        src = self.source.currentData()
        self.mic.setEnabled(src in ("mic", "both"))
        self.loopback.setEnabled(src in ("system", "both"))
        hints = {
            "mic": "Captures your microphone only (best for dictation and in-person rooms).",
            "system": "Captures Windows audio directly with zero speaker echo (best for YouTube, podcasts, videos).",
            "both": "Labels your mic as You and remote audio separately. Wear headphones so speaker sound doesn't echo into your mic.",
        }
        self.source_hint.setText(hints.get(src, ""))

    def set_vocabulary(self, vocab_str: str):
        self._vocab_value = (vocab_str or "").strip()
        self._sync_vocab_summary()

    def vocabulary(self) -> str:
        return self._vocab_value

    def _sync_vocab_summary(self):
        terms, pairs = parse_custom_vocabulary(self._vocab_value)
        if not terms and not pairs:
            self.vocab_summary.setText("No custom names or word corrections yet.")
        else:
            parts = []
            if terms:
                parts.append(f"{len(terms)} name{'s' if len(terms) != 1 else ''}")
            if pairs:
                parts.append(f"{len(pairs)} correction{'s' if len(pairs) != 1 else ''}")
            self.vocab_summary.setText(", ".join(parts) + " saved.")

    def _open_vocab_dialog(self):
        dlg = VocabularyDialog(self._vocab_value, self)
        if dlg.exec():
            self.set_vocabulary(dlg.vocabulary_string())

    def _sync_llm_enabled(self):
        on = (
            self.cleanup.isChecked()
            or (hasattr(self, "dictation_cleanup") and self.dictation_cleanup.isChecked())
            or (hasattr(self, "compose_hotkey") and bool(self._hotkey_text(self.compose_hotkey)))
        )
        for w in (self.key_box, self.model_box, self.fillers):
            w.setEnabled(on)
        if hasattr(self, "_fillers_row"):
            self._fillers_row.setEnabled(on)

    def selected_llm_model(self) -> str:
        raw = self.model.currentText().strip()
        if not raw:
            return DEFAULT_LLM_MODEL
        return self._model_label_to_id.get(raw, raw)

    def _sync_model_hint(self):
        mid = self.selected_llm_model()
        if mid.lower() == "auto":
            self.model_hint.setText(
                "Routes live polishing, dictation, Ask AI, and reports across fast Free Tier models with automatic fallback."
            )
        else:
            self.model_hint.setText(
                f"Pinned to {mid}. Pick Auto (Smart Free Tier) for automatic routing and rate-limit fallback."
            )

    def _toggle_key(self, show: bool):
        self.key.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password)
        self.btn_show.setText("Hide" if show else "Show")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Sessions folder", self.folder.text())
        if d:
            self.folder.setText(d)

    @staticmethod
    def _hotkey_text(edit: QKeySequenceEdit) -> str:
        seq = edit.keySequence()
        if seq.isEmpty():
            return ""
        return seq.toString(QKeySequence.PortableText).lower().replace("meta", "win")

    def _save(self):
        hk = self._hotkey_text(self.hotkey)
        phk = self._hotkey_text(self.pause_hotkey)
        dhk = self._hotkey_text(self.dictate_hotkey)
        chk = self._hotkey_text(self.compose_hotkey)
        smhk = self._hotkey_text(self.switch_model_hotkey)
        for label, val in (
            ("recording hotkey", hk),
            ("pause hotkey", phk),
            ("dictation hotkey", dhk),
            ("Ask AI hotkey", chk),
            ("switch model hotkey", smhk),
        ):
            if val and parse_hotkey(val) is None:
                self._select_tab(2)
                _show_warning(
                    self,
                    f"That {label} can't be used. Include Ctrl, Alt, Shift or Win plus one key,\n"
                    "for example Ctrl+Shift+Space, Ctrl+Shift+P, Ctrl+Alt+Space, Ctrl+Alt+A, or Ctrl+Alt+M.",
                )
                return
        used = {}
        for name, val in (
            ("Recording", hk), ("Pause", phk), ("Dictation", dhk), ("Ask AI", chk), ("Switch model", smhk)
        ):
            if not val:
                continue
            k = val.lower()
            if k in used:
                self._select_tab(2)
                _show_warning(self, f"{used[k]} and {name} shortcuts must be different.")
                return
            used[k] = name
        s = self.s
        s.gemini_api_key = self.key.text().strip()
        s.llm_model = self.selected_llm_model() or s.llm_model
        s.cleanup = self.cleanup.isChecked()
        s.keep_fillers = self.fillers.isChecked()
        s.custom_vocabulary = self.vocabulary()
        s.save_audio = self.audio.isChecked()
        s.diarize = self.diarize.isChecked()
        s.max_speakers = int(self.max_speakers.currentData() or 0)
        s.speech_language = normalize_speech_language(self.speech_lang.currentData())
        s.audio_source = self.source.currentData() or "mic"
        s.device = self.mic.currentData()
        s.loopback_device = self.loopback.currentData()
        s.hotkey = hk
        s.pause_hotkey = phk
        s.dictate_hotkey = dhk
        s.compose_hotkey = chk
        s.switch_model_hotkey = smhk
        s.dictation_cleanup = self.dictation_cleanup.isChecked()
        s.compose_selection = self.compose_selection.isChecked()
        s.restore_clipboard = self.restore_clipboard.isChecked()
        s.dictation_sound = self.dictation_sound.isChecked()
        s.island_size = self.selected_island_size()
        folder = self.folder.text().strip()
        s.sessions_dir = "" if folder == str(s.resolved_sessions_dir()) and not s.sessions_dir else folder
        s.save()
        self.accept()
