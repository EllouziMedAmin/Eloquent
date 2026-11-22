"""Apple Dynamic Island floating widget UI (PySide6).

Morphing top-center obsidian pill (always on top, draggable, magnetic top-center snap):
- Idle island:      ●  Eloquent  EN/AR
- Hover bloom:      ●  Eloquent  EN/AR   ▶  ⤢  ⋯  ✕
- Dictate / Ask AI: ✦/● [mode]  live words  ▁▃▆█▅▂  ■   ->  ✓ Pasted!
- Live meeting:     ● [speaker] live caption  ▁▄█▆▃  ■  ⤢  ⋯
- Expanded card:    polished transcript (speaker pills + overlap badges), show raw, generate, copy, folder.
Tray icon and Windows global hotkeys (RegisterHotKey) without stealing foreground focus.
"""

import ctypes
import html
import math
import os
import signal
import sys
import threading
import time
import urllib.parse
from ctypes import wintypes
from pathlib import Path

if sys.platform == "win32":
    # Preload Windows System32 ICU DLLs before importing PySide6 so Anaconda's
    # incompatible ICU DLLs in Library\bin don't shadow them on Conda-based Python installs.
    _sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for _dll in ("icudt.dll", "icuuc.dll", "icuin.dll"):
        _p = os.path.join(_sys32, _dll)
        if os.path.exists(_p):
            try:
                ctypes.WinDLL(_p)
            except OSError:
                pass
    os.environ.setdefault("QT_LOGGING_RULES", "qt.gui.icc.warning=false")

from PySide6.QtCore import (QEasingCurve, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, QUrl,
                            QVariantAnimation, Signal)
from PySide6.QtGui import (QAction, QActionGroup, QColor, QCursor, QDesktopServices, QFont, QGuiApplication, QIcon,
                           QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMenu, QMessageBox, QPushButton, QSizeGrip, QSizePolicy,
                               QStackedWidget, QSystemTrayIcon, QTextBrowser, QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..audio import AUDIO_SOURCES, AUDIO_SOURCE_LABELS
from ..config import (ISLAND_SIZES, ROOT, SETTINGS_FILE, config_dir, is_arabic_speech,
                      normalize_island_size, normalize_speech_language)
from ..llm.prompts import REPORT_FORMATS
from ..paste import capture_selection_copy, get_external_foreground, play_cue, send_paste
from ..session import _short_report_tag, delete_session, list_sessions
from ..textutil import parse_custom_vocabulary

WM_HOTKEY = 0x0312
HOTKEY_ID = 1
HOTKEY_DICTATE_ID = 2
HOTKEY_COMPOSE_ID = 3
HOTKEY_SWITCH_MODEL_ID = 4

# Dynamic Island shadow margins & user-selectable steady size presets
SHADOW_PAD_X = 8
SHADOW_PAD_TOP = 4
SHADOW_PAD_BOT = 10
ISLAND_IDLE_H = 30
ISLAND_TOAST_MIN_W, ISLAND_TOAST_MAX_W, ISLAND_TOAST_H = 92, 380, 32
COMPACT_WIDTH = 520
PANEL_HEIGHT = 350
MIN_EXPANDED_W, MIN_EXPANDED_H = 340, 210

ISLAND_PRESETS = {
    "small": {
        "idle_h": 28, "hover_w": 236, "hover_h": 32,
        "compact_w": 280, "compact_h": 34,
        "meeting_w": 360, "meeting_h": 34,
        "expanded_w": 460, "expanded_h": 310,
    },
    "medium": {
        "idle_h": 30, "hover_w": 248, "hover_h": 34,
        "compact_w": 340, "compact_h": 36,
        "meeting_w": 430, "meeting_h": 36,
        "expanded_w": 520, "expanded_h": 350,
    },
    "large": {
        "idle_h": 32, "hover_w": 260, "hover_h": 36,
        "compact_w": 420, "compact_h": 38,
        "meeting_w": 540, "meeting_h": 38,
        "expanded_w": 620, "expanded_h": 400,
    },
}

STATUS_COLORS = {
    "idle": "#636366",
    "loading": "#a1a1aa",
    "listening": "#f5f5f7",
    "dictating": "#f5f5f7",
    "composing": "#f5f5f7",
    "stopping": "#a1a1aa",
    "success": "#f5f5f7",
    "error": "#ff6961",
    "overlap": "#e4e4e7",
}

STYLE = """
#card { background: transparent; border: none; }
QLabel { color: #f5f5f7; font: 9.5pt "Segoe UI"; }
#brand { color: #f5f5f7; font: 600 9pt "Segoe UI"; }
#langPill { color: #a1a1aa; background: rgba(255,255,255,16); border-radius: 6px;
            padding: 1px 5px; font: 700 7.5pt "Segoe UI"; }
#modePill { color: #e4e4e7; background: rgba(255,255,255,18); border-radius: 7px;
            padding: 1px 7px; font: 600 8pt "Segoe UI"; }
#caption { color: #f5f5f7; font: 9.5pt "Segoe UI"; }
#caption[kind="partial"] { color: #a1a1aa; font-style: italic; }
#caption[kind="hint"] { color: #a1a1aa; }
#caption[kind="success"] { color: #f5f5f7; font-weight: 600; }
#caption[kind="error"] { color: #ff8a80; }
#llm { color: #8e8e93; font-size: 8.5pt; }
QToolButton { color: #d4d4d8; background: rgba(255,255,255,10); border: none; border-radius: 11px;
              min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px;
              padding: 0; font: 8.5pt "Segoe UI Symbol"; }
QToolButton:hover { background: rgba(255,255,255,26); color: #ffffff; }
QToolButton:disabled { color: #52525b; background: rgba(255,255,255,6); }
#toggle { background: rgba(255,255,255,22); color: #ffffff; font-weight: 700; padding-left: 1px; }
#toggle:hover { background: rgba(255,255,255,40); }
#toggle[on="true"] { background: rgba(255,255,255,28); color: #ffffff; padding-left: 0px; }
#toggle[on="true"]:hover { background: rgba(255,255,255,46); color: #ffffff; }
#toggle:disabled { background: #1c1c1e; color: #52525b; }
#closeBtn { background: transparent; color: #71717a; }
#closeBtn:hover { background: rgba(255,255,255,18); color: #f5f5f7; }
QTextBrowser { background: transparent; border: none; color: #f2f2f7; font: 10pt "Segoe UI";
               selection-background-color: #3f3f46; }
QPushButton { color: #e4e4e7; background: rgba(255,255,255,12); border: 1px solid rgba(255,255,255,18);
              border-radius: 8px; padding: 3px 8px; font: 8.5pt "Segoe UI"; }
QPushButton:hover { background: rgba(255,255,255,24); color: #ffffff; }
QFrame#drawerTabs { background: rgba(255,255,255,9); border: 1px solid rgba(255,255,255,14); border-radius: 8px; }
QPushButton#drawerTab { color: #8e8e93; background: transparent; border: none; border-radius: 6px;
                        padding: 2px 9px; font: 500 8.5pt "Segoe UI"; }
QPushButton#drawerTab:hover { color: #e4e4e7; background: rgba(255,255,255,8); }
QPushButton#drawerTab:checked { color: #ffffff; background: rgba(255,255,255,22); font-weight: 600; }
QLabel#sessionTag { color: #71717a; font: 8pt "Segoe UI"; }
QPushButton#reportPill { color: #e4e4e7; background: rgba(255,255,255,12); border: 1px solid rgba(255,255,255,16);
                         border-radius: 7px; padding: 2px 8px; font: 500 8pt "Segoe UI"; }
QPushButton#reportPill:hover { background: rgba(255,255,255,24); color: #ffffff; }
QLineEdit#historySearch { background: rgba(255,255,255,9); color: #f5f5f7; border: 1px solid rgba(255,255,255,16);
                          border-radius: 8px; padding: 4px 9px; font: 9pt "Segoe UI";
                          selection-background-color: #3f3f46; }
QLineEdit#historySearch:focus { border: 1px solid rgba(255,255,255,56); background: rgba(255,255,255,13); }
QListWidget#historyList { background: transparent; border: none; outline: 0; }
QListWidget#historyList::item { background: transparent; border: none; padding: 2px 0; }
QListWidget#historyList::item:selected { background: transparent; }
QLabel#historyBanner { color: #a1a1aa; background: rgba(255,255,255,8); border-radius: 7px;
                       padding: 4px 8px; font: 8pt "Segoe UI"; }
QLabel#historyEmpty { color: #71717a; font: 9pt "Segoe UI"; }
QLabel#histCount { color: #71717a; font: 8.5pt "Segoe UI"; }
QCheckBox { color: #a1a1aa; font: 8.5pt "Segoe UI"; spacing: 5px; }
QMenu { background: #121215; color: #f5f5f7; border: 1px solid rgba(255,255,255,20);
        border-radius: 10px; padding: 5px; font: 9pt "Segoe UI"; }
QMenu::item { padding: 5px 18px 5px 12px; border-radius: 6px; }
QMenu::item:selected { background: rgba(255,255,255,18); color: #ffffff; }
QMenu::item:disabled { color: #52525b; }
QMenu::separator { height: 1px; background: rgba(255,255,255,14); margin: 4px 8px; }
QScrollBar:vertical { background: transparent; width: 6px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,30); border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: rgba(255,255,255,52); }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
from ..hotkeys import parse_hotkey, single_instance  # noqa: E402


def make_icon(color="#7c5cff") -> QIcon:
    """Return Eloquent's custom app/taskbar/tray icon, with a drawn fallback if missing."""
    for name in ("icon.ico", "icon.png"):
        p = ROOT / "assets" / name
        if p.exists():
            ic = QIcon(str(p))
            if not ic.isNull() and ic.availableSizes():
                return ic
    pm = QPixmap(64, 64)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(4, 4, 56, 56)
    p.setPen(QColor("white"))
    f = QFont("Segoe UI", 28, QFont.Bold)
    p.setFont(f)
    p.drawText(pm.rect(), Qt.AlignCenter, "E")
    p.end()
    return QIcon(pm)


SPEAKER_COLORS = ("#7db4ff", "#6ee7b7", "#fbbf24", "#f472b6", "#c084fc", "#fb923c", "#38bdf8", "#a3e635")


def _speaker_color(spk_id: str) -> str:
    idx = sum(ord(c) for c in spk_id)
    try:
        idx = int(spk_id.rsplit(" ", 1)[-1]) - 1
    except ValueError:
        pass
    return SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]


def _speaker_dot_icon(color: str) -> QIcon:
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, 8, 8)
    p.end()
    return QIcon(pm)


def _speaker_badge(spk: str, esc, spk_id: str = "", item_ref: str = "") -> str:
    if not spk:
        return ""
    key = spk_id or spk
    color = _speaker_color(key)
    href = urllib.parse.quote(key, safe="")
    target = f"spk:{href}?ref={item_ref}" if item_ref else f"spk:{href}"
    return (f'<a href="{target}" title="Click to change, rename, or merge {esc(spk)}" '
            f'style="color:{color};background:rgba(255,255,255,14);'
            f'border-radius:4px;padding:0 5px;font-size:10px;font-weight:600;'
            f'text-decoration:none">{esc(spk)}</a>&nbsp; ')


def _overlap_badge(overlap: bool) -> str:
    if not overlap:
        return ""
    return ('<span dir="ltr" title="Multiple speakers talked at the same time here" '
            'style="color:#f59e0b;background:rgba(245,158,11,36);'
            'border-radius:4px;padding:0 5px;font-size:10px;font-weight:600">'
            'overlap</span>&nbsp; ')


def transcript_html(snap: dict, show_raw: bool, rtl: bool = False) -> str:
    esc = html.escape
    multi = bool(snap.get("multi_speaker"))
    dir_attr = 'dir="rtl" align="right"' if rtl else 'dir="ltr"'
    out = []
    for p in snap["paragraphs"]:
        pid = p.get("id")
        pref = f"p:{pid}" if pid is not None else ""
        badge = _speaker_badge(p.get("speaker", ""), esc, p.get("speaker_id", ""), item_ref=pref) if multi else ""
        ov = _overlap_badge(bool(p.get("overlap")))
        t_html = f'<span dir="ltr" style="color:#7d7d7d;font-size:11px">{esc(p["time"])}</span>'
        head = f'{badge}{ov}{t_html}'
        if p["status"] == "cleaning":
            head += '<span dir="ltr" style="color:#a1a1aa;font-size:11px"> · polishing…</span>'
            body = f'<span style="color:#9a9a9a"><i>{esc(p["raw"])}</i></span>'
        elif p["tokens"] and not show_raw:
            body = " ".join(
                f'<span style="background-color:#5c4b00">{esc(w)}</span>' if f else esc(w)
                for w, f in p["tokens"])
        else:
            body = esc(p["raw"])
        out.append(f'<p style="margin:0 0 2px 0" {dir_attr}>{head}</p>')
        if p["note"]:
            out.append(f'<p style="margin:0 0 2px 0" {dir_attr}>{body}</p>')
            out.append(f'<p style="margin:0 0 12px 0;color:#e0a526;font-size:11px" dir="ltr" align="left">{esc(p["note"])}</p>')
        else:
            out.append(f'<p style="margin:0 0 12px 0" {dir_attr}>{body}</p>')
    for x in snap["loose"]:
        xid = x.get("id")
        xref = f"l:{xid}" if xid is not None else ""
        badge = _speaker_badge(x.get("speaker", ""), esc, x.get("speaker_id", ""), item_ref=xref) if multi else ""
        ov = _overlap_badge(bool(x.get("overlap")))
        t_html = f'<span dir="ltr" style="color:#7d7d7d;font-size:11px">{esc(x["t"])}</span>'
        out.append(f'<p style="margin:0 0 2px 0" {dir_attr}>{badge}{ov}{t_html}</p>'
                   f'<p style="margin:0 0 6px 0" {dir_attr}>{esc(x["text"])}</p>')
    if not out:
        out.append('<p dir="ltr" align="left" style="color:#7d7d7d">The transcript will appear here.</p>')
    return "".join(out)


DARK_DIALOG_STYLE = """
QDialog { background: #0d0d10; color: #f5f5f7; font: 9.5pt "Segoe UI"; }
QLabel { color: #f5f5f7; font: 9.5pt "Segoe UI"; }
QLabel#hint { color: #8e8e93; font: 8.5pt "Segoe UI"; }
QLineEdit { background: #1b1b1f; color: #f5f5f7; border: 1px solid rgba(255,255,255,18); border-radius: 8px;
            padding: 6px 10px; font: 9.5pt "Segoe UI"; selection-background-color: #3f3f46; }
QLineEdit:focus { border: 1px solid rgba(255,255,255,72); background: #1f1f24; }
QPushButton { background: rgba(255,255,255,10); color: #e4e4e7; border: 1px solid rgba(255,255,255,16);
              border-radius: 8px; padding: 6px 14px; font: 9.5pt "Segoe UI"; }
QPushButton:hover { background: rgba(255,255,255,22); color: #ffffff; }
QPushButton#primary { background: #f5f5f7; border-color: #f5f5f7; color: #09090b; font-weight: 600; }
QPushButton#primary:hover { background: #ffffff; border-color: #ffffff; color: #000000; }
QPushButton#subtle { background: transparent; border: none; color: #8e8e93; padding: 5px 8px; }
QPushButton#subtle:hover { color: #f5f5f7; background: rgba(255,255,255,14); }
"""


class _TextPromptDialog(QDialog):
    """Minimalist dark obsidian single-line prompt dialog (replaces legacy white QInputDialog)."""

    def __init__(self, title: str, label: str, initial: str = "", ok_label: str = "Save",
                 placeholder: str = "", parent=None):
        super().__init__(parent)
        self._styled_chrome = False
        self.setWindowTitle(title)
        self.setMinimumWidth(380)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        lbl = QLabel(label)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        self.edit = QLineEdit(initial)
        if placeholder:
            self.edit.setPlaceholderText(placeholder)
        self.edit.returnPressed.connect(self.accept)
        layout.addWidget(self.edit)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton(ok_label)
        btn_ok.setObjectName("primary")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        layout.addLayout(row)

        self.edit.setFocus()
        self.edit.selectAll()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._styled_chrome:
            self._styled_chrome = True
            apply_dark_titlebar(self)

    def value(self) -> str:
        return self.edit.text().strip()


class _ConfirmDialog(QDialog):
    """Minimalist obsidian confirmation dialog (replaces legacy white QMessageBox)."""

    def __init__(self, title: str, message: str, ok_label: str = "Delete", parent=None):
        super().__init__(parent)
        self._styled_chrome = False
        self.setWindowTitle(title)
        self.setMinimumWidth(340)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(12)

        lbl = QLabel(message)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_ok = QPushButton(ok_label)
        btn_ok.setObjectName("primary")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_ok)
        layout.addLayout(row)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._styled_chrome:
            self._styled_chrome = True
            apply_dark_titlebar(self)


class _SessionRowWidget(QFrame):
    """Clean Apple-style 2-line session card inside the History list."""

    def __init__(self, meta: dict, active: bool = False, locked: bool = False, parent=None):
        super().__init__(parent)
        self.meta = meta
        self._active = active
        self._hovered = False
        self.setObjectName("histRow")
        self.setCursor(Qt.ForbiddenCursor if locked else Qt.PointingHandCursor)
        self.setMouseTracking(True)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 6)
        lay.setSpacing(3)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(5)

        if active:
            dot = QLabel("●")
            dot.setStyleSheet("color:#f5f5f7;font:7pt 'Segoe UI';background:transparent;")
            top.addWidget(dot, 0, Qt.AlignVCenter)

        date_lbl = QLabel(meta["date_label"])
        date_lbl.setStyleSheet("color:#f5f5f7;font:600 8.5pt 'Segoe UI';background:transparent;")
        top.addWidget(date_lbl, 0, Qt.AlignVCenter)
        top.addStretch(1)

        badges = []
        if meta.get("rtl"):
            badges.append("AR")
        if meta.get("speaker_count", 0) >= 2:
            badges.append(f"{meta['speaker_count']} spk")
        if meta.get("report_tag"):
            badges.append(meta["report_tag"])
        if meta.get("has_audio"):
            badges.append("WAV")
        for btxt in badges:
            b = QLabel(btxt)
            b.setStyleSheet(
                "color:#a1a1aa;background:rgba(255,255,255,14);border-radius:5px;"
                "padding:0px 5px;font:600 7.5pt 'Segoe UI';"
            )
            top.addWidget(b, 0, Qt.AlignVCenter)
        lay.addLayout(top)

        self._raw_preview = meta.get("preview") or "(empty)"
        self._rtl = bool(meta.get("rtl"))
        self.preview_lbl = QLabel()
        self.preview_lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.preview_lbl.setMinimumWidth(0)
        self.preview_lbl.setLayoutDirection(Qt.RightToLeft if self._rtl else Qt.LeftToRight)
        self.preview_lbl.setAlignment(
            (Qt.AlignRight if self._rtl else Qt.AlignLeft) | Qt.AlignAbsolute | Qt.AlignVCenter
        )
        self.preview_lbl.setStyleSheet("color:#8e8e93;font:8.5pt 'Segoe UI';background:transparent;")
        lay.addWidget(self.preview_lbl)
        self._apply_bg()
        self._elide_preview()

    def _apply_bg(self):
        if self._hovered:
            bg = "rgba(255,255,255,18)"
            bd = "rgba(255,255,255,26)"
        elif self._active:
            bg = "rgba(255,255,255,13)"
            bd = "rgba(255,255,255,24)"
        else:
            bg = "rgba(255,255,255,6)"
            bd = "rgba(255,255,255,10)"
        self.setStyleSheet(f"QFrame#histRow {{ background:{bg}; border:1px solid {bd}; border-radius:9px; }}")

    def _elide_preview(self):
        fm = self.preview_lbl.fontMetrics()
        avail = max(80, self.preview_lbl.width() - 4)
        elided = fm.elidedText(self._raw_preview, Qt.ElideRight, avail)
        self.preview_lbl.setText(("\u200F" + elided) if (self._rtl and elided) else elided)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._elide_preview()

    def enterEvent(self, e):
        self._hovered = True
        self._apply_bg()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self._hovered = False
        self._apply_bg()
        super().leaveEvent(e)


class _RenameSpeakerDialog(QDialog):
    """Compact dark-themed single-speaker rename dialog."""

    def __init__(self, spk_id: str, current_name: str, parent=None):
        super().__init__(parent)
        self._styled_chrome = False
        self.setWindowTitle(f"Rename {spk_id}")
        self.setMinimumWidth(320)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        lbl = QLabel(f'<b style="color:{_speaker_color(spk_id)}">{html.escape(spk_id)}</b>')
        layout.addWidget(lbl)

        self.edit = QLineEdit("" if current_name == spk_id else current_name)
        self.edit.setPlaceholderText(f"Name for {spk_id}…")
        self.edit.returnPressed.connect(self.accept)
        layout.addWidget(self.edit)

        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save")
        btn_save.setObjectName("primary")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self.accept)
        row.addWidget(btn_cancel)
        row.addWidget(btn_save)
        layout.addLayout(row)

        self.edit.setFocus()
        self.edit.selectAll()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._styled_chrome:
            self._styled_chrome = True
            apply_dark_titlebar(self)

    def value(self) -> str:
        return self.edit.text().strip()


class RenameSpeakersDialog(QDialog):
    """Clean dark-themed dialog to rename all detected speakers at once."""

    def __init__(self, speakers: list[tuple[str, str]], edit_count: int = 0, parent=None):
        super().__init__(parent)
        self._styled_chrome = False
        self._special_action = ""  # "", "undo", "reset_edits"
        self.setWindowTitle(f"{APP_NAME} - Rename Speakers")
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        hint = QLabel("Assign names to speakers. Click any speaker badge in the transcript to merge or reassign.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        form.setSpacing(8)
        self.inputs: dict[str, QLineEdit] = {}
        for cid, name in speakers:
            lbl = QLabel(f'<b style="color:{_speaker_color(cid)}">{html.escape(cid)}</b>')
            edit = QLineEdit("" if name == cid else name)
            edit.setPlaceholderText(cid)
            form.addRow(lbl, edit)
            self.inputs[cid] = edit
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        self.btn_undo = None
        self.btn_reset = None
        if edit_count >= 1:
            self.btn_undo = QPushButton("Undo last" if edit_count >= 2 else "Undo merge")
            self.btn_undo.setObjectName("subtle")
            self.btn_undo.setAutoDefault(False)
            self.btn_undo.clicked.connect(lambda: self._trigger_special("undo"))
            btn_row.addWidget(self.btn_undo)
            if edit_count >= 2:
                self.btn_reset = QPushButton("Reset edits")
                self.btn_reset.setObjectName("subtle")
                self.btn_reset.setAutoDefault(False)
                self.btn_reset.clicked.connect(lambda: self._trigger_special("reset_edits"))
                btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setAutoDefault(False)
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("Save")
        btn_save.setObjectName("primary")
        btn_save.setDefault(True)
        btn_save.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        layout.addLayout(btn_row)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._styled_chrome:
            self._styled_chrome = True
            apply_dark_titlebar(self)

    def _trigger_special(self, kind: str):
        self._special_action = kind
        self.accept()

    def special_action(self) -> str:
        return self._special_action

    def values(self) -> dict[str, str]:
        return {cid: edit.text().strip() for cid, edit in self.inputs.items()}


class Bridge(QObject):
    """Carries engine events (any thread) to the GUI thread."""
    event = Signal(str, object)


class _CaptionLabel(QLabel):
    """Responsive single-line caption label that never pushes past its layout bounds."""

    def __init__(self, on_resize=None, **kwargs):
        super().__init__(**kwargs)
        self._on_resize = on_resize
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._on_resize is not None:
            self._on_resize()


class StatusOrb(QWidget):
    """Minimalist Apple Dynamic Island monochrome left indicator (orb, sparkle, checkmark, or spinner)."""

    def __init__(self):
        super().__init__()
        self.setFixedSize(16, 16)
        self.status = "idle"
        self.color = QColor(STATUS_COLORS["idle"])
        self.level = 0.0
        self._phase = 0.0

    def set_state(self, status: str, level: float = 0.0):
        self.status = status
        self.color = QColor(STATUS_COLORS.get(status, "#636366"))
        self.level = float(level)
        self.update()

    def advance(self, dt: float = 0.033):
        self._phase = (self._phase + dt * 4.5) % (2 * math.pi)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cx = self.width() / 2.0
        cy = self.height() / 2.0

        if self.status == "success":
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#f5f5f7"))
            p.drawEllipse(QRectF(cx - 6.5, cy - 6.5, 13, 13))
            pen = QPen(QColor("#09090b"), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            path = QPainterPath()
            path.moveTo(cx - 3.0, cy + 0.1)
            path.lineTo(cx - 0.7, cy + 2.4)
            path.lineTo(cx + 3.4, cy - 2.3)
            p.drawPath(path)
            return

        if self.status == "composing":
            p.setPen(Qt.NoPen)
            pulse = 1.0 + 0.10 * math.sin(self._phase * 1.4) + min(0.20, self.level * 2.5)
            r_out = 5.2 * pulse
            r_in = 1.5 * pulse
            star = QPainterPath()
            for i in range(8):
                ang = i * (math.pi / 4.0) - math.pi / 2.0
                r = r_out if i % 2 == 0 else r_in
                if i == 0:
                    star.moveTo(cx + r * math.cos(ang), cy + r * math.sin(ang))
                else:
                    star.lineTo(cx + r * math.cos(ang), cy + r * math.sin(ang))
            star.closeSubpath()
            p.setBrush(QColor("#f5f5f7"))
            p.drawPath(star)
            return

        if self.status in ("loading", "stopping"):
            ring_bg = QPen(QColor(255, 255, 255, 28), 1.8, Qt.SolidLine, Qt.RoundCap)
            p.setPen(ring_bg)
            p.setBrush(Qt.NoBrush)
            rect = QRectF(cx - 5.2, cy - 5.2, 10.4, 10.4)
            p.drawEllipse(rect)
            arc_pen = QPen(QColor("#e4e4e7"), 1.9, Qt.SolidLine, Qt.RoundCap)
            p.setPen(arc_pen)
            start_deg = int((-self._phase * 180.0 / math.pi) * 16)
            p.drawArc(rect, start_deg, 115 * 16)
            return

        # Subtle monochrome orb (with gentle white pulse while recording)
        p.setPen(Qt.NoPen)
        if self.status in ("listening", "dictating", "error"):
            r_halo = 4.2 + min(3.2, self.level * 42.0) + 0.4 * math.sin(self._phase)
            grad = QRadialGradient(cx, cy, r_halo)
            c0 = QColor(self.color)
            c0.setAlpha(75)
            grad.setColorAt(0.0, c0)
            grad.setColorAt(1.0, QColor(self.color.red(), self.color.green(), self.color.blue(), 0))
            p.setBrush(grad)
            p.drawEllipse(QRectF(cx - r_halo, cy - r_halo, r_halo * 2, r_halo * 2))

        r_core = 3.2 if self.status == "idle" else 3.6
        p.setBrush(self.color)
        p.drawEllipse(QRectF(cx - r_core, cy - r_core, r_core * 2, r_core * 2))


LevelDot = StatusOrb


class AudioWaveform(QWidget):
    """Minimalist 5-bar monochrome Apple Dynamic Island audio visualizer."""

    N_BARS = 5

    def __init__(self):
        super().__init__()
        self.setFixedSize(28, 18)
        self.color = QColor("#f5f5f7")
        self._bars = [0.20] * self.N_BARS
        self._phase = 0.0
        self._mode = "idle"
        self._level = 0.0

    def set_state(self, status: str, level: float = 0.0, overlap_glow: bool = False):
        self._mode = status
        self._level = float(level)
        self.color = QColor("#ffffff" if overlap_glow else "#e4e4e7")

    def tick(self, dt: float = 0.033):
        self._phase += dt * 7.5
        mid = (self.N_BARS - 1) / 2.0
        amp = min(1.0, (self._level * 18.0) ** 0.72) if self._level > 0.0005 else 0.0
        for i in range(self.N_BARS):
            dist = abs(i - mid) / max(1.0, mid)
            envelope = 1.0 - 0.48 * (dist * dist)
            if self._mode in ("loading", "stopping"):
                target = 0.22 + 0.34 * (0.5 + 0.5 * math.sin(self._phase * 0.75 - i * 0.75)) * envelope
            elif self._mode in ("listening", "dictating", "composing"):
                w1 = 0.55 + 0.45 * math.sin(self._phase * (1.15 + 0.14 * i) + i * 1.3)
                w2 = 0.50 + 0.50 * math.cos(self._phase * 0.7 - i * 0.9)
                floor_h = 0.18 + 0.08 * math.sin(self._phase * 0.4 + i)
                target = floor_h + amp * (0.35 + 0.65 * w1 * w2) * envelope
            else:
                target = 0.18
            target = max(0.16, min(1.0, target))
            alpha = 0.48 if target > self._bars[i] else 0.24
            self._bars[i] += (target - self._bars[i]) * alpha
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        bar_w = 2.6
        gap = 2.4
        total_w = self.N_BARS * bar_w + (self.N_BARS - 1) * gap
        x0 = (self.width() - total_w) / 2.0
        cy = self.height() / 2.0
        max_h = self.height() - 3.0
        for i, v in enumerate(self._bars):
            bh = max(2.8, v * max_h)
            x = x0 + i * (bar_w + gap)
            c = QColor(self.color)
            c.setAlpha(int(130 + 125 * v))
            p.setBrush(c)
            p.drawRoundedRect(QRectF(x, cy - bh / 2.0, bar_w, bh), bar_w / 2.0, bar_w / 2.0)


from .settings_dialog import SettingsDialog, VocabularyDialog, apply_dark_titlebar  # noqa: E402


# --------------------------------------------------------------------------- #
# main widget
# --------------------------------------------------------------------------- #
class EloquentWidget(QWidget):
    def __init__(self, engine, settings):
        super().__init__()
        self.engine, self.s = engine, settings
        self.show_raw = False
        self.view_mode = "transcript"
        self.drawer_tab = "transcript"  # "transcript" | "history"
        self._loaded_session_dir: Path | None = None
        self._loaded_session_label = ""
        self._loaded_session_rtl: bool | None = None
        self._loaded_reports: list[dict] = []
        self._scroll_to_top_once = False
        self.dirty = True
        self.drag_offset = None
        self._press_global = None
        self._drag_moved = False
        self.quitting = False
        self.hotkey_ok = False
        self.dictate_hotkey_ok = False
        self.compose_hotkey_ok = False
        self.switch_model_hotkey_ok = False
        self.dictate_target_hwnd = 0
        self._compose_starting = False
        self.caption_text, self.caption_kind = "Starting…", "hint"
        self.last_final = ""

        # Dynamic Island state
        self._island_mode = "idle"
        self._hovered = False
        self._menu_open = False
        self._toast_active = False
        self._active_speaker = ""
        self._active_speaker_id = ""
        self._overlap_flash_until = 0.0
        self._last_overlap_seen = False
        _preset = self._island_preset()
        self._expanded_w = _preset["expanded_w"]
        self._expanded_h = _preset["expanded_h"]
        self._anchor_cx = 0
        self._anchor_y = 8
        self._anim_rect_from = None
        self._anim_rect_to = None
        self._morph_anim = QVariantAnimation(self)
        self._morph_anim.setDuration(240)
        self._morph_anim.valueChanged.connect(self._on_morph_step)
        self._morph_anim.finished.connect(self._on_morph_done)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        flags = Qt.FramelessWindowHint | Qt.Window
        if settings.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setStyleSheet(STYLE)
        self._build()

        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        engine.subscribe(lambda k, d: self.bridge.event.emit(k, d))

        self.level_timer = QTimer(self, interval=33, timeout=self._tick_level)
        self.rebuild_timer = QTimer(self, singleShot=True, interval=80, timeout=self._rebuild)
        self.caption_reset = QTimer(self, singleShot=True, interval=4500, timeout=self._idle_caption)
        self._leave_timer = QTimer(self, singleShot=True, interval=450, timeout=self._on_leave_timeout)
        self._hist_search_timer = QTimer(self, singleShot=True, interval=120, timeout=self._refresh_history)

        self._place()
        self._tray()
        self._update_lang_pill()
        self._update_controls()
        self._sync_island(animate=False)

    # -- layout -------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SHADOW_PAD_X, SHADOW_PAD_TOP, SHADOW_PAD_X, SHADOW_PAD_BOT)
        outer.setSpacing(0)
        self.card = QFrame(objectName="card")
        self.card.setLayoutDirection(Qt.LeftToRight)
        outer.addWidget(self.card)
        self._card_layout = QVBoxLayout(self.card)
        self._card_layout.setContentsMargins(12, 5, 10, 5)
        self._card_layout.setSpacing(6)

        self.top_bar = QWidget()
        top = QHBoxLayout(self.top_bar)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self.dot = StatusOrb()
        self.brand_lbl = QLabel(APP_NAME, objectName="brand")
        self.lang_pill = QLabel("EN", objectName="langPill")
        self.lang_pill.setAlignment(Qt.AlignCenter)
        self.mode_pill = QLabel("", objectName="modePill")
        self.mode_pill.setAlignment(Qt.AlignCenter)
        self.mode_pill.hide()
        self.caption = _CaptionLabel(on_resize=self._render_caption, objectName="caption")
        self.waveform = AudioWaveform()
        self.waveform.hide()
        self.btn_toggle = QToolButton(objectName="toggle", text="▶")
        self.btn_toggle.clicked.connect(self._toggle_meeting)
        self.btn_expand = QToolButton(text="⤢", toolTip="Show transcript")
        self.btn_expand.clicked.connect(self.toggle_panel)
        self.btn_menu = QToolButton(text="⋯", toolTip="Menu")
        self.btn_menu.clicked.connect(self._show_menu)
        self.btn_close = QToolButton(objectName="closeBtn", text="✕", toolTip=f"Quit {APP_NAME}")
        self.btn_close.clicked.connect(self.quit)

        top.addWidget(self.dot, 0, Qt.AlignVCenter)
        top.addWidget(self.brand_lbl, 0, Qt.AlignVCenter)
        top.addWidget(self.lang_pill, 0, Qt.AlignVCenter)
        top.addWidget(self.mode_pill, 0, Qt.AlignVCenter)
        top.addWidget(self.caption, 1, Qt.AlignVCenter)
        self._idle_spacer = QWidget()
        self._idle_spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self._idle_spacer.hide()
        top.addWidget(self._idle_spacer)
        top.addWidget(self.waveform, 0, Qt.AlignVCenter)
        top.addWidget(self.btn_toggle, 0, Qt.AlignVCenter)
        top.addWidget(self.btn_expand, 0, Qt.AlignVCenter)
        top.addWidget(self.btn_menu, 0, Qt.AlignVCenter)
        top.addWidget(self.btn_close, 0, Qt.AlignVCenter)
        self._card_layout.addWidget(self.top_bar)

        self.panel = QWidget()
        pv = QVBoxLayout(self.panel)
        pv.setContentsMargins(2, 4, 2, 4)
        pv.setSpacing(6)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: rgba(255,255,255,22);")
        pv.addWidget(line)

        # Compact Apple segmented drawer bar: [ Transcript | History ]   <session tag>   [ Report → ]
        dbar = QHBoxLayout()
        dbar.setContentsMargins(0, 0, 0, 0)
        dbar.setSpacing(6)
        tabs_box = QFrame(objectName="drawerTabs")
        tb_lay = QHBoxLayout(tabs_box)
        tb_lay.setContentsMargins(2, 2, 2, 2)
        tb_lay.setSpacing(2)
        self.tab_transcript = QPushButton("Transcript")
        self.tab_transcript.setObjectName("drawerTab")
        self.tab_transcript.setCheckable(True)
        self.tab_transcript.setChecked(True)
        self.tab_transcript.setCursor(Qt.PointingHandCursor)
        self.tab_transcript.clicked.connect(lambda: self._set_drawer_tab("transcript"))
        self.tab_history = QPushButton("History")
        self.tab_history.setObjectName("drawerTab")
        self.tab_history.setCheckable(True)
        self.tab_history.setCursor(Qt.PointingHandCursor)
        self.tab_history.clicked.connect(lambda: self._set_drawer_tab("history"))
        tb_lay.addWidget(self.tab_transcript)
        tb_lay.addWidget(self.tab_history)
        dbar.addWidget(tabs_box, 0, Qt.AlignVCenter)

        self.lbl_session_tag = QLabel("", objectName="sessionTag")
        self.lbl_session_tag.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_session_tag.setMinimumWidth(0)
        dbar.addWidget(self.lbl_session_tag, 1, Qt.AlignVCenter)

        self.btn_back = QPushButton("← Transcript", objectName="reportPill")
        self.btn_back.setCursor(Qt.PointingHandCursor)
        self.btn_back.clicked.connect(self._switch_view)
        self.btn_back.hide()
        dbar.addWidget(self.btn_back, 0, Qt.AlignVCenter)
        pv.addLayout(dbar)

        # Stacked body: Page 0 = Transcript / Report viewer, Page 1 = Session History
        self.drawer_stack = QStackedWidget()
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_anchor)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_view_context_menu)
        self.view.setMinimumHeight(96)
        self._sync_view_direction()
        self.drawer_stack.addWidget(self.view)

        self.history_page = QWidget()
        hp = QVBoxLayout(self.history_page)
        hp.setContentsMargins(0, 0, 0, 0)
        hp.setSpacing(6)
        self.history_banner = QLabel(
            "Recording in progress. Stop recording to open a past session.",
            objectName="historyBanner",
        )
        self.history_banner.setWordWrap(True)
        self.history_banner.hide()
        hp.addWidget(self.history_banner)
        self.history_search = QLineEdit(objectName="historySearch")
        self.history_search.setPlaceholderText("Search past sessions, speakers, or words…")
        self.history_search.setClearButtonEnabled(True)
        self.history_search.textChanged.connect(lambda _t: self._hist_search_timer.start())
        hp.addWidget(self.history_search)
        self.history_list = QListWidget(objectName="historyList")
        self.history_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.history_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.history_list.setSpacing(2)
        self.history_list.setMinimumHeight(80)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        self.history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.history_list.customContextMenuRequested.connect(self._show_history_context_menu)
        hp.addWidget(self.history_list, 1)
        self.history_empty = QLabel("No saved sessions yet.", objectName="historyEmpty")
        self.history_empty.setAlignment(Qt.AlignCenter)
        self.history_empty.hide()
        hp.addWidget(self.history_empty, 1)
        self.drawer_stack.addWidget(self.history_page)
        pv.addWidget(self.drawer_stack, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(5)
        self.chk_raw = QCheckBox("show raw")
        self.chk_raw.toggled.connect(self._set_raw)
        self.lbl_hist_count = QLabel("", objectName="histCount")
        self.lbl_hist_count.hide()
        self.lbl_llm = QLabel(objectName="llm")
        self.lbl_llm.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_llm.setMinimumWidth(0)
        self.btn_hist_refresh = QPushButton("Refresh")
        self.btn_hist_refresh.clicked.connect(self._refresh_history)
        self.btn_hist_refresh.hide()
        self.btn_gen = QPushButton("Generate ▾")
        self.btn_gen.clicked.connect(self._show_gen_menu)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._copy)
        self.btn_folder = QPushButton("Folder")
        self.btn_folder.setToolTip("Open session folder")
        self.btn_folder.clicked.connect(self._open_folder)
        bottom.addWidget(self.chk_raw)
        bottom.addWidget(self.lbl_hist_count)
        bottom.addWidget(self.lbl_llm, 1)
        bottom.addWidget(self.btn_hist_refresh)
        bottom.addWidget(self.btn_gen)
        bottom.addWidget(self.btn_copy)
        bottom.addWidget(self.btn_folder)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        pv.addLayout(bottom)
        self.panel.hide()
        self._card_layout.addWidget(self.panel, 1)

        self.gen_menu = QMenu(self)
        self.act_rename = self.gen_menu.addAction("Rename speakers…", self.rename_speakers)
        self.act_rename_sep = self.gen_menu.addSeparator()
        for fmt in REPORT_FORMATS:
            self.gen_menu.addAction(fmt["label"], lambda fid=fmt["id"]: self._trigger_gen(fid))
        self.gen_menu.addSeparator()
        self.gen_menu.addAction("Custom Prompt…", self._trigger_custom_gen)

        self.compact_height = ISLAND_IDLE_H + SHADOW_PAD_TOP + SHADOW_PAD_BOT

        self.menu = QMenu(self)
        self.act_settings = self.menu.addAction("Settings…", self.open_settings)
        self.menu.addAction("Personal dictionary…", self.open_vocabulary)
        self.act_switch_model = self.menu.addAction("Switch speech model", self._toggle_speech_model)
        src_menu = self.menu.addMenu("Audio source")
        self.src_group = QActionGroup(self)
        self.src_actions: dict[str, QAction] = {}
        for key, label in AUDIO_SOURCES:
            act = src_menu.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda _chk=False, k=key: self._set_audio_source(k))
            self.src_group.addAction(act)
            self.src_actions[key] = act
        self.max_spk_menu = self.menu.addMenu("Max speakers")
        self.max_spk_group = QActionGroup(self)
        self.max_spk_actions: dict[int, QAction] = {}
        for cap, label in [(0, "Auto")] + [(k, f"{k} speakers") for k in (2, 3, 4, 5, 6, 8)]:
            act = self.max_spk_menu.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda _chk=False, c=cap: self._set_max_speakers(c))
            self.max_spk_group.addAction(act)
            self.max_spk_actions[cap] = act
        self.act_audio = self.menu.addAction("Save audio recording")
        self.act_audio.setCheckable(True)
        self.act_audio.toggled.connect(self._set_audio)
        self.size_menu = self.menu.addMenu("Island size")
        self.size_group = QActionGroup(self)
        self.size_actions: dict[str, QAction] = {}
        for sz_key, sz_label in ISLAND_SIZES:
            act = self.size_menu.addAction(sz_label)
            act.setCheckable(True)
            act.triggered.connect(lambda _chk=False, k=sz_key: self._set_island_size(k))
            self.size_group.addAction(act)
            self.size_actions[sz_key] = act
        self.act_top = self.menu.addAction("Always on top")
        self.act_top.setCheckable(True)
        self.act_top.toggled.connect(self._set_on_top)
        self.menu.addSeparator()
        self.menu.addAction("Open sessions folder", lambda: self._open(self.s.resolved_sessions_dir()))
        self.menu.addAction("Clear Ask AI memory", self._clear_compose_history)
        self.menu.addAction("Create shortcuts + launch hotkey", self._make_shortcut)
        self.menu.addAction("Reset island position", self._reset_island_pos)
        self.menu.addAction("Minimize", self.showMinimized)
        self.menu.addSeparator()
        self.menu.addAction(f"Quit {APP_NAME}", self.quit)

    # -- Dynamic Island geometry, morphing & painting -----------------------
    def _screen_geo(self) -> QRect:
        scr = QGuiApplication.screenAt(QPoint(int(self._anchor_cx), max(0, int(self._anchor_y + 20))))
        if scr is None:
            scr = self.screen() or QGuiApplication.primaryScreen()
        return scr.availableGeometry()

    def _place(self):
        geo = QGuiApplication.primaryScreen().availableGeometry()
        pos = self.s.widget_pos
        if (isinstance(pos, (list, tuple)) and len(pos) >= 3 and pos[2] == "island"
                and QGuiApplication.screenAt(QPoint(int(pos[0]), int(pos[1]) + 20)) is not None):
            self._anchor_cx = int(pos[0])
            self._anchor_y = int(pos[1])
        else:
            # Dock cleanly at top-center like Apple Dynamic Island
            self._anchor_cx = geo.center().x()
            self._anchor_y = geo.top() + 8

    def _reset_island_pos(self):
        scr = self.screen() or QGuiApplication.primaryScreen()
        geo = scr.availableGeometry()
        self._anchor_cx = geo.center().x()
        self._anchor_y = geo.top() + 8
        self._save_island_pos()
        self._sync_island(animate=True)

    def _target_window_rect(self, card_w: int, card_h: int) -> QRect:
        win_w = int(card_w) + 2 * SHADOW_PAD_X
        win_h = int(card_h) + SHADOW_PAD_TOP + SHADOW_PAD_BOT
        geo = self._screen_geo()
        x = int(round(self._anchor_cx - win_w / 2.0))
        y = int(round(self._anchor_y - SHADOW_PAD_TOP))
        min_x = geo.left() - SHADOW_PAD_X + 6
        max_x = geo.right() - win_w + SHADOW_PAD_X - 6
        min_y = geo.top() - SHADOW_PAD_TOP + 2
        max_y = geo.bottom() - win_h + SHADOW_PAD_BOT - 6
        x = max(min_x, min(max_x, x)) if max_x >= min_x else min_x
        y = max(min_y, min(max_y, y)) if max_y >= min_y else min_y
        return QRect(x, y, win_w, win_h)

    def _update_lang_pill(self):
        parts = ["AR" if self._is_rtl_language() else "EN"]
        if self.s.audio_source == "system":
            parts.append("SYS")
        elif self.s.audio_source == "both":
            parts.append("BOTH")
        self.lang_pill.setText(" · ".join(parts))

    def _update_mode_pill(self):
        st = self.engine.status
        mode = self.engine.session_mode
        mono_pill = (
            "color:#e4e4e7;background:rgba(255,255,255,18);border-radius:7px;"
            "padding:1px 7px;font:600 8pt 'Segoe UI';"
        )
        if st in ("listening", "stopping"):
            if mode == "compose":
                self.mode_pill.setText("Ask AI")
                self.mode_pill.setStyleSheet(mono_pill)
                return True
            if mode == "dictation":
                self.mode_pill.setText("Dictate")
                self.mode_pill.setStyleSheet(mono_pill)
                return True
            if time.monotonic() < self._overlap_flash_until:
                self.mode_pill.setText("overlap")
                self.mode_pill.setStyleSheet(
                    "color:#f5f5f7;background:rgba(255,255,255,28);border-radius:7px;"
                    "padding:1px 7px;font:600 8pt 'Segoe UI';"
                )
                return True
            if self._active_speaker:
                lbl = self._active_speaker if len(self._active_speaker) <= 14 else self._active_speaker[:12] + "…"
                self.mode_pill.setText(lbl)
                self.mode_pill.setStyleSheet(mono_pill)
                return True
        return False

    def _island_preset(self) -> dict:
        key = normalize_island_size(getattr(self.s, "island_size", "medium"))
        return ISLAND_PRESETS[key]

    def _desired_island_spec(self) -> tuple[str, int, int]:
        geo = self._screen_geo()
        scr_w, scr_h = max(800, geo.width()), max(600, geo.height())
        preset = self._island_preset()
        if self.panel.isVisible():
            ew = max(MIN_EXPANDED_W, min(self._expanded_w, int(scr_w * 0.86)))
            eh = max(MIN_EXPANDED_H, min(self._expanded_h, int(scr_h * 0.80)))
            return "expanded", ew, eh
        st = self.engine.status
        mode = self.engine.session_mode
        if st == "loading":
            return "toast", min(preset["compact_w"], int(scr_w * 0.44)), ISLAND_TOAST_H
        if st in ("listening", "stopping"):
            if mode in ("dictation", "compose"):
                return "active_compact", min(preset["compact_w"], int(scr_w * 0.50)), preset["compact_h"]
            return "active_meeting", min(preset["meeting_w"], int(scr_w * 0.62)), preset["meeting_h"]
        if self._toast_active:
            fm = self.caption.fontMetrics()
            txt_w = fm.horizontalAdvance(self.caption_text or "")
            max_w = min(ISLAND_TOAST_MAX_W, int(scr_w * 0.38))
            return "toast", max(ISLAND_TOAST_MIN_W, min(max_w, txt_w + 46)), ISLAND_TOAST_H
        if self._hovered or self._menu_open:
            lang_w = self.lang_pill.sizeHint().width()
            brand_w = self.brand_lbl.sizeHint().width()
            hov_w = max(preset["hover_w"], 16 + brand_w + lang_w + 156)
            return "hover", min(hov_w, int(scr_w * 0.36)), preset["hover_h"]
        lang_w = self.lang_pill.sizeHint().width()
        brand_w = self.brand_lbl.sizeHint().width()
        idle_w = max(122, min(192, 16 + brand_w + lang_w + 38))
        return "idle", idle_w, preset["idle_h"]

    def _sync_island(self, animate: bool = True):
        mode, card_w, card_h = self._desired_island_spec()
        prev_mode = self._island_mode
        self._island_mode = mode
        self._update_lang_pill()
        has_mode_pill = self._update_mode_pill()

        is_exp = (mode == "expanded")
        is_idle = (mode == "idle")
        is_hover = (mode == "hover")
        is_toast = (mode == "toast")
        is_compact = (mode == "active_compact")
        is_meeting = (mode == "active_meeting")
        is_busy = self.engine.status in ("listening", "stopping", "loading")

        # Configure clean tight inner padding for pill vs expanded card
        if is_exp:
            self._card_layout.setContentsMargins(12, 8, 12, 8)
        elif is_idle:
            self._card_layout.setContentsMargins(10, 3, 10, 3)
        elif is_toast:
            self._card_layout.setContentsMargins(10, 4, 12, 4)
        else:
            self._card_layout.setContentsMargins(11, 4, 8, 4)

        self.brand_lbl.setVisible(is_idle or is_hover)
        self.lang_pill.setVisible(is_idle or is_hover or (is_exp and not has_mode_pill))
        self.mode_pill.setVisible(has_mode_pill and (is_compact or is_meeting or is_exp))
        self.caption.setVisible(is_toast or is_compact or is_meeting or is_exp)
        self._idle_spacer.setVisible(is_idle or is_hover)
        self.waveform.setVisible(is_busy and (is_compact or is_meeting or is_exp or self.engine.status == "loading"))
        self.btn_toggle.setVisible(is_hover or is_compact or is_meeting or is_exp)
        self.btn_expand.setVisible(is_hover or is_meeting or is_exp)
        self.btn_menu.setVisible(is_hover or is_meeting or is_exp)
        self.btn_close.setVisible(is_hover or is_exp)
        self.btn_expand.setText("⤡" if is_exp else "⤢")

        target = self._target_window_rect(card_w, card_h)
        cur = self.geometry()
        if not animate or not self.isVisible() or cur == target:
            self._morph_anim.stop()
            self._apply_size_constraints(is_exp, target)
            self.setGeometry(target)
            self._render_caption()
            self.update()
            return

        # Smooth Apple Dynamic Island spring morph centered around anchor_cx
        self._morph_anim.stop()
        self.setMinimumSize(0, 0)
        self.setMaximumSize(16777215, 16777215)
        self._anim_rect_from = cur
        self._anim_rect_to = target
        growing = (target.width() > cur.width() and not is_exp and prev_mode != "expanded")
        curve = QEasingCurve(QEasingCurve.OutBack if growing else QEasingCurve.OutCubic)
        if growing:
            curve.setOvershoot(0.72)
        self._morph_anim.setEasingCurve(curve)
        self._morph_anim.setDuration(250 if (is_exp or prev_mode == "expanded") else 220)
        self._morph_anim.setStartValue(0.0)
        self._morph_anim.setEndValue(1.0)
        self._morph_anim.start()
        self._render_caption()

    def _apply_size_constraints(self, is_expanded: bool, target: QRect):
        if is_expanded:
            self.setMinimumSize(MIN_EXPANDED_W + 2 * SHADOW_PAD_X, MIN_EXPANDED_H + SHADOW_PAD_TOP + SHADOW_PAD_BOT)
            self.setMaximumSize(16777215, 16777215)
        else:
            self.setFixedSize(target.width(), target.height())

    def _on_morph_step(self, t: float):
        if self._anim_rect_from is None or self._anim_rect_to is None:
            return
        r0, r1 = self._anim_rect_from, self._anim_rect_to
        w = max(92, int(round(r0.width() + (r1.width() - r0.width()) * t)))
        h = max(36, int(round(r0.height() + (r1.height() - r0.height()) * t)))
        x = int(round(self._anchor_cx - w / 2.0))
        y = int(round(r0.y() + (r1.y() - r0.y()) * t))
        geo = self._screen_geo()
        min_x = geo.left() - SHADOW_PAD_X + 6
        max_x = geo.right() - w + SHADOW_PAD_X - 6
        if max_x >= min_x:
            x = max(min_x, min(max_x, x))
        self.setGeometry(x, y, w, h)
        self.update()

    def _on_morph_done(self):
        if self._anim_rect_to is not None:
            self._apply_size_constraints(self.panel.isVisible(), self._anim_rect_to)
            self.setGeometry(self._anim_rect_to)
            self._render_caption()
            self.update()

    def _rim_accent_color(self) -> QColor:
        # Pure minimalist monochrome Apple Dynamic Island rim (never colorful neon borders)
        if self._hovered or time.monotonic() < self._overlap_flash_until:
            return QColor(255, 255, 255, 38)
        if self.engine.status in ("listening", "stopping", "loading"):
            return QColor(255, 255, 255, 30)
        return QColor(255, 255, 255, 20)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        card = QRectF(
            SHADOW_PAD_X,
            SHADOW_PAD_TOP,
            max(10.0, self.width() - 2 * SHADOW_PAD_X),
            max(10.0, self.height() - SHADOW_PAD_TOP - SHADOW_PAD_BOT),
        )
        radius = min(20.0, card.height() / 2.0)

        # Compact subtle Apple floating shadow
        p.setPen(Qt.NoPen)
        for dy, expand, alpha in ((4.0, 2.5, 18), (2.5, 1.0, 30), (1.2, 0.2, 42)):
            srect = card.adjusted(-expand, dy - expand * 0.3, expand, dy + expand * 0.5)
            sr = min(radius + expand * 0.4, srect.height() / 2.0)
            p.setBrush(QColor(0, 0, 0, alpha))
            p.drawRoundedRect(srect, sr, sr)

        # Pure obsidian Dynamic Island body
        path = QPainterPath()
        path.addRoundedRect(card, radius, radius)
        p.fillPath(path, QColor(8, 8, 10, 248))

        # Ultra-subtle neutral top sheen
        sheen = QLinearGradient(card.left(), card.top(), card.left(), card.top() + min(20.0, card.height() * 0.45))
        sheen.setColorAt(0.0, QColor(255, 255, 255, 11))
        sheen.setColorAt(1.0, QColor(255, 255, 255, 0))
        p.fillPath(path, sheen)

        # Clean minimalist 1px monochrome hairline
        p.setPen(QPen(self._rim_accent_color(), 1.0))
        p.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), max(1.0, radius - 0.5), max(1.0, radius - 0.5))

    def _refresh_active_speaker(self):
        snap = self.engine.state.snapshot()
        latest_spk, latest_id, latest_ov = "", "", False
        items = list(snap["paragraphs"]) + list(snap["loose"])
        if items:
            latest_ov = bool(items[-1].get("overlap", False))
            for item in reversed(items):
                if item.get("speaker"):
                    latest_spk = item["speaker"]
                    latest_id = item.get("speaker_id", "")
                    if not latest_ov:
                        latest_ov = bool(item.get("overlap", False))
                    break
        changed = (latest_spk != self._active_speaker or latest_id != self._active_speaker_id)
        self._active_speaker = latest_spk
        self._active_speaker_id = latest_id
        if latest_ov and not self._last_overlap_seen:
            self._overlap_flash_until = time.monotonic() + 2.6
            changed = True
        self._last_overlap_seen = latest_ov
        if changed and self.engine.status == "listening" and self.engine.session_mode == "meeting":
            self._sync_island(animate=False)

    def _tray(self):
        self.tray = None
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return
        self.tray = QSystemTrayIcon(make_icon(), self)
        self.tray.setToolTip(APP_NAME)
        m = QMenu()
        m.addAction("Show / hide", self._toggle_visible)
        self.tray_toggle = m.addAction("Start", self._toggle_meeting)
        m.addAction("Settings…", self.open_settings)
        m.addSeparator()
        m.addAction("Quit", self.quit)
        self.tray_menu = m
        self.tray.setContextMenu(m)
        self.tray.activated.connect(
            lambda r: self._toggle_visible() if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()

    # -- engine events (GUI thread) ------------------------------------------
    def _on_event(self, kind, data):
        if kind == "status":
            self._update_controls()
            if data == "idle" and not self.quitting:
                self._idle_caption()
        elif kind == "partial":
            if data:
                self._caption(data, "partial")
            elif self.engine.status == "listening":
                self._caption(self.last_final or self._listening_hint(), "" if self.last_final else "hint")
        elif kind == "segment":
            self.last_final = data["text"]
            self._caption(data["text"], "")
            self._refresh_active_speaker()
            self._mark_dirty()
        elif kind in ("paragraph", "reset", "speakers"):
            if kind == "reset":
                self.last_final = ""
                self._active_speaker = ""
                self._active_speaker_id = ""
                self._overlap_flash_until = 0.0
                self._last_overlap_seen = False
                self.view_mode = "transcript"
            else:
                self._refresh_active_speaker()
            self._mark_dirty()
        elif kind == "report":
            if data and data.get("status") in ("generating", "done", "error"):
                self.view_mode = "report"
                self._set_drawer_tab("transcript")
                if not self.panel.isVisible():
                    self.toggle_panel()
                if data["status"] == "generating":
                    self._caption(f"Generating {data['label']}…", "hint")
                elif data["status"] == "done":
                    self._refresh_loaded_reports()
                    self._caption(f"{data['label']} ready (saved in session folder)", "hint")
                    self.caption_reset.start()
                elif data["status"] == "error":
                    self._caption(data.get("error", "Could not generate output"), "error")
                    self.caption_reset.start()
            self._mark_dirty()
        elif kind == "session_started":
            mode = (data or {}).get("mode", "meeting")
            if mode in ("dictation", "compose") and self.s.dictation_sound:
                play_cue("start")
            if mode == "meeting":
                self._loaded_session_dir = None
                self._loaded_session_label = "Live"
                self._loaded_session_rtl = None
                self._loaded_reports = []
                if self.drawer_tab == "history":
                    self._set_drawer_tab("transcript")
                else:
                    self._sync_drawer_chrome()
            self._caption(self._listening_hint(mode), "hint")
        elif kind == "session_ended":
            if not self.quitting:
                mode = (data or {}).get("mode", "meeting")
                if mode in ("dictation", "compose"):
                    self._handle_dictation_ended(data)
                else:
                    d_str = (data or {}).get("dir", "")
                    if d_str:
                        self._loaded_session_dir = Path(d_str)
                        self._loaded_session_label = "Latest"
                        self._refresh_loaded_reports()
                    self._sync_drawer_chrome()
                    if self.drawer_tab == "history":
                        self._refresh_history()
                    self._caption("Saved. Open the transcript with ⤢ or browse History", "hint")
                    self.caption_reset.start()
        elif kind == "compose_ready":
            self._on_compose_ready(*data)
        elif kind == "compose_unavailable":
            if self.s.dictation_sound:
                play_cue("error")
            self._caption(data or "Add a Gemini API key in Settings to use Ask AI.", "error")
            self.caption_reset.start()
        elif kind == "paste_done":
            self._on_paste_done(*data)
        elif kind in ("llm", "error"):
            self._update_llm_label()
            if self.engine.status == "idle" and not self.quitting:
                self._idle_caption()
        elif kind == "log":
            level, msg = data
            if self.engine.status == "loading":
                self._caption(msg, "hint")
            elif level == "error":
                self._caption(msg, "error")
                self.caption_reset.start()

    def _listening_hint(self, mode: str = None) -> str:
        if mode is None:
            mode = self.engine.session_mode
        if mode == "compose":
            ck = self.s.compose_hotkey or "hotkey"
            return f"Ask AI… speak prompt, press {ck} to paste"
        if mode == "dictation":
            dk = self.s.dictate_hotkey or "hotkey"
            return f"Dictating… press {dk} to paste"
        return "Listening…"

    def _toggle_meeting(self):
        self.engine.toggle("meeting")

    def _toggle_dictation(self):
        if self.engine.status == "listening" and self.engine.session_mode != "dictation":
            self._caption("Stop the current recording first", "hint")
            self.caption_reset.start()
            return
        if self.engine.status == "idle":
            self.dictate_target_hwnd = get_external_foreground()
        self.engine.toggle("dictation")

    def _toggle_compose(self):
        if self.engine.status == "listening":
            if self.engine.session_mode != "compose":
                self._caption("Stop the current recording first", "hint")
                self.caption_reset.start()
                return
            self.engine.toggle("compose")
            return
        if self.engine.status != "idle" or self._compose_starting:
            return
        hwnd = get_external_foreground()
        self.dictate_target_hwnd = hwnd
        if self.s.compose_selection and hwnd:
            clip = QGuiApplication.clipboard()
            prev_text = None
            mime = clip.mimeData()
            if mime is not None and mime.hasText():
                prev_text = clip.text()
            self._compose_starting = True
            self._caption("Reading selection…", "hint")

            def cap():
                changed = capture_selection_copy(hwnd)
                self.bridge.event.emit("compose_ready", (hwnd, changed, prev_text))

            threading.Thread(target=cap, daemon=True, name="eloquent-capture").start()
            return
        if not self.engine.start(mode="compose", selected_text=""):
            self.dictate_target_hwnd = 0

    def _on_compose_ready(self, _hwnd: int, changed: bool, prev_text):
        self._compose_starting = False
        if self.quitting or self.engine.status != "idle":
            return
        selected = ""
        if changed:
            clip = QGuiApplication.clipboard()
            mime = clip.mimeData()
            if mime is not None and mime.hasText():
                selected = clip.text()
            if self.s.restore_clipboard and prev_text is not None and selected:
                clip.setText(prev_text)
        if not self.engine.start(mode="compose", selected_text=selected):
            self.dictate_target_hwnd = 0

    def _clear_compose_history(self):
        self.engine.clear_compose_history()
        self._caption("Ask AI memory cleared", "hint")
        self.caption_reset.start()

    def _handle_dictation_ended(self, data: dict):
        mode = (data or {}).get("mode", "dictation")
        composing = (mode == "compose")
        paste_text = (data or {}).get("paste_text", "")
        cancelled = bool((data or {}).get("cancelled"))
        err = (data or {}).get("error", "")
        target_hwnd = self.dictate_target_hwnd
        self.dictate_target_hwnd = 0
        if cancelled:
            if self.s.dictation_sound:
                play_cue("cancel")
            msg = "Ask AI cancelled (kept in ⤢)" if composing else "Dictation cancelled (kept in ⤢)"
            self._caption(msg, "hint")
            self.caption_reset.start()
            return
        if err:
            if self.s.dictation_sound:
                play_cue("error")
            self._caption(f"Ask AI failed (kept prompt in ⤢): {err}" if composing else err, "error")
            self.caption_reset.start()
            return
        if not paste_text:
            if self.s.dictation_sound:
                play_cue("error")
            self._caption("No speech detected", "hint")
            self.caption_reset.start()
            return
        clip = QGuiApplication.clipboard()
        prev_text = None
        if self.s.restore_clipboard:
            mime = clip.mimeData()
            if mime is not None and mime.hasText():
                prev_text = clip.text()
        clip.setText(paste_text)

        def do_paste():
            ok = send_paste(target_hwnd)
            self.bridge.event.emit("paste_done", (ok, paste_text, prev_text))

        threading.Thread(target=do_paste, daemon=True, name="eloquent-paste").start()

    def _on_paste_done(self, ok: bool, pasted_text: str, prev_text):
        if self.quitting:
            return
        if ok:
            if prev_text is not None:
                QTimer.singleShot(380, lambda pt=pasted_text, prev=prev_text: self._restore_clipboard(pt, prev))
            if self.s.dictation_sound:
                play_cue("done")
            self._caption("Pasted!", "success")
            self.caption_reset.start(2200)
        else:
            if self.s.dictation_sound:
                play_cue("error")
            self._caption("Copied to clipboard (press Ctrl+V)", "hint")
            self.caption_reset.start(4500)

    @staticmethod
    def _restore_clipboard(pasted_text: str, prev_text: str):
        clip = QGuiApplication.clipboard()
        if clip.text() == pasted_text:
            clip.setText(prev_text)

    def _caption(self, text: str, kind: str, _from_idle: bool = False):
        self.caption_text = text
        if kind != self.caption_kind:
            self.caption_kind = kind
            self.caption.setProperty("kind", kind)
            self.caption.style().unpolish(self.caption)
            self.caption.style().polish(self.caption)
        if _from_idle:
            self._toast_active = False
        elif self.engine.status == "idle" and text:
            self._toast_active = True
            self.dot.set_state(kind if kind in ("success", "error") else "idle")
        self._render_caption()
        if not _from_idle and (self.engine.status == "idle" or self._toast_active):
            self._sync_island(animate=True)

    @staticmethod
    def _has_arabic(text: str) -> bool:
        return any("\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff" for ch in (text or ""))

    def _is_rtl_language(self) -> bool:
        return hasattr(self.s, "resolved_speech_language") and is_arabic_speech(self.s.resolved_speech_language())

    def _is_view_rtl(self) -> bool:
        if self._loaded_session_rtl is not None:
            return self._loaded_session_rtl
        return self._is_rtl_language()

    def _sync_view_direction(self):
        rtl = self._is_view_rtl()
        self.view.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)

    def _render_caption(self):
        rtl_speech = (self._is_rtl_language() or self._has_arabic(self.caption_text)) and self.caption_kind in ("", "partial")
        self.caption.setLayoutDirection(Qt.RightToLeft if rtl_speech else Qt.LeftToRight)
        # Qt.AlignAbsolute prevents Qt from double-flipping AlignRight to visual-left under RightToLeft
        self.caption.setAlignment((Qt.AlignRight if rtl_speech else Qt.AlignLeft) | Qt.AlignAbsolute | Qt.AlignVCenter)
        if not self.caption.isVisible():
            txt = ("\u200F" + self.caption_text) if (rtl_speech and self.caption_text) else self.caption_text
            self.caption.setText(txt)
            self.caption.setToolTip(self.caption_text if len(self.caption_text) > 60 else "")
            return
        fm = self.caption.fontMetrics()
        elide = Qt.ElideLeft if self.caption_kind in ("", "partial") else Qt.ElideRight
        avail = max(16, self.caption.contentsRect().width() - 4)
        shown = fm.elidedText(self.caption_text, elide, avail)
        self.caption.setText(("\u200F" + shown) if (rtl_speech and shown) else shown)
        self.caption.setToolTip(self.caption_text if len(self.caption_text) > 60 else "")

    def _dot_status(self) -> str:
        if self.engine.status == "listening":
            if self.engine.session_mode == "compose":
                return "composing"
            if self.engine.session_mode == "dictation":
                return "dictating"
        if self.engine.status == "idle" and self._toast_active:
            if self.caption_kind == "success":
                return "success"
            if self.caption_kind == "error":
                return "error"
        return self.engine.status

    def _idle_caption(self):
        e = self.engine
        if e.status != "idle":
            return
        parts = []
        if self._is_rtl_language():
            parts.append("AR")
        src_tag = {"system": "System audio", "both": "Mic + System"}.get(self.s.audio_source, "")
        if src_tag:
            parts.append(src_tag)
        if self.hotkey_ok and self.s.hotkey:
            parts.append(f"▶ {self.s.hotkey}")
        elif not (self.dictate_hotkey_ok or self.compose_hotkey_ok):
            parts.append("Ready. Press ▶")
        if self.dictate_hotkey_ok and self.s.dictate_hotkey:
            parts.append(f"Dictate: {self.s.dictate_hotkey}")
        if self.compose_hotkey_ok and self.s.compose_hotkey:
            parts.append(f"Ask AI: {self.s.compose_hotkey}")
        if e.llm["state"] == "off" and self.s.cleanup:
            parts.append("no Gemini key, raw text only")
        idle_str = " · ".join(parts) if parts else "Ready. Press ▶"
        self.brand_lbl.setToolTip(idle_str)
        self.lang_pill.setToolTip(idle_str)
        self._caption(idle_str, "hint", _from_idle=True)
        self.caption_reset.setInterval(4500)
        self.dot.set_state("idle")
        self._sync_island(animate=True)

    def _update_controls(self):
        st = self.engine.status
        dot_st = self._dot_status()
        on = st == "listening"
        mode = self.engine.session_mode
        dictating = (mode == "dictation")
        composing = (mode == "compose")
        if st != "idle":
            self._toast_active = False
        self.btn_toggle.setText("■" if on else "▶")
        self.btn_toggle.setEnabled(st in ("idle", "listening"))
        self.btn_toggle.setProperty("on", "true" if on else "false")
        self.btn_toggle.style().unpolish(self.btn_toggle)
        self.btn_toggle.style().polish(self.btn_toggle)
        if on and composing:
            self.btn_toggle.setToolTip("Cancel Ask AI (keep prompt in drawer)")
        elif on and dictating:
            self.btn_toggle.setToolTip("Cancel dictation (keep in drawer)")
        else:
            hk = f" ({self.s.hotkey})" if self.hotkey_ok and self.s.hotkey else ""
            self.btn_toggle.setToolTip(("Stop" if on else "Start") + hk)
        if self.tray:
            self.tray_toggle.setText("Stop" if on else "Start")
            self.tray_toggle.setEnabled(st in ("idle", "listening"))
            self.tray.setToolTip(f"{APP_NAME} - {dot_st}")
        if st in ("listening", "stopping", "loading"):
            self.level_timer.start()
        else:
            self.level_timer.stop()
            self.dot.set_state(dot_st)
            self.waveform.set_state("idle", 0.0)
        if st == "stopping":
            if composing:
                msg = "Generating & pasting…"
            elif dictating:
                msg = "Cleaning & pasting…"
            else:
                msg = "Finishing the last sentence…"
            self._caption(msg, "hint")
        self._sync_island(animate=True)

    def _tick_level(self):
        dot_st = self._dot_status()
        ov_active = time.monotonic() < self._overlap_flash_until
        self.dot.set_state(dot_st, self.engine.level)
        self.dot.advance(0.033)
        self.waveform.set_state(dot_st, self.engine.level, overlap_glow=ov_active)
        self.waveform.tick(0.033)
        if ov_active != getattr(self, "_was_ov_active", False):
            self._was_ov_active = ov_active
            self._update_mode_pill()
            self.update()

    def _update_llm_label(self):
        llm, err = self.engine.llm, self.engine.state.error
        text = "LLM error, raw text kept" if err or llm["state"] == "error" else ""
        self.lbl_llm.setText(text)
        self.lbl_llm.setToolTip(err or llm.get("message", "") if text else "")

    # -- transcript panel ---------------------------------------------------
    def _mark_dirty(self):
        self.dirty = True
        if self.panel.isVisible() and not self.rebuild_timer.isActive():
            self.rebuild_timer.start()

    def _refresh_loaded_reports(self):
        from ..session import _list_saved_reports
        target = self._loaded_session_dir or self.engine.last_session_dir
        self._loaded_reports = _list_saved_reports(Path(target)) if target and Path(target).exists() else []

    def _sync_drawer_chrome(self):
        in_hist = (self.drawer_tab == "history")
        self.tab_transcript.setChecked(not in_hist)
        self.tab_history.setChecked(in_hist)
        self.drawer_stack.setCurrentIndex(1 if in_hist else 0)
        self.lbl_session_tag.setText("" if in_hist else self._loaded_session_label)
        self.lbl_hist_count.setVisible(in_hist)
        self.btn_hist_refresh.setVisible(in_hist)
        self.btn_gen.setVisible(not in_hist)
        self.btn_copy.setVisible(not in_hist)
        self.btn_folder.setToolTip("Open sessions folder" if in_hist else "Open session folder")
        if in_hist:
            self.chk_raw.hide()
            self.btn_back.hide()
            self.history_banner.setVisible(self.engine.busy)

    def _set_drawer_tab(self, tab: str):
        self.drawer_tab = "history" if tab == "history" else "transcript"
        self._sync_drawer_chrome()
        if self.drawer_tab == "history":
            self._refresh_history()
        else:
            self._rebuild()

    def _refresh_history(self):
        query = self.history_search.text() if hasattr(self, "history_search") else ""
        sessions = list_sessions(self.s.resolved_sessions_dir(), query=query)
        self.history_list.clear()
        cur_dir = None
        if self._loaded_session_dir:
            try:
                cur_dir = Path(self._loaded_session_dir).resolve()
            except OSError:
                cur_dir = None
        elif self.engine.last_session_dir:
            try:
                cur_dir = Path(self.engine.last_session_dir).resolve()
            except OSError:
                cur_dir = None

        busy = self.engine.busy
        self.history_banner.setVisible(busy)
        for meta in sessions:
            try:
                is_active = cur_dir is not None and Path(meta["dir"]).resolve() == cur_dir
            except OSError:
                is_active = False
            row_w = _SessionRowWidget(meta, active=is_active, locked=busy)
            item = QListWidgetItem(self.history_list)
            item.setData(Qt.UserRole, meta)
            item.setSizeHint(QSize(0, 52))
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, row_w)

        n = len(sessions)
        self.lbl_hist_count.setText(f"{n} session{'s' if n != 1 else ''}")
        empty = (n == 0)
        self.history_list.setVisible(not empty)
        self.history_empty.setText(
            "No matching sessions." if (empty and query.strip()) else "No saved sessions yet."
        )
        self.history_empty.setVisible(empty)

    def _on_history_item_clicked(self, item: QListWidgetItem):
        meta = item.data(Qt.UserRole)
        if meta:
            self._open_history_session(meta)

    def _open_history_session(self, meta: dict):
        if self.engine.busy:
            self._caption("Stop the active recording before opening a past session", "hint")
            self.caption_reset.start()
            return
        data = self.engine.load_session(meta["dir"])
        if not data:
            self._caption("Could not open that session", "error")
            self.caption_reset.start()
            return
        self._loaded_session_dir = Path(data["dir"])
        self._loaded_session_label = data["date_label"]
        self._loaded_session_rtl = bool(data["rtl"])
        self._loaded_reports = list(data.get("reports") or [])
        self._scroll_to_top_once = True
        self.view_mode = "transcript"
        self._set_drawer_tab("transcript")

    def _open_saved_report_item(self, rep_meta: dict):
        if self.engine.open_saved_report(rep_meta):
            self.view_mode = "report"
            self._set_drawer_tab("transcript")

    def _delete_history_session(self, meta: dict):
        target = Path(meta["dir"])
        if self.engine.busy and self.engine.session is not None:
            try:
                if Path(self.engine.session.dir).resolve() == target.resolve():
                    self._caption("Cannot delete the active recording session", "hint")
                    self.caption_reset.start()
                    return
            except OSError:
                pass
        dlg = _ConfirmDialog(
            f"{APP_NAME} - Delete Session",
            f'Delete session "{meta["date_label"]}" and all files inside its folder? This cannot be undone.',
            ok_label="Delete",
            parent=self,
        )
        if not dlg.exec():
            return
        if not delete_session(target, self.s.resolved_sessions_dir()):
            self._caption("Could not delete session folder", "error")
            self.caption_reset.start()
            return
        # If the deleted session was currently open in the viewer, reset state cleanly
        if self._loaded_session_dir:
            try:
                if Path(self._loaded_session_dir).resolve() == target.resolve():
                    self._loaded_session_dir = None
                    self._loaded_session_label = ""
                    self._loaded_session_rtl = None
                    self._loaded_reports = []
                    self.engine.last_session_dir = None
                    self.engine.state.reset()
            except OSError:
                pass
        self._refresh_history()

    def _show_history_context_menu(self, pos: QPoint):
        item = self.history_list.itemAt(pos)
        if item is None:
            return
        meta = item.data(Qt.UserRole)
        if not meta:
            return
        menu = QMenu(self)
        act_open = menu.addAction("Open session", lambda m=meta: self._open_history_session(m))
        act_open.setEnabled(not self.engine.busy)
        menu.addAction("Open folder", lambda p=meta["dir"]: self._open(Path(p)))
        menu.addSeparator()
        menu.addAction("Delete session…", lambda m=meta: self._delete_history_session(m))
        self._menu_open = True
        try:
            menu.exec(self.history_list.viewport().mapToGlobal(pos))
        finally:
            self._menu_open = False

    def _rebuild(self):
        if not self.panel.isVisible():
            return
        self._sync_drawer_chrome()
        if self.drawer_tab == "history":
            self.dirty = False
            return
        rtl = self._is_view_rtl()
        self._sync_view_direction()
        snap = self.engine.state.snapshot()
        rep = snap.get("report")
        has_rep = bool(rep and rep.get("status") in ("generating", "done", "error"))
        if self.view_mode == "report" and has_rep:
            self.chk_raw.hide()
            self.btn_back.setText("← Transcript")
            self.btn_back.show()
            st = rep["status"]
            if st == "generating":
                self.view.setHtml(
                    f'<p dir="ltr" align="left" style="color:#a1a1aa"><i>Generating {html.escape(rep["label"])}…</i></p>')
            elif st == "error":
                self.view.setHtml(
                    f'<p dir="ltr" align="left" style="color:#ff8a80"><b>{html.escape(rep["label"])}</b><br>'
                    f'{html.escape(rep.get("error", ""))}</p>')
            else:
                from eloquent.ui.markdown_math import render_report_qt
                max_w = max(240, self.view.viewport().width() - 28)
                self.view.setHtml(
                    render_report_qt(self.view.document(), rep["label"], rep["text"], max_width=max_w, rtl=rtl)
                )
        else:
            self.view_mode = "transcript"
            self.chk_raw.show()
            if has_rep and rep.get("status") == "done":
                short_lbl = _short_report_tag(rep["label"])
                self.btn_back.setText(f"{short_lbl} →")
                self.btn_back.setToolTip(f"View {rep['label']}")
                self.btn_back.show()
            else:
                self.btn_back.hide()
            opt = self.view.document().defaultTextOption()
            opt.setTextDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)
            opt.setAlignment(Qt.AlignRight if rtl else Qt.AlignLeft)
            self.view.document().setDefaultTextOption(opt)
            sb = self.view.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 30
            keep = sb.value()
            self.view.setHtml(transcript_html(snap, self.show_raw, rtl=rtl))
            if self._scroll_to_top_once:
                self._scroll_to_top_once = False
                sb.setValue(0)
            else:
                sb.setValue(sb.maximum() if at_bottom else keep)
        self.dirty = False

    def _switch_view(self):
        self.view_mode = "transcript" if self.view_mode == "report" else "report"
        self._rebuild()

    def _show_gen_menu(self):
        self._refresh_loaded_reports()
        multi = bool(self.engine.state.snapshot().get("multi_speaker")) or self.engine.speaker_edit_count() > 0
        menu = QMenu(self)
        if self._loaded_reports:
            saved_sub = menu.addMenu(f"Saved outputs ({len(self._loaded_reports)})")
            for rep_meta in self._loaded_reports:
                saved_sub.addAction(
                    rep_meta["label"],
                    lambda rm=rep_meta: self._open_saved_report_item(rm),
                )
            menu.addSeparator()
        if multi:
            menu.addAction("Rename speakers…", self.rename_speakers)
            menu.addSeparator()
        for fmt in REPORT_FORMATS:
            menu.addAction(fmt["label"], lambda fid=fmt["id"]: self._trigger_gen(fid))
        menu.addSeparator()
        menu.addAction("Custom Prompt…", self._trigger_custom_gen)
        self._menu_open = True
        try:
            menu.exec(self.btn_gen.mapToGlobal(QPoint(0, self.btn_gen.height())))
        finally:
            self._menu_open = False
            self._sync_island(animate=True)

    def _build_speaker_menu(self, spk_id: str, item_kind: str = "", item_id: int = None) -> QMenu:
        """Build the 1-click dark popover menu shown when clicking a speaker badge."""
        cur = self.engine.state.display_speaker(spk_id) or spk_id
        src_anchors = self.engine.speaker_anchor_ids(spk_id)
        others = [
            (cid, name, self.engine.speaker_anchor_ids(cid))
            for cid, name in self.engine.state.speakers()
            if cid != spk_id
        ]
        can_dissolve = self.engine.can_dissolve_speaker(spk_id) and len(others) >= 2
        edit_count = self.engine.speaker_edit_count()

        menu = QMenu(self)
        safe_cur = cur.replace("&", "&&")

        if item_kind and item_id is not None and others:
            line_sub = menu.addMenu("Change this line to")
            for ocid, oname, oanc in others:
                olbl = f"{oname} ({ocid})" if (oname and oname != ocid) else ocid
                act_l = line_sub.addAction(_speaker_dot_icon(_speaker_color(ocid)), olbl.replace("&", "&&"))
                act_l.triggered.connect(
                    lambda _c=False, ik=item_kind, iid=item_id, d=ocid, da=oanc, dn=oname:
                        self._reassign_one_line(ik, iid, d, da, dn)
                )

        act_ren = menu.addAction(f'Rename "{safe_cur}"…')
        act_ren.triggered.connect(
            lambda _c=False, sid=spk_id, cname=cur, anc=src_anchors: self._rename_one_speaker(sid, cname, anc)
        )

        if others:
            menu.addSeparator()
            merge_sub = menu.addMenu(f'Merge all "{safe_cur}" into')
            for ocid, oname, oanc in others:
                olbl = f"{oname} ({ocid})" if (oname and oname != ocid) else ocid
                act_m = merge_sub.addAction(_speaker_dot_icon(_speaker_color(ocid)), olbl.replace("&", "&&"))
                act_m.triggered.connect(
                    lambda _c=False, s=spk_id, sa=src_anchors, sn=cur, d=ocid, da=oanc, dn=oname:
                        self._merge_one_speaker(s, sa, sn, d, da, dn)
                )
            if can_dissolve:
                act_dis = menu.addAction("Reassign all overlap turns")
                act_dis.triggered.connect(
                    lambda _c=False, s=spk_id, sa=src_anchors, sn=cur: self._dissolve_one_speaker(s, sa, sn)
                )

        if edit_count >= 1:
            menu.addSeparator()
            act_u = menu.addAction("Undo last speaker change")
            act_u.triggered.connect(self._undo_speaker_edit)
            if edit_count >= 2:
                act_r = menu.addAction("Reset speaker edits")
                act_r.triggered.connect(self._reset_speaker_edits)
        return menu

    def _reassign_one_line(self, item_kind: str, item_id: int, dst_id: str,
                           dst_anchors: list[int], dst_name: str):
        if self.engine.reassign_line_speaker(item_kind, item_id, dst_id, dst_anchors=dst_anchors):
            self._rebuild()
            self._caption(f"Changed line to {dst_name}", "hint")
            self.caption_reset.start()

    def _rename_one_speaker(self, spk_id: str, cur_name: str, src_anchors: list[int]):
        live_id = self.engine.resolve_speaker_from_anchors(spk_id, src_anchors)
        if not src_anchors:
            src_anchors = self.engine.speaker_anchor_ids(live_id)
        dlg = _RenameSpeakerDialog(live_id, self.engine.state.display_speaker(live_id) or cur_name, self)
        if dlg.exec():
            target_id = self.engine.resolve_speaker_from_anchors(live_id, src_anchors)
            self.engine.rename_speakers({target_id: dlg.value()})
            self._rebuild()

    def _merge_one_speaker(self, src_id: str, src_anchors: list[int], src_name: str,
                           dst_id: str, dst_anchors: list[int], dst_name: str):
        if self.engine.merge_speakers(src_id, dst_id, src_anchors=src_anchors, dst_anchors=dst_anchors):
            self._rebuild()
            self._caption(f"Merged all {src_name} into {dst_name}", "hint")
            self.caption_reset.start()

    def _dissolve_one_speaker(self, src_id: str, src_anchors: list[int], src_name: str):
        if self.engine.dissolve_speaker(src_id, src_anchors=src_anchors):
            self._rebuild()
            self._caption(f"Reassigned {src_name} turns to closest speakers", "hint")
            self.caption_reset.start()

    def _undo_speaker_edit(self):
        if self.engine.undo_last_speaker_edit():
            self._rebuild()
            self._caption("Undid last speaker change", "hint")
            self.caption_reset.start()

    def _reset_speaker_edits(self):
        if self.engine.reset_speaker_edits():
            self._rebuild()
            self._caption("Reset all speaker edits", "hint")
            self.caption_reset.start()

    @staticmethod
    def _parse_speaker_url(url: QUrl) -> tuple[str, str, int | None]:
        raw = url.toString()
        if not raw.startswith("spk:"):
            return "", "", None
        payload = raw[4:]
        item_kind, item_id = "", None
        if "?" in payload:
            spk_part, query = payload.split("?", 1)
            spk_id = urllib.parse.unquote(spk_part)
            ref_vals = urllib.parse.parse_qs(query).get("ref") or []
            ref_part = ref_vals[0] if ref_vals else ""
            if ":" in ref_part:
                rk, rid = ref_part.split(":", 1)
                if rk in ("p", "l"):
                    try:
                        item_kind, item_id = rk, int(rid)
                    except ValueError:
                        pass
        else:
            spk_id = urllib.parse.unquote(payload)
        return spk_id, item_kind, item_id

    def _on_anchor(self, url: QUrl):
        spk_id, item_kind, item_id = self._parse_speaker_url(url)
        if not spk_id:
            return
        menu = self._build_speaker_menu(spk_id, item_kind=item_kind, item_id=item_id)
        menu.exec(QCursor.pos())

    def rename_speakers(self):
        speakers = self.engine.state.speakers()
        edit_count = self.engine.speaker_edit_count()
        if not speakers and edit_count == 0:
            return
        anchors = {cid: self.engine.speaker_anchor_ids(cid) for cid, _ in speakers}
        dlg = RenameSpeakersDialog(speakers, edit_count=edit_count, parent=self)
        if dlg.exec():
            special = dlg.special_action()
            if special == "undo":
                self._undo_speaker_edit()
            elif special == "reset_edits":
                self._reset_speaker_edits()
            else:
                resolved = {}
                for cid, val in dlg.values().items():
                    target = self.engine.resolve_speaker_from_anchors(cid, anchors.get(cid, []))
                    if target not in resolved or val:
                        resolved[target] = val
                self.engine.rename_speakers(resolved)
                self._rebuild()

    def _show_view_context_menu(self, pos: QPoint):
        menu = self.view.createStandardContextMenu(pos)
        if self.view_mode == "transcript":
            sel = self.view.textCursor().selectedText().replace("\u2029", " ").strip()
            if 1 <= len(sel) <= 80:
                short = sel if len(sel) <= 28 else sel[:25] + "…"
                safe_short = short.replace("&", "&&")
                act_fix = QAction(f'Correct "{safe_short}"…', menu)
                act_fix.triggered.connect(lambda _c=False, t=sel: self._fix_and_remember_selection(t))
                acts = menu.actions()
                if acts:
                    sep = menu.insertSeparator(acts[0])
                    menu.insertAction(sep, act_fix)
                else:
                    menu.addAction(act_fix)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _fix_and_remember_selection(self, selected: str):
        dlg = _TextPromptDialog(
            f"{APP_NAME} - Correct Word",
            f'Correct "{selected}" to (also remembered for future transcripts):',
            initial=selected,
            ok_label="Save",
            parent=self,
        )
        if not dlg.exec():
            return
        replacement = dlg.value()
        if not replacement or replacement == selected:
            return
        count = self.engine.apply_vocabulary_fix(selected, replacement, remember=True)
        self._rebuild()
        msg = (f'Corrected ({count}): {selected} -> {replacement}'
               if count else f'Remembered: {selected} -> {replacement}')
        self._caption(msg, "hint")
        self.caption_reset.start()

    def _trigger_gen(self, format_id: str, custom_prompt: str = ""):
        self.engine.generate_report(format_id, custom_prompt)

    def _trigger_custom_gen(self):
        dlg = _TextPromptDialog(
            f"{APP_NAME} - Custom Prompt",
            "How should the LLM format or analyze this transcript?",
            initial="Summarize the key points and list any action items.",
            ok_label="Generate",
            parent=self,
        )
        if dlg.exec() and dlg.value():
            self._trigger_gen("custom", dlg.value())

    def toggle_panel(self):
        show = not self.panel.isVisible()
        self.panel.setVisible(show)
        self.btn_expand.setToolTip("Hide transcript" if show else "Show transcript")
        if show:
            if self.drawer_tab == "history":
                self._refresh_history()
            self._rebuild()
        self._sync_island(animate=True)

    def _set_raw(self, on):
        self.show_raw = on
        self._rebuild()

    def _copy(self):
        snap = self.engine.state.snapshot()
        rep = snap.get("report")
        if self.view_mode == "report" and rep and rep.get("text"):
            QGuiApplication.clipboard().setText(rep["text"])
            self._caption(f"{rep['label']} copied to the clipboard", "hint")
        else:
            QGuiApplication.clipboard().setText(self.engine.state.full_text(raw=self.show_raw))
            self._caption("Transcript copied to the clipboard", "hint")
        self.caption_reset.start()

    def _open(self, path: Path):
        Path(path).mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_folder(self):
        if self.drawer_tab == "history":
            self._open(self.s.resolved_sessions_dir())
        else:
            self._open(self._loaded_session_dir or self.engine.last_session_dir or self.s.resolved_sessions_dir())

    def _make_shortcut(self):
        from ..hotkeys import spawn_listener
        from ..shortcuts import create_windows_shortcuts
        if create_windows_shortcuts():
            spawn_listener()
            self._caption(f"Shortcuts created. Launch anytime with {self.s.launch_hotkey}", "hint")
        else:
            self._caption("Could not create shortcuts", "error")
        self.caption_reset.start()

    # -- menu / settings ------------------------------------------------------
    def _show_menu(self):
        is_ar = self._is_rtl_language()
        hk = f" ({self.s.switch_model_hotkey})" if self.s.switch_model_hotkey else ""
        self.act_switch_model.setText(("Switch to Parakeet (Default)" if is_ar else "Switch to Arabic") + hk)
        self.act_switch_model.setEnabled(not self.engine.busy and self.engine.status != "loading")
        cur_src = self.s.audio_source if self.s.audio_source in self.src_actions else "mic"
        for k, act in self.src_actions.items():
            act.blockSignals(True)
            act.setChecked(k == cur_src)
            act.blockSignals(False)
        self.max_spk_menu.setEnabled(bool(self.s.diarize))
        cur_cap = int(getattr(self.s, "max_speakers", 0) or 0)
        if cur_cap not in self.max_spk_actions:
            cur_cap = 0
        for cap, act in self.max_spk_actions.items():
            act.blockSignals(True)
            act.setChecked(cap == cur_cap)
            act.blockSignals(False)
        self.act_audio.blockSignals(True)
        self.act_audio.setChecked(self.s.save_audio)
        self.act_audio.blockSignals(False)
        cur_sz = normalize_island_size(getattr(self.s, "island_size", "medium"))
        for k, act in self.size_actions.items():
            act.blockSignals(True)
            act.setChecked(k == cur_sz)
            act.blockSignals(False)
        self.act_top.blockSignals(True)
        self.act_top.setChecked(self.s.always_on_top)
        self.act_top.blockSignals(False)
        self._menu_open = True
        try:
            self.menu.exec(self.btn_menu.mapToGlobal(QPoint(0, self.btn_menu.height())))
        finally:
            self._menu_open = False
            self._leave_timer.start()

    def _set_audio_source(self, src: str):
        self.s.audio_source = src
        self.s.save()
        label = AUDIO_SOURCE_LABELS.get(src, src)
        if self.engine.status != "idle":
            self._caption(f"Audio source set to {label} (applies to next recording)", "hint")
            self.caption_reset.start()
        else:
            self._idle_caption()

    def _set_max_speakers(self, cap: int):
        cap = int(cap) if int(cap or 0) >= 2 else 0
        self.s.max_speakers = cap
        self.s.save()
        self.engine.set_max_speakers(cap)
        lbl = "Auto" if cap == 0 else f"{cap} speakers"
        self._caption(f"Max speakers set to {lbl}", "hint")
        self.caption_reset.start()

    def _set_island_size(self, size_key: str):
        sz = normalize_island_size(size_key)
        self.s.island_size = sz
        self.s.save()
        preset = self._island_preset()
        self._expanded_w = preset["expanded_w"]
        self._expanded_h = preset["expanded_h"]
        self._sync_island(animate=True)

    def _set_audio(self, on):
        self.s.save_audio = on
        self.s.save()
        if self.engine.status == "listening":
            self._caption("Audio setting applies to the next recording", "hint")
            self.caption_reset.start()

    def _set_on_top(self, on):
        self.s.always_on_top = on
        self.s.save()
        self.unregister_hotkey()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        self.show()
        self.register_hotkey()

    def _toggle_speech_model(self):
        if self.engine.busy or self.engine.status == "loading":
            self._caption("Stop the current recording before switching models", "hint")
            self.caption_reset.start()
            return
        target = "parakeet" if self._is_rtl_language() else "arabic"
        if hasattr(self.s, "set_active_speech_language"):
            self.s.set_active_speech_language(target)
        else:
            self.s.speech_language = target
        self._sync_view_direction()
        self._render_caption()
        if self.panel.isVisible():
            self._rebuild()
        label = "Arabic (Cohere)" if target == "arabic" else "Parakeet (Default)"
        self._caption(f"Switching to {label}…", "hint")
        threading.Thread(target=self.engine.reload_speech_model, daemon=True,
                         name="eloquent-asr-reload").start()

    def open_settings(self, first_run=False):
        old_active = self.s.resolved_speech_language() if hasattr(self.s, "resolved_speech_language") else "parakeet"
        old_saved = normalize_speech_language(getattr(self.s, "speech_language", "parakeet"))
        old_size = normalize_island_size(getattr(self.s, "island_size", "medium"))
        dlg = SettingsDialog(self.s, self, first_run=first_run,
                             speech_busy=(self.engine.busy or self.engine.status == "loading"))
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        if dlg.exec():
            if normalize_speech_language(self.s.speech_language) != old_saved and hasattr(self.s, "set_active_speech_language"):
                self.s.set_active_speech_language(None)
            if normalize_island_size(getattr(self.s, "island_size", "medium")) != old_size:
                preset = self._island_preset()
                self._expanded_w = preset["expanded_w"]
                self._expanded_h = preset["expanded_h"]
            new_active = self.s.resolved_speech_language() if hasattr(self.s, "resolved_speech_language") else "parakeet"
            hk_ok = self.register_hotkey()
            self.engine.refresh_provider()
            self.engine.sync_diarizer()
            self._sync_view_direction()
            self._update_llm_label()
            self._update_controls()
            if self.panel.isVisible():
                self._rebuild()
            if new_active != old_active and not self.engine.busy:
                threading.Thread(target=self.engine.reload_speech_model, daemon=True,
                                 name="eloquent-asr-reload").start()
            elif self.engine.status == "idle" and hk_ok:
                self._idle_caption()

    def open_vocabulary(self):
        dlg = VocabularyDialog(getattr(self.s, "custom_vocabulary", ""), self)
        if dlg.exec():
            self.s.custom_vocabulary = dlg.vocabulary_string()
            self.s.save()
            self.engine.refresh_provider()
            terms, pairs = parse_custom_vocabulary(self.s.custom_vocabulary)
            self._caption(f"Dictionary saved ({len(terms)} names, {len(pairs)} corrections)", "hint")
            self.caption_reset.start()

    def _toggle_visible(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    # -- global hotkey (Windows) -----------------------------------------------
    def register_hotkey(self) -> bool:
        self.unregister_hotkey()
        if sys.platform != "win32":
            return True
        hwnd = int(self.winId())
        failed = []
        for hk_id, text, attr in (
            (HOTKEY_ID, self.s.hotkey, "hotkey_ok"),
            (HOTKEY_DICTATE_ID, self.s.dictate_hotkey, "dictate_hotkey_ok"),
            (HOTKEY_COMPOSE_ID, self.s.compose_hotkey, "compose_hotkey_ok"),
            (HOTKEY_SWITCH_MODEL_ID, self.s.switch_model_hotkey, "switch_model_hotkey_ok"),
        ):
            ok = False
            if text:
                parsed = parse_hotkey(text)
                if parsed:
                    mods, vk = parsed
                    ok = bool(ctypes.windll.user32.RegisterHotKey(hwnd, hk_id, mods, vk))
                    if not ok:
                        failed.append(text)
            setattr(self, attr, ok)
        if failed:
            joined = ", ".join(failed)
            self._caption(f"Hotkey {joined} is used by another app (change it in Settings)", "error")
            self.caption_reset.start()
            return False
        return True

    def unregister_hotkey(self):
        if sys.platform == "win32":
            hwnd = int(self.winId())
            for hk_id, attr in (
                (HOTKEY_ID, "hotkey_ok"),
                (HOTKEY_DICTATE_ID, "dictate_hotkey_ok"),
                (HOTKEY_COMPOSE_ID, "compose_hotkey_ok"),
                (HOTKEY_SWITCH_MODEL_ID, "switch_model_hotkey_ok"),
            ):
                if getattr(self, attr, False):
                    ctypes.windll.user32.UnregisterHotKey(hwnd, hk_id)
                    setattr(self, attr, False)

    def nativeEvent(self, eventType, message):
        try:
            et = eventType.data() if hasattr(eventType, "data") else bytes(eventType)
            if et == b"windows_generic_MSG":
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY:
                    if msg.wParam == HOTKEY_ID:
                        self._toggle_meeting()
                        return True, 0
                    if msg.wParam == HOTKEY_DICTATE_ID:
                        self._toggle_dictation()
                        return True, 0
                    if msg.wParam == HOTKEY_COMPOSE_ID:
                        self._toggle_compose()
                        return True, 0
                    if msg.wParam == HOTKEY_SWITCH_MODEL_ID:
                        self._toggle_speech_model()
                        return True, 0
        except Exception:
            pass
        return super().nativeEvent(eventType, message)

    # -- window behaviour ---------------------------------------------------------
    def enterEvent(self, e):
        super().enterEvent(e)
        self._leave_timer.stop()
        if not self._hovered:
            self._hovered = True
            self._sync_island(animate=True)

    def leaveEvent(self, e):
        super().leaveEvent(e)
        if not self._menu_open:
            self._leave_timer.start()

    def _on_leave_timeout(self):
        if self._menu_open or self.drag_offset is not None:
            return
        if self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            return
        if self._hovered:
            self._hovered = False
            self._sync_island(animate=True)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.panel.isVisible() and self._morph_anim.state() != QVariantAnimation.Running:
            self._expanded_w = max(MIN_EXPANDED_W, self.width() - 2 * SHADOW_PAD_X)
            self._expanded_h = max(MIN_EXPANDED_H, self.height() - SHADOW_PAD_TOP - SHADOW_PAD_BOT)
        self._render_caption()

    def _save_island_pos(self):
        self.s.widget_pos = [int(self._anchor_cx), int(self._anchor_y), "island"]
        self.s.save()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_global = e.globalPosition().toPoint()
            self._drag_moved = False
            self.drag_offset = self._press_global - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self.drag_offset is not None and e.buttons() & Qt.LeftButton:
            gpos = e.globalPosition().toPoint()
            if self._press_global is not None and (gpos - self._press_global).manhattanLength() >= 5:
                self._drag_moved = True
            nx, ny = gpos.x() - self.drag_offset.x(), gpos.y() - self.drag_offset.y()
            # Magnetic snap back to top-center Apple Dynamic Island home
            scr = QGuiApplication.screenAt(gpos) or self.screen() or QGuiApplication.primaryScreen()
            geo = scr.availableGeometry()
            cx = nx + self.width() // 2
            top_y = ny + SHADOW_PAD_TOP
            if abs(cx - geo.center().x()) <= 36 and top_y <= geo.top() + 36:
                cx = geo.center().x()
                top_y = geo.top() + 8
                nx = int(round(cx - self.width() / 2.0))
                ny = top_y - SHADOW_PAD_TOP
            self.move(nx, ny)
            self._anchor_cx = cx
            self._anchor_y = top_y

    def mouseReleaseEvent(self, e):
        if self.drag_offset is not None:
            moved = self._drag_moved
            self.drag_offset = None
            self._press_global = None
            self._drag_moved = False
            if moved:
                self._save_island_pos()
            elif e.button() == Qt.LeftButton:
                # Single-click on the pill header smoothly expands/collapses the Dynamic Island
                local_y = e.position().y()
                if not self.panel.isVisible() or local_y <= (SHADOW_PAD_TOP + 38):
                    self.toggle_panel()

    def mouseDoubleClickEvent(self, e):
        pass

    def closeEvent(self, e):
        if not self.quitting:
            e.ignore()
            self.quit()
        else:
            e.accept()

    def quit(self):
        if self.quitting:
            return
        self.quitting = True
        self._save_island_pos()
        self.unregister_hotkey()
        if self.engine.busy:
            self.show()
            self._caption("Saving… finishing the last sentence and cleanup", "hint")
            self.engine.stop()
            self._wait_ticks = 0
            self._quit_timer = QTimer(self, interval=100, timeout=self._quit_when_idle)
            self._quit_timer.start()
        else:
            self._final_quit()

    def _quit_when_idle(self):
        self._wait_ticks += 1
        if not self.engine.busy or self._wait_ticks > 400:
            self._quit_timer.stop()
            self._final_quit()

    def _final_quit(self):
        if self.tray:
            self.tray.hide()
        QApplication.instance().quit()

    # -- startup ------------------------------------------------------------------
    def start_loading(self):
        first_run = not SETTINGS_FILE.exists()

        def load():
            try:
                self.engine.load()
            except Exception as ex:
                self.bridge.event.emit("log", ("error", f"Could not load the speech model: {ex}"))
                return
            if first_run and not self.s.resolved_api_key():
                self.bridge.event.emit("first_run", None)

        self.bridge.event.connect(lambda k, d: self.open_settings(first_run=True) if k == "first_run" else None)
        threading.Thread(target=load, daemon=True).start()
        if first_run:
            self.s.save()


def _redirect_streams():
    """pythonw has no console: send output to a log file instead of crashing."""
    if sys.stdout is None or sys.stderr is None:
        f = open(config_dir() / "eloquent.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f


def run_widget(engine, settings, start_minimized: bool = False):
    _redirect_streams()
    if sys.platform == "win32" and not single_instance("Eloquent.Widget"):
        # Already running: bring the existing window to the front instead of opening a second one
        u32 = ctypes.windll.user32
        hwnd = u32.FindWindowW(None, APP_NAME)
        if hwnd:
            u32.ShowWindow(hwnd, 9)  # SW_RESTORE
            u32.SetForegroundWindow(hwnd)
        return 0
    if sys.platform == "win32":
        try:  # own taskbar/tray identity instead of python.exe
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(f"{APP_NAME}.{__version__}")
        except Exception:
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(make_icon())
    app.setQuitOnLastWindowClosed(False)
    w = EloquentWidget(engine, settings)
    signal.signal(signal.SIGINT, lambda *_: w.quit())
    sig_timer = QTimer(w, interval=250, timeout=lambda: None)
    sig_timer.start()
    if start_minimized:
        w.showMinimized()
    else:
        w.show()
        w.raise_()
        w.activateWindow()
    w.register_hotkey()
    w.start_loading()
    return app.exec()
