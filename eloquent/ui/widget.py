"""Floating widget UI (PySide6).

Compact pill (always on top, draggable):  ● level   live caption   ▶/■   ⤢   ⋯
Expanded panel: polished transcript (fixes highlighted), show raw, copy, open folder.
Tray icon, optional global hotkey (Win32 RegisterHotKey) to start/stop.
"""

import ctypes
import html
import os
import signal
import sys
import threading
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

from PySide6.QtCore import QObject, QPoint, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QCursor, QDesktopServices, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                               QMessageBox, QPushButton, QSizeGrip, QSizePolicy, QSystemTrayIcon, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..audio import AUDIO_SOURCES, AUDIO_SOURCE_LABELS
from ..config import ROOT, SETTINGS_FILE, config_dir, is_arabic_speech, normalize_speech_language
from ..llm.prompts import REPORT_FORMATS
from ..paste import capture_selection_copy, get_external_foreground, play_cue, send_paste
from ..textutil import parse_custom_vocabulary

WM_HOTKEY = 0x0312
HOTKEY_ID = 1
HOTKEY_DICTATE_ID = 2
HOTKEY_COMPOSE_ID = 3
HOTKEY_SWITCH_MODEL_ID = 4
COMPACT_WIDTH = 660
PANEL_HEIGHT = 380

STATUS_COLORS = {
    "idle": "#8a8a8a",
    "loading": "#4aa3df",
    "listening": "#34c759",
    "dictating": "#8f8cff",
    "composing": "#38bdf8",
    "stopping": "#e0a526",
}

STYLE = """
#card { background: rgba(24, 24, 27, 240); border: 1px solid rgba(255,255,255,28); border-radius: 16px; }
QLabel { color: #ececec; font: 11pt "Segoe UI"; }
#caption[kind="partial"] { color: #9a9a9a; font-style: italic; }
#caption[kind="hint"] { color: #9a9a9a; }
#caption[kind="error"] { color: #ff8a80; }
#llm { color: #8f8f8f; font-size: 9pt; }
QToolButton { color: #ececec; background: transparent; border: none; border-radius: 8px;
              padding: 3px 7px; font: 11pt "Segoe UI Symbol"; }
QToolButton:hover { background: rgba(255,255,255,26); }
QToolButton:disabled { color: #555; }
#toggle { background: #2e9e5b; color: white; min-width: 30px; }
#toggle:hover { background: #36b368; }
#toggle[on="true"] { background: #d64545; }
#toggle[on="true"]:hover { background: #e85a5a; }
#toggle:disabled { background: #444; color: #888; }
QTextBrowser { background: transparent; border: none; color: #ececec; font: 11pt "Segoe UI";
               selection-background-color: #2b6cb0; }
QPushButton { color: #ececec; background: rgba(255,255,255,18); border: 1px solid rgba(255,255,255,30);
              border-radius: 7px; padding: 3px 10px; font: 9pt "Segoe UI"; }
QPushButton:hover { background: rgba(255,255,255,32); }
QCheckBox { color: #a8a8a8; font: 9pt "Segoe UI"; }
QMenu { background: #232326; color: #ececec; border: 1px solid #3a3a3e; padding: 4px; font: 10pt "Segoe UI"; }
QMenu::item { padding: 5px 22px 5px 22px; border-radius: 5px; }
QMenu::item:selected { background: #3a3a40; }
QMenu::separator { height: 1px; background: #3a3a3e; margin: 4px 8px; }
QScrollBar:vertical { background: transparent; width: 8px; }
QScrollBar::handle:vertical { background: rgba(255,255,255,40); border-radius: 4px; min-height: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
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
            head += '<span dir="ltr" style="color:#7db4ff;font-size:11px"> · polishing…</span>'
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
QDialog { background: #1e1e22; color: #ececec; font: 10pt "Segoe UI"; }
QLabel { color: #ececec; font: 10pt "Segoe UI"; }
QLabel#hint { color: #9a9aa2; font: 9pt "Segoe UI"; }
QLineEdit { background: #2a2a30; color: #f2f2f2; border: 1px solid #3f3f46; border-radius: 7px;
            padding: 6px 9px; font: 10pt "Segoe UI"; selection-background-color: #2b6cb0; }
QLineEdit:focus { border-color: #7db4ff; }
QPushButton { background: #2d2d34; color: #ececec; border: 1px solid #44444d; border-radius: 7px;
              padding: 5px 14px; font: 9.5pt "Segoe UI"; }
QPushButton:hover { background: #3a3a44; }
QPushButton#primary { background: #2e9e5b; border-color: #2e9e5b; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #36b368; }
QPushButton#subtle { background: transparent; border: none; color: #9a9aa2; padding: 5px 8px; }
QPushButton#subtle:hover { color: #ececec; background: rgba(255,255,255,14); }
"""


class _RenameSpeakerDialog(QDialog):
    """Compact dark-themed single-speaker rename dialog."""

    def __init__(self, spk_id: str, current_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Rename {spk_id}")
        self.setMinimumWidth(320)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        lbl = QLabel(f'<b style="color:{_speaker_color(spk_id)}">{html.escape(spk_id)}</b>')
        layout.addWidget(lbl)

        self.edit = QLineEdit("" if current_name == spk_id else current_name)
        self.edit.setPlaceholderText(f"Name for {spk_id}…")
        self.edit.returnPressed.connect(self.accept)
        layout.addWidget(self.edit)

        row = QHBoxLayout()
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

    def value(self) -> str:
        return self.edit.text().strip()


class RenameSpeakersDialog(QDialog):
    """Clean dark-themed dialog to rename all detected speakers at once."""

    def __init__(self, speakers: list[tuple[str, str]], edit_count: int = 0, parent=None):
        super().__init__(parent)
        self._special_action = ""  # "", "undo", "reset_edits"
        self.setWindowTitle(f"{APP_NAME} - Rename Speakers")
        self.setMinimumWidth(360)
        self.setStyleSheet(DARK_DIALOG_STYLE)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
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


class LevelDot(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(22, 22)
        self.color = QColor(STATUS_COLORS["idle"])
        self.level = 0.0

    def set_state(self, status: str, level: float = 0.0):
        self.color = QColor(STATUS_COLORS.get(status, "#8a8a8a"))
        self.level = level
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        c = self.rect().center()
        if self.level > 0:
            halo = QColor(self.color)
            halo.setAlpha(70)
            r = 5 + min(1.0, self.level * 14) * 6
            p.setBrush(halo)
            p.drawEllipse(c, r, r)
        p.setBrush(self.color)
        p.drawEllipse(c, 5, 5)


from .settings_dialog import SettingsDialog, VocabularyDialog  # noqa: E402


# --------------------------------------------------------------------------- #
# main widget
# --------------------------------------------------------------------------- #
class EloquentWidget(QWidget):
    def __init__(self, engine, settings):
        super().__init__()
        self.engine, self.s = engine, settings
        self.show_raw = False
        self.view_mode = "transcript"
        self.dirty = True
        self.drag_offset = None
        self.quitting = False
        self.hotkey_ok = False
        self.dictate_hotkey_ok = False
        self.compose_hotkey_ok = False
        self.switch_model_hotkey_ok = False
        self.dictate_target_hwnd = 0
        self._compose_starting = False
        self.caption_text, self.caption_kind = "Starting…", "hint"
        self.last_final = ""

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        flags = Qt.FramelessWindowHint | Qt.Window
        if settings.always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setStyleSheet(STYLE)
        self._build()

        self.bridge = Bridge()
        self.bridge.event.connect(self._on_event)
        engine.subscribe(lambda k, d: self.bridge.event.emit(k, d))

        self.level_timer = QTimer(self, interval=60, timeout=self._tick_level)
        self.rebuild_timer = QTimer(self, singleShot=True, interval=80, timeout=self._rebuild)
        self.caption_reset = QTimer(self, singleShot=True, interval=6000, timeout=self._idle_caption)

        self._place()
        self._tray()
        self._update_controls()

    # -- layout -------------------------------------------------------------
    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.card = QFrame(objectName="card")
        self.card.setLayoutDirection(Qt.LeftToRight)
        outer.addWidget(self.card)
        v = QVBoxLayout(self.card)
        v.setContentsMargins(12, 8, 10, 8)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.dot = LevelDot()
        self.caption = _CaptionLabel(on_resize=self._render_caption, objectName="caption")
        self.btn_toggle = QToolButton(objectName="toggle", text="▶")
        self.btn_toggle.clicked.connect(self._toggle_meeting)
        self.btn_expand = QToolButton(text="⤢", toolTip="Show transcript")
        self.btn_expand.clicked.connect(self.toggle_panel)
        self.btn_menu = QToolButton(text="⋯", toolTip="Menu")
        self.btn_menu.clicked.connect(self._show_menu)
        self.btn_close = QToolButton(text="✕", toolTip=f"Quit {APP_NAME}")
        self.btn_close.clicked.connect(self.quit)
        top.addWidget(self.dot)
        top.addWidget(self.caption, 1)
        top.addWidget(self.btn_toggle)
        top.addWidget(self.btn_expand)
        top.addWidget(self.btn_menu)
        top.addWidget(self.btn_close)
        v.addLayout(top)

        self.panel = QWidget()
        pv = QVBoxLayout(self.panel)
        pv.setContentsMargins(0, 4, 0, 0)
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background: rgba(255,255,255,25);")
        pv.addWidget(line)
        self.view = QTextBrowser()
        self.view.setOpenLinks(False)
        self.view.anchorClicked.connect(self._on_anchor)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_view_context_menu)
        self.view.setMinimumHeight(PANEL_HEIGHT - 60)
        self._sync_view_direction()
        pv.addWidget(self.view, 1)
        bottom = QHBoxLayout()
        self.chk_raw = QCheckBox("show raw")
        self.chk_raw.toggled.connect(self._set_raw)
        self.btn_back = QPushButton("← Transcript")
        self.btn_back.clicked.connect(self._switch_view)
        self.btn_back.hide()
        self.lbl_llm = QLabel(objectName="llm")
        self.lbl_llm.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_llm.setMinimumWidth(0)
        self.btn_gen = QPushButton("Generate ▾")
        self.btn_gen.clicked.connect(self._show_gen_menu)
        btn_copy = QPushButton("Copy")
        btn_copy.clicked.connect(self._copy)
        btn_folder = QPushButton("Open folder")
        btn_folder.clicked.connect(self._open_folder)
        bottom.addWidget(self.chk_raw)
        bottom.addWidget(self.btn_back)
        bottom.addWidget(self.lbl_llm, 1)
        bottom.addWidget(self.btn_gen)
        bottom.addWidget(btn_copy)
        bottom.addWidget(btn_folder)
        bottom.addWidget(QSizeGrip(self), 0, Qt.AlignBottom | Qt.AlignRight)
        pv.addLayout(bottom)
        self.panel.hide()
        v.addWidget(self.panel, 1)

        self.gen_menu = QMenu(self)
        self.act_rename = self.gen_menu.addAction("Rename speakers…", self.rename_speakers)
        self.act_rename_sep = self.gen_menu.addSeparator()
        for fmt in REPORT_FORMATS:
            self.gen_menu.addAction(fmt["label"], lambda fid=fmt["id"]: self._trigger_gen(fid))
        self.gen_menu.addSeparator()
        self.gen_menu.addAction("Custom Prompt…", self._trigger_custom_gen)

        self.setMinimumWidth(380)
        self.compact_height = self.sizeHint().height()
        self.setFixedHeight(self.compact_height)
        self.resize(COMPACT_WIDTH, self.compact_height)

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
        self.act_top = self.menu.addAction("Always on top")
        self.act_top.setCheckable(True)
        self.act_top.toggled.connect(self._set_on_top)
        self.menu.addSeparator()
        self.menu.addAction("Open sessions folder", lambda: self._open(self.s.resolved_sessions_dir()))
        self.menu.addAction("Clear Ask AI memory", self._clear_compose_history)
        self.menu.addAction("Create shortcuts + launch hotkey", self._make_shortcut)
        self.menu.addAction("Minimize", self.showMinimized)
        self.menu.addSeparator()
        self.menu.addAction(f"Quit {APP_NAME}", self.quit)

    def _place(self):
        pos = self.s.widget_pos
        if pos and QGuiApplication.screenAt(QPoint(pos[0] + 40, pos[1] + 20)) is not None:
            self.move(pos[0], pos[1])
        else:
            geo = QGuiApplication.primaryScreen().availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.top() + 40)

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
            self._mark_dirty()
        elif kind in ("paragraph", "reset", "speakers"):
            if kind == "reset":
                self.last_final = ""
                self.view_mode = "transcript"
            self._mark_dirty()
        elif kind == "report":
            if data and data.get("status") in ("generating", "done", "error"):
                self.view_mode = "report"
                if not self.panel.isVisible():
                    self.toggle_panel()
                if data["status"] == "generating":
                    self._caption(f"Generating {data['label']}…", "hint")
                elif data["status"] == "done":
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
            self._caption(self._listening_hint(mode), "hint")
        elif kind == "session_ended":
            if not self.quitting:
                mode = (data or {}).get("mode", "meeting")
                if mode in ("dictation", "compose"):
                    self._handle_dictation_ended(data)
                else:
                    self._caption("Saved. Open the transcript with ⤢ or the folder from ⋯", "hint")
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
            self._caption("Pasted!", "hint")
        else:
            if self.s.dictation_sound:
                play_cue("error")
            self._caption("Copied to clipboard (press Ctrl+V)", "hint")
        self.caption_reset.start()

    @staticmethod
    def _restore_clipboard(pasted_text: str, prev_text: str):
        clip = QGuiApplication.clipboard()
        if clip.text() == pasted_text:
            clip.setText(prev_text)

    def _caption(self, text: str, kind: str):
        self.caption_text = text
        if kind != self.caption_kind:
            self.caption_kind = kind
            self.caption.setProperty("kind", kind)
            self.caption.style().unpolish(self.caption)
            self.caption.style().polish(self.caption)
        self._render_caption()

    def _is_rtl_language(self) -> bool:
        return hasattr(self.s, "resolved_speech_language") and is_arabic_speech(self.s.resolved_speech_language())

    def _sync_view_direction(self):
        rtl = self._is_rtl_language()
        self.view.setLayoutDirection(Qt.RightToLeft if rtl else Qt.LeftToRight)

    def _render_caption(self):
        rtl_speech = self._is_rtl_language() and self.caption_kind in ("", "partial")
        self.caption.setLayoutDirection(Qt.RightToLeft if rtl_speech else Qt.LeftToRight)
        self.caption.setAlignment((Qt.AlignRight if rtl_speech else Qt.AlignLeft) | Qt.AlignVCenter)
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
        self._caption(" · ".join(parts) if parts else "Ready. Press ▶", "hint")

    def _update_controls(self):
        st = self.engine.status
        dot_st = self._dot_status()
        on = st == "listening"
        mode = self.engine.session_mode
        dictating = (mode == "dictation")
        composing = (mode == "compose")
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
        if on:
            self.level_timer.start()
        else:
            self.level_timer.stop()
            self.dot.set_state(dot_st)
        if st == "stopping":
            if composing:
                msg = "Generating & pasting…"
            elif dictating:
                msg = "Cleaning & pasting…"
            else:
                msg = "Finishing the last sentence…"
            self._caption(msg, "hint")

    def _tick_level(self):
        self.dot.set_state(self._dot_status(), self.engine.level)

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

    def _rebuild(self):
        if not self.panel.isVisible():
            return
        rtl = self._is_rtl_language()
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
                    f'<p dir="ltr" align="left" style="color:#7db4ff"><i>Generating {html.escape(rep["label"])}…</i></p>')
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
                self.btn_back.setText(f"View {rep['label']} →")
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
            sb.setValue(sb.maximum() if at_bottom else keep)
        self.dirty = False

    def _switch_view(self):
        self.view_mode = "transcript" if self.view_mode == "report" else "report"
        self._rebuild()

    def _show_gen_menu(self):
        multi = bool(self.engine.state.snapshot().get("multi_speaker")) or self.engine.speaker_edit_count() > 0
        self.act_rename.setVisible(multi)
        self.act_rename_sep.setVisible(multi)
        self.gen_menu.exec(self.btn_gen.mapToGlobal(QPoint(0, self.btn_gen.height())))

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
        dlg = QInputDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} - Correct Word")
        dlg.setLabelText(f'Correct "{selected}" to (also remembered for future transcripts):')
        dlg.setTextValue(selected)
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        dlg.setStyleSheet("")
        if not dlg.exec():
            return
        replacement = dlg.textValue().strip()
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
        dlg = QInputDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} - Custom Prompt")
        dlg.setLabelText("How should the LLM format or analyze this transcript?")
        dlg.setTextValue("Summarize the key points and list any action items.")
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        dlg.setStyleSheet("")
        if dlg.exec() and dlg.textValue().strip():
            self._trigger_gen("custom", dlg.textValue().strip())

    def toggle_panel(self):
        show = not self.panel.isVisible()
        self.panel.setVisible(show)
        self.btn_expand.setToolTip("Hide transcript" if show else "Show transcript")
        self.layout().activate()
        self.card.layout().activate()
        if show:
            self.setMinimumHeight(200)
            self.setMaximumHeight(16777215)
            self.resize(self.width(), PANEL_HEIGHT)
            self._rebuild()
            geo = self.screen().availableGeometry()
            if self.frameGeometry().bottom() > geo.bottom():
                self.move(self.x(), max(geo.top(), geo.bottom() - self.height()))
        else:
            self.setFixedHeight(self.compact_height)
            self.resize(self.width(), self.compact_height)

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
        self._open(self.engine.last_session_dir or self.s.resolved_sessions_dir())

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
        self.act_top.blockSignals(True)
        self.act_top.setChecked(self.s.always_on_top)
        self.act_top.blockSignals(False)
        self.menu.exec(self.btn_menu.mapToGlobal(QPoint(0, self.btn_menu.height())))

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
        dlg = SettingsDialog(self.s, self, first_run=first_run,
                             speech_busy=(self.engine.busy or self.engine.status == "loading"))
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        if dlg.exec():
            if normalize_speech_language(self.s.speech_language) != old_saved and hasattr(self.s, "set_active_speech_language"):
                self.s.set_active_speech_language(None)
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
    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_caption()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e):
        if self.drag_offset is not None and e.buttons() & Qt.LeftButton:
            self.move(e.globalPosition().toPoint() - self.drag_offset)

    def mouseReleaseEvent(self, e):
        if self.drag_offset is not None:
            self.drag_offset = None
            self.s.widget_pos = [self.x(), self.y()]
            self.s.save()

    def mouseDoubleClickEvent(self, e):
        self.toggle_panel()

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
        self.s.widget_pos = [self.x(), self.y()]
        self.s.save()
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
