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
from PySide6.QtGui import QAction, QActionGroup, QColor, QDesktopServices, QFont, QGuiApplication, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMenu,
                               QMessageBox, QPushButton, QSizeGrip, QSystemTrayIcon, QTextBrowser, QToolButton,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..audio import AUDIO_SOURCES, AUDIO_SOURCE_LABELS
from ..config import SETTINGS_FILE, config_dir
from ..llm.prompts import REPORT_FORMATS

WM_HOTKEY = 0x0312
HOTKEY_ID = 1
COMPACT_WIDTH = 560
PANEL_HEIGHT = 380

STATUS_COLORS = {"idle": "#8a8a8a", "loading": "#4aa3df", "listening": "#34c759", "stopping": "#e0a526"}

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


def _speaker_badge(spk: str, esc, spk_id: str = "") -> str:
    if not spk:
        return ""
    key = spk_id or spk
    color = _speaker_color(key)
    href = urllib.parse.quote(key, safe="")
    return (f'<a href="spk:{href}" title="Click to rename {esc(spk)}" '
            f'style="color:{color};background:rgba(255,255,255,14);'
            f'border-radius:4px;padding:0 5px;font-size:10px;font-weight:600;'
            f'text-decoration:none">{esc(spk)}</a>&nbsp; ')


def transcript_html(snap: dict, show_raw: bool) -> str:
    esc = html.escape
    multi = bool(snap.get("multi_speaker"))
    out = []
    for p in snap["paragraphs"]:
        badge = _speaker_badge(p.get("speaker", ""), esc, p.get("speaker_id", "")) if multi else ""
        head = f'{badge}<span style="color:#7d7d7d;font-size:11px">{esc(p["time"])}</span>'
        if p["status"] == "cleaning":
            head += '<span style="color:#7db4ff;font-size:11px"> · polishing…</span>'
            body = f'<span style="color:#9a9a9a"><i>{esc(p["raw"])}</i></span>'
        elif p["tokens"] and not show_raw:
            body = " ".join(
                f'<span style="background-color:#5c4b00">{esc(w)}</span>' if f else esc(w)
                for w, f in p["tokens"])
        else:
            body = esc(p["raw"])
        note = (f'<br><span style="color:#e0a526;font-size:11px">{esc(p["note"])}</span>'
                if p["note"] else "")
        out.append(f'<p style="margin:0 0 12px 0">{head}<br>{body}{note}</p>')
    for x in snap["loose"]:
        badge = _speaker_badge(x.get("speaker", ""), esc, x.get("speaker_id", "")) if multi else ""
        out.append(f'<p style="margin:0 0 6px 0">{badge}<span style="color:#7d7d7d;font-size:11px">'
                   f'{esc(x["t"])}</span>&nbsp;&nbsp;{esc(x["text"])}</p>')
    if not out:
        out.append('<p style="color:#7d7d7d">The transcript will appear here.</p>')
    return "".join(out)


class RenameSpeakersDialog(QDialog):
    def __init__(self, speakers: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Rename Speakers")
        self.setMinimumWidth(340)
        self.setStyleSheet("")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        layout = QVBoxLayout(self)
        hint = QLabel("Assign names to speakers before generating notes or minutes.")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QFormLayout()
        self.inputs: dict[str, QLineEdit] = {}
        for cid, name in speakers:
            lbl = QLabel(f"<b>{html.escape(cid)}</b>")
            lbl.setStyleSheet(f"color: {_speaker_color(cid)};")
            edit = QLineEdit("" if name == cid else name)
            edit.setPlaceholderText(cid)
            form.addRow(lbl, edit)
            self.inputs[cid] = edit
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        return {cid: edit.text().strip() for cid, edit in self.inputs.items()}


class Bridge(QObject):
    """Carries engine events (any thread) to the GUI thread."""
    event = Signal(str, object)


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


from .settings_dialog import SettingsDialog  # noqa: E402


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
        outer.addWidget(self.card)
        v = QVBoxLayout(self.card)
        v.setContentsMargins(12, 8, 10, 8)
        v.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self.dot = LevelDot()
        self.caption = QLabel(objectName="caption")
        self.caption.setMinimumWidth(200)
        self.btn_toggle = QToolButton(objectName="toggle", text="▶")
        self.btn_toggle.clicked.connect(self.engine.toggle)
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
        self.view.setMinimumHeight(PANEL_HEIGHT - 60)
        pv.addWidget(self.view, 1)
        bottom = QHBoxLayout()
        self.chk_raw = QCheckBox("show raw")
        self.chk_raw.toggled.connect(self._set_raw)
        self.btn_back = QPushButton("← Transcript")
        self.btn_back.clicked.connect(self._switch_view)
        self.btn_back.hide()
        self.lbl_llm = QLabel(objectName="llm")
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
        src_menu = self.menu.addMenu("Audio source")
        self.src_group = QActionGroup(self)
        self.src_actions: dict[str, QAction] = {}
        for key, label in AUDIO_SOURCES:
            act = src_menu.addAction(label)
            act.setCheckable(True)
            act.triggered.connect(lambda _chk=False, k=key: self._set_audio_source(k))
            self.src_group.addAction(act)
            self.src_actions[key] = act
        self.act_audio = self.menu.addAction("Save audio recording")
        self.act_audio.setCheckable(True)
        self.act_audio.toggled.connect(self._set_audio)
        self.act_top = self.menu.addAction("Always on top")
        self.act_top.setCheckable(True)
        self.act_top.toggled.connect(self._set_on_top)
        self.menu.addSeparator()
        self.menu.addAction("Open sessions folder", lambda: self._open(self.s.resolved_sessions_dir()))
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
        self.tray_toggle = m.addAction("Start", self.engine.toggle)
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
                self._caption(self.last_final or "Listening…", "" if self.last_final else "hint")
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
            self._caption("Listening…", "hint")
        elif kind == "session_ended":
            if not self.quitting:
                self._caption("Saved. Open the transcript with ⤢ or the folder from ⋯", "hint")
                self.caption_reset.start()
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

    def _caption(self, text: str, kind: str):
        self.caption_text = text
        if kind != self.caption_kind:
            self.caption_kind = kind
            self.caption.setProperty("kind", kind)
            self.caption.style().unpolish(self.caption)
            self.caption.style().polish(self.caption)
        self._render_caption()

    def _render_caption(self):
        fm = self.caption.fontMetrics()
        self.caption.setText(fm.elidedText(self.caption_text, Qt.ElideLeft, max(50, self.caption.width() - 4)))
        self.caption.setToolTip(self.caption_text if len(self.caption_text) > 60 else "")

    def _idle_caption(self):
        e = self.engine
        if e.status != "idle":
            return
        hk = f" or {self.s.hotkey}" if self.hotkey_ok and self.s.hotkey else ""
        src_tag = {"system": " (System audio)", "both": " (Mic + System)"}.get(self.s.audio_source, "")
        msg = f"Ready{src_tag}. Press ▶{hk} to start"
        if e.llm["state"] == "off" and self.s.cleanup:
            msg += "  ·  no Gemini key, raw text only"
        self._caption(msg, "hint")

    def _update_controls(self):
        st = self.engine.status
        on = st == "listening"
        self.btn_toggle.setText("■" if on else "▶")
        self.btn_toggle.setEnabled(st in ("idle", "listening"))
        self.btn_toggle.setProperty("on", "true" if on else "false")
        self.btn_toggle.style().unpolish(self.btn_toggle)
        self.btn_toggle.style().polish(self.btn_toggle)
        hk = f" ({self.s.hotkey})" if self.hotkey_ok and self.s.hotkey else ""
        self.btn_toggle.setToolTip(("Stop" if on else "Start") + hk)
        if self.tray:
            self.tray_toggle.setText("Stop" if on else "Start")
            self.tray_toggle.setEnabled(st in ("idle", "listening"))
            self.tray.setToolTip(f"{APP_NAME} - {st}")
        if on:
            self.level_timer.start()
        else:
            self.level_timer.stop()
            self.dot.set_state(st)
        if st == "stopping":
            self._caption("Finishing the last sentence…", "hint")

    def _tick_level(self):
        self.dot.set_state("listening", self.engine.level)

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
                    f'<p style="color:#7db4ff"><i>Generating {html.escape(rep["label"])}…</i></p>')
            elif st == "error":
                self.view.setHtml(
                    f'<p style="color:#ff8a80"><b>{html.escape(rep["label"])}</b><br>'
                    f'{html.escape(rep.get("error", ""))}</p>')
            else:
                self.view.setMarkdown(f"# {rep['label']}\n\n{rep['text']}")
        else:
            self.view_mode = "transcript"
            self.chk_raw.show()
            if has_rep and rep.get("status") == "done":
                self.btn_back.setText(f"View {rep['label']} →")
                self.btn_back.show()
            else:
                self.btn_back.hide()
            sb = self.view.verticalScrollBar()
            at_bottom = sb.value() >= sb.maximum() - 30
            keep = sb.value()
            self.view.setHtml(transcript_html(snap, self.show_raw))
            sb.setValue(sb.maximum() if at_bottom else keep)
        self.dirty = False

    def _switch_view(self):
        self.view_mode = "transcript" if self.view_mode == "report" else "report"
        self._rebuild()

    def _show_gen_menu(self):
        multi = bool(self.engine.state.snapshot().get("multi_speaker"))
        self.act_rename.setVisible(multi)
        self.act_rename_sep.setVisible(multi)
        self.gen_menu.exec(self.btn_gen.mapToGlobal(QPoint(0, self.btn_gen.height())))

    def _on_anchor(self, url: QUrl):
        raw = url.toString()
        if not raw.startswith("spk:"):
            return
        spk_id = urllib.parse.unquote(raw[4:])
        cur = self.engine.state.display_speaker(spk_id) or spk_id
        dlg = QInputDialog(self)
        dlg.setWindowTitle(f"{APP_NAME} - Rename {spk_id}")
        dlg.setLabelText(f"Name for {spk_id}:")
        dlg.setTextValue("" if cur == spk_id else cur)
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        dlg.setStyleSheet("")
        if dlg.exec():
            self.engine.rename_speakers({spk_id: dlg.textValue().strip()})

    def rename_speakers(self):
        speakers = self.engine.state.speakers()
        if not speakers:
            return
        dlg = RenameSpeakersDialog(speakers, self)
        if dlg.exec():
            self.engine.rename_speakers(dlg.values())

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
        cur_src = self.s.audio_source if self.s.audio_source in self.src_actions else "mic"
        for k, act in self.src_actions.items():
            act.blockSignals(True)
            act.setChecked(k == cur_src)
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

    def open_settings(self, first_run=False):
        dlg = SettingsDialog(self.s, self, first_run=first_run)
        dlg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        if dlg.exec():
            self.register_hotkey()
            self.engine.refresh_provider()
            self.engine.sync_diarizer()
            self._update_llm_label()
            self._update_controls()
            if self.engine.status == "idle":
                self._idle_caption()

    def _toggle_visible(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    # -- global hotkey (Windows) -----------------------------------------------
    def register_hotkey(self):
        self.unregister_hotkey()
        if sys.platform != "win32" or not self.s.hotkey:
            return
        parsed = parse_hotkey(self.s.hotkey)
        if not parsed:
            return
        mods, vk = parsed
        self.hotkey_ok = bool(ctypes.windll.user32.RegisterHotKey(int(self.winId()), HOTKEY_ID, mods, vk))
        if not self.hotkey_ok:
            self._caption(f"Hotkey {self.s.hotkey} is used by another app (change it in Settings)", "error")
            self.caption_reset.start()

    def unregister_hotkey(self):
        if self.hotkey_ok:
            ctypes.windll.user32.UnregisterHotKey(int(self.winId()), HOTKEY_ID)
            self.hotkey_ok = False

    def nativeEvent(self, eventType, message):
        try:
            et = eventType.data() if hasattr(eventType, "data") else bytes(eventType)
            if et == b"windows_generic_MSG":
                msg = wintypes.MSG.from_address(int(message))
                if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                    self.engine.toggle()
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
