"""Eloquent settings dialog (own dark theme, grouped sections)."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QKeySequenceEdit,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget)

from .. import APP_NAME
from ..audio import AUDIO_SOURCES, list_input_devices, list_loopback_devices
from ..config import config_dir
from ..hotkeys import parse_hotkey


def _check_icon_path() -> str:
    """White tick used inside checked checkboxes (drawn once, cached next to the settings)."""
    path = config_dir() / "check.png"
    if not path.exists():
        pm = QPixmap(36, 36)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("white"), 4.5)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.drawPolyline([QPoint(9, 19), QPoint(15, 25), QPoint(27, 11)])
        p.end()
        pm.save(str(path), "PNG")
    return str(path).replace("\\", "/")


DIALOG_STYLE = """
SettingsDialog { background: #18181b; }
QWidget#body { background: transparent; }
QScrollArea { background: transparent; border: none; }
QLabel { color: #d8d8dc; font: 10pt "Segoe UI"; background: transparent; }
QLabel#title { color: #ffffff; font: 600 15pt "Segoe UI"; }
QLabel#intro { color: #b4b4ba; font: 10pt "Segoe UI"; }
QLabel#section { color: #8f8cff; font: 600 9pt "Segoe UI"; letter-spacing: 1px; }
QLabel#hint { color: #8a8a92; font: 9pt "Segoe UI"; }
QLabel#hint a, QLabel a { color: #8fb4ff; text-decoration: none; }
QFrame#card { background: #222226; border: 1px solid #2f2f35; border-radius: 12px; }
QLineEdit, QComboBox, QKeySequenceEdit {
    background: #2a2a2f; color: #ececec; border: 1px solid #3a3a41; border-radius: 8px;
    padding: 5px 10px; font: 10pt "Segoe UI"; selection-background-color: #4b5bd6;
}
QLineEdit:focus, QComboBox:focus, QKeySequenceEdit:focus { border: 1px solid #7c7cff; }
QLineEdit:disabled, QComboBox:disabled { color: #6a6a72; background: #232327; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #2a2a2f; color: #ececec; border: 1px solid #3a3a41;
                              selection-background-color: #3b3b8f; outline: 0; }
QCheckBox { color: #d8d8dc; font: 10pt "Segoe UI"; spacing: 10px; background: transparent; }
QCheckBox::indicator { width: 18px; height: 18px; border-radius: 5px; border: 1px solid #55555e; background: #2a2a2f; }
QCheckBox::indicator:hover { border-color: #7c7cff; }
QCheckBox::indicator:checked { background: #6c6cff; border-color: #6c6cff; image: none; }
QPushButton { color: #ececec; background: #2f2f36; border: 1px solid #40404a; border-radius: 8px;
              padding: 7px 16px; font: 10pt "Segoe UI"; }
QPushButton:hover { background: #3a3a43; }
QPushButton#primary { background: #5b5bf0; border: 1px solid #5b5bf0; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #6d6df5; }
QPushButton#small { padding: 6px 12px; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 2px; }
QScrollBar::handle:vertical { background: #44444c; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
"""


def _label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    lb = QLabel(text)
    if name:
        lb.setObjectName(name)
    lb.setWordWrap(wrap)
    pal = lb.palette()
    pal.setColor(QPalette.Link, QColor("#8fb4ff"))
    pal.setColor(QPalette.LinkVisited, QColor("#8fb4ff"))
    lb.setPalette(pal)
    return lb


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None, first_run=False):
        super().__init__(parent)
        self.s = settings
        self._positioned = False
        self.setWindowTitle(f"{APP_NAME} settings")
        self.setMinimumSize(460, 380)
        self.setSizeGripEnabled(True)
        self.setStyleSheet(DIALOG_STYLE.replace("image: none;", f"image: url({_check_icon_path()});"))

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        root.addWidget(_label("Settings", "title"))
        if first_run:
            root.addWidget(_label(
                "Welcome to Eloquent! Speech recognition runs locally on your CPU. "
                "Optionally add a free Gemini API key to polish the transcript "
                "(fix misheard words and punctuation).", "intro", wrap=True))

        body = QWidget(objectName="body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 6, 0)
        bl.setSpacing(12)
        bl.addLayout(self._section_polish())
        bl.addLayout(self._section_recording())
        bl.addLayout(self._section_shortcuts())
        bl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        footer = QHBoxLayout()
        footer.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save")
        save.setObjectName("primary")
        save.setDefault(True)
        save.clicked.connect(self._save)
        footer.addWidget(cancel)
        footer.addWidget(save)
        root.addLayout(footer)

        self._sync_llm_enabled()
        screen = (parent.screen() if parent else None) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        max_w = max(460, avail.width() - 40) if avail else 640
        max_h = max(380, int(avail.height() * 0.72)) if avail else 560
        default_h = 560 if first_run else 520
        self.resize(min(520, max_w), max(380, min(body.sizeHint().height() + (190 if first_run else 140), default_h, max_h)))

    def showEvent(self, event):
        super().showEvent(event)
        if self._positioned:
            return
        self._positioned = True
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
        lay.setContentsMargins(14, 11, 14, 11)
        lay.setSpacing(8)
        return card, lay

    @staticmethod
    def _field(title: str, widget: QWidget, hint: str = "") -> QVBoxLayout:
        col = QVBoxLayout()
        col.setSpacing(4)
        col.addWidget(_label(title))
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
    def _section_polish(self) -> QVBoxLayout:
        s = self.s
        card, lay = self._card()

        self.cleanup = QCheckBox("Polish the transcript with an LLM")
        self.cleanup.setChecked(s.cleanup)
        self.cleanup.toggled.connect(self._sync_llm_enabled)
        lay.addWidget(self.cleanup)

        key_row = QHBoxLayout()
        self.key = QLineEdit(s.gemini_api_key)
        self.key.setEchoMode(QLineEdit.Password)
        self.key.setPlaceholderText("Paste your key (or leave empty to use .env)")
        self.btn_show = QPushButton("Show")
        self.btn_show.setObjectName("small")
        self.btn_show.setCheckable(True)
        self.btn_show.toggled.connect(self._toggle_key)
        key_row.addWidget(self.key, 1)
        key_row.addWidget(self.btn_show)
        key_box = QWidget()
        key_col = QVBoxLayout(key_box)
        key_col.setContentsMargins(0, 0, 0, 0)
        key_col.setSpacing(5)
        key_col.addWidget(_label("Gemini API key"))
        key_col.addLayout(key_row)
        link = _label('<a href="https://aistudio.google.com/apikey" style="text-decoration:none">'
                      '<span style="color:#8fb4ff">Get a free key from Google AI Studio  ↗</span></a>', "hint")
        link.setOpenExternalLinks(True)
        key_col.addWidget(link)
        self.key_box = key_box
        lay.addWidget(key_box)

        self.model = QLineEdit(s.llm_model)
        self.model_box = QWidget()
        mb = QVBoxLayout(self.model_box)
        mb.setContentsMargins(0, 0, 0, 0)
        mb.addLayout(self._field("Model", self.model))
        lay.addWidget(self.model_box)

        self.fillers = QCheckBox("Keep filler words (uh, um)")
        self.fillers.setChecked(s.keep_fillers)
        lay.addWidget(self.fillers)
        return self._section("Transcript polishing", card)

    def _section_recording(self) -> QVBoxLayout:
        s = self.s
        card, lay = self._card()

        self.source = QComboBox()
        for key, label in AUDIO_SOURCES:
            self.source.addItem(label, key)
        self.source.setCurrentIndex(max(0, self.source.findData(s.audio_source)))
        self.source.currentIndexChanged.connect(self._sync_audio_source)
        self.source_hint = _label("", "hint", wrap=True)
        lay.addLayout(self._field("Audio source", self.source))
        lay.addWidget(self.source_hint)

        self.mic = QComboBox()
        self.mic.addItem("System default", None)
        for i, name in list_input_devices():
            self.mic.addItem(name, i)
        self.mic.setCurrentIndex(max(0, self.mic.findData(s.device)))
        lay.addLayout(self._field("Microphone", self.mic))

        self.loopback = QComboBox()
        self.loopback.addItem("Default playback speakers", None)
        for _, name, is_def in list_loopback_devices():
            self.loopback.addItem(f"{name} (default)" if is_def else name, name)
        if s.loopback_device and self.loopback.findData(s.loopback_device) < 0:
            self.loopback.addItem(s.loopback_device, s.loopback_device)
        self.loopback.setCurrentIndex(max(0, self.loopback.findData(s.loopback_device)))
        lay.addLayout(self._field("System output (loopback)", self.loopback))
        self._sync_audio_source()

        self.audio = QCheckBox("Save the audio recording (audio.wav, about 115 MB per hour)")
        self.audio.setChecked(s.save_audio)
        lay.addWidget(self.audio)

        self.diarize = QCheckBox("Identify speakers (Speaker 1, Speaker 2, …)")
        self.diarize.setChecked(s.diarize)
        lay.addWidget(self.diarize)
        lay.addWidget(_label("Labels speakers on 1 background thread so live caption speed is unaffected.",
                             "hint", wrap=True))

        row = QHBoxLayout()
        self.folder = QLineEdit(str(s.resolved_sessions_dir()))
        browse = QPushButton("Browse…")
        browse.setObjectName("small")
        browse.clicked.connect(self._browse)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)
        col.addWidget(_label("Sessions folder"))
        col.addLayout(row)
        lay.addWidget(box)
        return self._section("Recording & storage", card)

    def _section_shortcuts(self) -> QVBoxLayout:
        card, lay = self._card()
        self.hotkey = QKeySequenceEdit()
        self.hotkey.setMaximumSequenceLength(1)
        if self.s.hotkey:
            self.hotkey.setKeySequence(QKeySequence(self.s.hotkey))
        row = QHBoxLayout()
        clear = QPushButton("Clear")
        clear.setObjectName("small")
        clear.clicked.connect(self.hotkey.clear)
        row.addWidget(self.hotkey, 1)
        row.addWidget(clear)
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)
        col.addWidget(_label("Start / stop recording"))
        col.addLayout(row)
        col.addWidget(_label("Click the box and press the keys, for example Ctrl+Shift+Space. "
                             "Leave empty to turn the hotkey off.", "hint", wrap=True))
        lay.addWidget(box)
        lay.addWidget(_label(f"Launch Eloquent from anywhere: {self.s.launch_hotkey.title()} "
                             "(set up by setup.bat).", "hint", wrap=True))
        return self._section("Shortcuts", card)

    # ------------------------------------------------------------------ #
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

    def _sync_llm_enabled(self):
        on = self.cleanup.isChecked()
        for w in (self.key_box, self.model_box, self.fillers):
            w.setEnabled(on)

    def _toggle_key(self, show: bool):
        self.key.setEchoMode(QLineEdit.Normal if show else QLineEdit.Password)
        self.btn_show.setText("Hide" if show else "Show")

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Sessions folder", self.folder.text())
        if d:
            self.folder.setText(d)

    def _hotkey_text(self) -> str:
        seq = self.hotkey.keySequence()
        if seq.isEmpty():
            return ""
        return seq.toString(QKeySequence.PortableText).lower().replace("meta", "win")

    def _save(self):
        hk = self._hotkey_text()
        if hk and parse_hotkey(hk) is None:
            QMessageBox.warning(self, APP_NAME,
                                "That hotkey can't be used. Include Ctrl, Alt, Shift or Win plus one key,\n"
                                "for example Ctrl+Shift+Space.")
            return
        s = self.s
        s.gemini_api_key = self.key.text().strip()
        s.llm_model = self.model.text().strip() or s.llm_model
        s.cleanup = self.cleanup.isChecked()
        s.keep_fillers = self.fillers.isChecked()
        s.save_audio = self.audio.isChecked()
        s.diarize = self.diarize.isChecked()
        s.audio_source = self.source.currentData() or "mic"
        s.device = self.mic.currentData()
        s.loopback_device = self.loopback.currentData()
        s.hotkey = hk
        folder = self.folder.text().strip()
        s.sessions_dir = "" if folder == str(s.resolved_sessions_dir()) and not s.sessions_dir else folder
        s.save()
        self.accept()
