"""Eloquent settings dialog (own dark theme, grouped sections)."""

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QFileDialog, QFrame,
                               QHBoxLayout, QHeaderView, QKeySequenceEdit, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import APP_NAME
from ..audio import AUDIO_SOURCES, list_input_devices, list_loopback_devices
from ..config import SPEECH_LANGUAGES, config_dir, is_arabic_speech, normalize_speech_language
from ..hotkeys import parse_hotkey
from ..textutil import format_custom_vocabulary, parse_custom_vocabulary


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
SettingsDialog, VocabularyDialog { background: #18181b; }
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
QTableWidget {
    background: #26262b; color: #ececec; border: 1px solid #36363e; border-radius: 8px;
    gridline-color: #32323a; font: 10pt "Segoe UI"; selection-background-color: #3b3b8f;
}
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section {
    background: #2d2d34; color: #b4b4ba; border: none; border-bottom: 1px solid #3a3a42;
    padding: 5px 8px; font: 600 9pt "Segoe UI";
}
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


class VocabularyDialog(QDialog):
    """Guided editor with separate Names & Terms and Misheard -> Corrected lists."""

    def __init__(self, vocab_str: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Personal Dictionary")
        self.setMinimumSize(500, 480)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setStyleSheet(DIALOG_STYLE)

        terms, pairs = parse_custom_vocabulary(vocab_str)
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(12)

        root.addWidget(_label("Personal Dictionary", "title"))
        root.addWidget(_label(
            "Items below are only passed to the AI when your speech sounds similar, so normal sentences stay untouched.",
            "hint", wrap=True,
        ))

        # --- Section 1: Names & Terms ---
        root.addWidget(_label("NAMES & TERMS", "section"))
        names_card = QFrame(objectName="card")
        nc_lay = QVBoxLayout(names_card)
        nc_lay.setContentsMargins(14, 11, 14, 11)
        nc_lay.setSpacing(8)
        nc_lay.addWidget(_label(
            "Add your name, brands, or domain terms so they are spelled cleanly whenever mentioned.",
            "hint", wrap=True,
        ))

        n_row = QHBoxLayout()
        n_row.setSpacing(8)
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
        nh.setSectionResizeMode(0, QHeaderView.Stretch)
        nh.setSectionResizeMode(1, QHeaderView.Fixed)
        self.names_table.setColumnWidth(1, 42)
        self.names_table.setMinimumHeight(115)
        nc_lay.addWidget(self.names_table)
        root.addWidget(names_card, 1)

        # --- Section 2: Misheard -> Corrected ---
        root.addWidget(_label("WORD CORRECTIONS", "section"))
        pairs_card = QFrame(objectName="card")
        pc_lay = QVBoxLayout(pairs_card)
        pc_lay.setContentsMargins(14, 11, 14, 11)
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
        p_row.addWidget(_label("→"))
        p_row.addWidget(self.right_input, 1)
        p_row.addWidget(self.btn_add_pair)
        pc_lay.addLayout(p_row)

        self.pairs_table = QTableWidget(0, 3)
        self.pairs_table.setHorizontalHeaderLabels(["Misheard", "Corrected", ""])
        self.pairs_table.verticalHeader().setVisible(False)
        self.pairs_table.setSelectionMode(QAbstractItemView.NoSelection)
        ph = self.pairs_table.horizontalHeader()
        ph.setSectionResizeMode(0, QHeaderView.Stretch)
        ph.setSectionResizeMode(1, QHeaderView.Stretch)
        ph.setSectionResizeMode(2, QHeaderView.Fixed)
        self.pairs_table.setColumnWidth(2, 42)
        self.pairs_table.setMinimumHeight(130)
        pc_lay.addWidget(self.pairs_table)
        root.addWidget(pairs_card, 1)

        footer = QHBoxLayout()
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

    def _make_delete_btn(self, table: QTableWidget, del_col: int) -> QPushButton:
        btn = QPushButton("✕")
        btn.setObjectName("small")
        btn.setAutoDefault(False)
        btn.setFixedWidth(32)
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
        self.names_table.setRowHeight(row, 32)

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
        self.pairs_table.setRowHeight(row, 32)

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
    def __init__(self, settings, parent=None, first_run=False, speech_busy=False):
        super().__init__(parent)
        self.s = settings
        self._speech_busy = speech_busy
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

        self._vocab_value = getattr(s, "custom_vocabulary", "")
        self.vocab_summary = _label("", "hint", wrap=True)
        self.btn_vocab = QPushButton("Manage…")
        self.btn_vocab.setObjectName("small")
        self.btn_vocab.clicked.connect(self._open_vocab_dialog)
        v_row = QHBoxLayout()
        v_col = QVBoxLayout()
        v_col.setSpacing(2)
        v_col.addWidget(_label("Personal dictionary"))
        v_col.addWidget(self.vocab_summary)
        v_row.addLayout(v_col, 1)
        v_row.addWidget(self.btn_vocab, 0, Qt.AlignVCenter)
        lay.addLayout(v_row)
        self._sync_vocab_summary()
        return self._section("Transcript polishing", card)

    def _section_recording(self) -> QVBoxLayout:
        s = self.s
        card, lay = self._card()

        self.speech_lang = QComboBox()
        for key, label in SPEECH_LANGUAGES:
            self.speech_lang.addItem(label, key)
        cur_lang = normalize_speech_language(getattr(s, "speech_language", "parakeet"))
        self.speech_lang.setCurrentIndex(max(0, self.speech_lang.findData(cur_lang)))
        self.speech_lang.setEnabled(not self._speech_busy)
        self.speech_lang.currentIndexChanged.connect(self._sync_speech_language)
        self.speech_lang_hint = _label("", "hint", wrap=True)
        lay.addLayout(self._field("Default speech model", self.speech_lang))
        lay.addWidget(self.speech_lang_hint)
        self._sync_speech_language()

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

        self.max_speakers = QComboBox()
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
        ms_box = QWidget()
        ms_col = QVBoxLayout(ms_box)
        ms_col.setContentsMargins(0, 0, 0, 0)
        ms_col.setSpacing(5)
        ms_col.addWidget(_label("Maximum speakers"))
        ms_col.addWidget(self.max_speakers)
        ms_col.addWidget(_label(
            "Leave on Auto for unknown groups, or pick a fixed limit (e.g. 3) to prevent extra phantom speakers during noisy debates. Applies live.",
            "hint", wrap=True,
        ))
        lay.addWidget(ms_box)

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

    @staticmethod
    def _hotkey_row(title: str, value: str, hint: str) -> tuple[QKeySequenceEdit, QWidget]:
        edit = QKeySequenceEdit()
        edit.setMaximumSequenceLength(1)
        if value:
            edit.setKeySequence(QKeySequence(value))
        row = QHBoxLayout()
        clear = QPushButton("Clear")
        clear.setObjectName("small")
        clear.clicked.connect(edit.clear)
        row.addWidget(edit, 1)
        row.addWidget(clear)
        box = QWidget()
        col = QVBoxLayout(box)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(5)
        col.addWidget(_label(title))
        col.addLayout(row)
        col.addWidget(_label(hint, "hint", wrap=True))
        return edit, box

    def _section_shortcuts(self) -> QVBoxLayout:
        card, lay = self._card()
        self.hotkey, rec_box = self._hotkey_row(
            "Start / stop recording (notes & meetings)",
            self.s.hotkey,
            "Click the box and press the keys, for example Ctrl+Shift+Space. Leave empty to turn off.",
        )
        lay.addWidget(rec_box)

        self.dictate_hotkey, dict_box = self._hotkey_row(
            "Dictate & paste at cursor (any app)",
            self.s.dictate_hotkey,
            "Press once in ChatGPT, Reddit, or any text box to talk, then press again to clean and paste.",
        )
        lay.addWidget(dict_box)

        self.compose_hotkey, comp_box = self._hotkey_row(
            "Ask AI & paste at cursor (any app)",
            self.s.compose_hotkey,
            "Press once in any text box, speak a prompt (like writing an email), then press again to generate and paste.",
        )
        if hasattr(self.compose_hotkey, "keySequenceChanged"):
            self.compose_hotkey.keySequenceChanged.connect(self._sync_llm_enabled)
        elif hasattr(self.compose_hotkey, "textChanged"):
            self.compose_hotkey.textChanged.connect(self._sync_llm_enabled)
        lay.addWidget(comp_box)

        self.switch_model_hotkey, sw_box = self._hotkey_row(
            "Switch speech model (Parakeet ↔ Arabic)",
            self.s.switch_model_hotkey,
            "Toggle live between Parakeet (default) and Cohere Arabic without changing your startup default.",
        )
        lay.addWidget(sw_box)

        self.dictation_cleanup = QCheckBox("Clean dictated text with AI (dictation prompt)")
        self.dictation_cleanup.setChecked(self.s.dictation_cleanup)
        self.dictation_cleanup.toggled.connect(self._sync_llm_enabled)
        lay.addWidget(self.dictation_cleanup)

        self.compose_selection = QCheckBox("Read highlighted text for Ask AI when available")
        self.compose_selection.setChecked(self.s.compose_selection)
        lay.addWidget(self.compose_selection)

        self.restore_clipboard = QCheckBox("Restore previous clipboard after pasting")
        self.restore_clipboard.setChecked(self.s.restore_clipboard)
        lay.addWidget(self.restore_clipboard)

        self.dictation_sound = QCheckBox("Play soft sound cues for dictation and Ask AI")
        self.dictation_sound.setChecked(self.s.dictation_sound)
        lay.addWidget(self.dictation_sound)

        lay.addWidget(_label(f"Launch Eloquent from anywhere: {self.s.launch_hotkey.title()} "
                             "(set up by setup.bat).", "hint", wrap=True))
        return self._section("Shortcuts & dictation", card)

    # ------------------------------------------------------------------ #
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
                f"Default main model covering 25 European languages (fastest on CPU, word-level speaker splitting). Switch to Arabic anytime with {sw}."
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
        dhk = self._hotkey_text(self.dictate_hotkey)
        chk = self._hotkey_text(self.compose_hotkey)
        smhk = self._hotkey_text(self.switch_model_hotkey)
        for label, val in (
            ("recording hotkey", hk),
            ("dictation hotkey", dhk),
            ("Ask AI hotkey", chk),
            ("switch model hotkey", smhk),
        ):
            if val and parse_hotkey(val) is None:
                QMessageBox.warning(self, APP_NAME,
                                    f"That {label} can't be used. Include Ctrl, Alt, Shift or Win plus one key,\n"
                                    "for example Ctrl+Shift+Space, Ctrl+Alt+Space, Ctrl+Alt+A, or Ctrl+Alt+M.")
                return
        used = {}
        for name, val in (("Recording", hk), ("Dictation", dhk), ("Ask AI", chk), ("Switch model", smhk)):
            if not val:
                continue
            k = val.lower()
            if k in used:
                QMessageBox.warning(self, APP_NAME,
                                    f"{used[k]} and {name} shortcuts must be different.")
                return
            used[k] = name
        s = self.s
        s.gemini_api_key = self.key.text().strip()
        s.llm_model = self.model.text().strip() or s.llm_model
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
        s.dictate_hotkey = dhk
        s.compose_hotkey = chk
        s.switch_model_hotkey = smhk
        s.dictation_cleanup = self.dictation_cleanup.isChecked()
        s.compose_selection = self.compose_selection.isChecked()
        s.restore_clipboard = self.restore_clipboard.isChecked()
        s.dictation_sound = self.dictation_sound.isChecked()
        folder = self.folder.text().strip()
        s.sessions_dir = "" if folder == str(s.resolved_sessions_dir()) and not s.sessions_dir else folder
        s.save()
        self.accept()
