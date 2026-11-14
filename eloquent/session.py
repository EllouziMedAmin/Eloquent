"""Session files: sessions/<date_time>/ raw.txt, clean.txt and (optional) audio.wav."""

import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .speech import SAMPLE_RATE


def _format_session_texts(state) -> tuple[str, str]:
    multi, raw_items, clean_items = state.export_records()
    raw_lines = []
    for stamp, spk, txt in raw_items:
        prefix = f"[{stamp}] {spk}:" if (multi and spk) else f"[{stamp}]"
        raw_lines.append(f"{prefix} {txt}\n")
    clean_lines = []
    for stamp, spk, txt in clean_items:
        if multi and spk:
            clean_lines.append(f"[{stamp}] {spk}: {txt.strip()}\n\n")
        else:
            clean_lines.append(txt.strip() + "\n\n")
    return "".join(raw_lines), "".join(clean_lines)


def write_session_files(session_dir: Path, state) -> None:
    """Rewrite closed session raw.txt and clean.txt on disk (e.g. after speaker rename)."""
    if not session_dir.exists():
        return
    raw_body, clean_body = _format_session_texts(state)
    (session_dir / "raw.txt").write_text(raw_body, encoding="utf-8")
    (session_dir / "clean.txt").write_text(clean_body, encoding="utf-8")


class Session:
    def __init__(self, base: Path, save_audio: bool):
        self.dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.raw_f = open(self.dir / "raw.txt", "w", encoding="utf-8")
        self.clean_f = open(self.dir / "clean.txt", "w", encoding="utf-8")
        self.wav = None
        if save_audio:
            self.wav = wave.open(str(self.dir / "audio.wav"), "wb")
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)
            self.wav.setframerate(SAMPLE_RATE)

    def add_raw(self, stamp: str, text: str):
        with self._lock:
            if not self.raw_f.closed:
                self.raw_f.write(f"[{stamp}] {text}\n")
                self.raw_f.flush()

    def add_clean(self, text: str):
        with self._lock:
            if not self.clean_f.closed:
                self.clean_f.write(text.strip() + "\n\n")
                self.clean_f.flush()

    def sync_from_state(self, state):
        """Rewrite raw.txt and clean.txt with finalized multi-speaker splits and labels."""
        raw_body, clean_body = _format_session_texts(state)
        with self._lock:
            if not self.raw_f.closed:
                self.raw_f.seek(0)
                self.raw_f.truncate()
                self.raw_f.write(raw_body)
                self.raw_f.flush()
            if not self.clean_f.closed:
                self.clean_f.seek(0)
                self.clean_f.truncate()
                self.clean_f.write(clean_body)
                self.clean_f.flush()

    def write_audio(self, frame: np.ndarray):
        if self.wav is not None:
            self.wav.writeframes((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())

    def close(self):
        with self._lock:
            for f in (self.raw_f, self.clean_f):
                f.close()
            if self.wav is not None:
                self.wav.close()
                self.wav = None
