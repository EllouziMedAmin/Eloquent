"""Session files: sessions/<date_time>/ raw.txt, clean.txt and (optional) audio.wav."""

import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from .speech import SAMPLE_RATE


class Session:
    def __init__(self, base: Path, save_audio: bool):
        self.dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.raw_f = open(self.dir / "raw.txt", "a", encoding="utf-8")
        self.clean_f = open(self.dir / "clean.txt", "a", encoding="utf-8")
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
