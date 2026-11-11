"""User settings, stored as JSON in %APPDATA%\\Eloquent\\settings.json."""

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from . import APP_NAME

ROOT = Path(__file__).resolve().parent.parent          # repo folder when run from source
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"


def config_dir() -> Path:
    base = os.getenv("APPDATA") or str(Path.home() / ".config")
    d = Path(base) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


SETTINGS_FILE = config_dir() / "settings.json"


def default_threads() -> int:
    return max(1, min(8, (os.cpu_count() or 4) // 2))


@dataclass
class Settings:
    # --- LLM cleanup ---
    cleanup: bool = True
    gemini_api_key: str = ""          # empty -> GEMINI_API_KEY / GOOGLE_API_KEY from env or .env
    llm_model: str = DEFAULT_LLM_MODEL
    keep_fillers: bool = False
    chunk_sentences: int = 3
    idle_flush: float = 6.0
    min_similarity: float = 0.5
    # --- speech ---
    threads: int = 0                  # 0 = auto
    diarize: bool = True              # background multi-speaker identification (Speaker 1, 2, ...)
    device: Optional[int] = None      # None = system default microphone
    silence: float = 0.5
    max_utterance: float = 15.0
    vad_threshold: float = 0.5
    partial_interval: float = 0.5
    # --- output ---
    save_audio: bool = False
    sessions_dir: str = ""            # empty -> <repo>/sessions
    models_dir: str = ""              # empty -> <repo>/models
    # --- interface ---
    ui: str = "widget"                # widget | terminal | web
    hotkey: str = "ctrl+shift+space"  # start/stop recording while the widget runs (empty = off)
    launch_hotkey: str = "ctrl+shift+e"  # starts Eloquent from anywhere (background launcher)
    always_on_top: bool = True
    widget_pos: Optional[list] = None
    web_port: int = 8765

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls) -> "Settings":
        s = cls()
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            names = {f.name for f in fields(cls)}
            for k, v in data.items():
                if k in names:
                    setattr(s, k, v)
        except FileNotFoundError:
            pass
        except Exception as e:  # corrupted file: keep defaults, do not crash
            print(f"[settings] could not read {SETTINGS_FILE}: {e}")
        return s

    def save(self):
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp.replace(SETTINGS_FILE)

    # ------------------------------------------------------------------ #
    def resolved_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        th = default_threads()
        # Reserve 1 core for background diarization so Parakeet never suffers thread contention
        return max(2, th - 1) if self.diarize and th >= 4 else th

    def resolved_sessions_dir(self) -> Path:
        return Path(self.sessions_dir) if self.sessions_dir else ROOT / "sessions"

    def resolved_models_dir(self) -> Path:
        return Path(self.models_dir) if self.models_dir else ROOT / "models"

    def resolved_api_key(self) -> str:
        if self.gemini_api_key.strip():
            return self.gemini_api_key.strip()
        try:
            from dotenv import load_dotenv
            load_dotenv(ROOT / ".env")
            load_dotenv(Path.cwd() / ".env")
        except ImportError:
            pass
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
