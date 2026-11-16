"""User settings, stored as JSON in %APPDATA%\\Eloquent\\settings.json."""

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from . import APP_NAME

ROOT = Path(__file__).resolve().parent.parent          # repo folder when run from source
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"

SPEECH_LANGUAGES = (
    ("parakeet", "25 European languages (Parakeet v3, Default)"),
    ("arabic", "Arabic & Dialects (Cohere, Vulkan / CPU)"),
)


def is_arabic_speech(value: Optional[str]) -> bool:
    """Return True when *value* selects the Arabic Cohere speech backend."""
    return str(value or "").strip().lower() in ("ar", "arabic", "cohere", "cohere-ar", "cohere-arabic")


def normalize_speech_language(value: Optional[str]) -> str:
    """Normalize a speech backend identifier to ``'parakeet'`` (25 European languages) or ``'arabic'``."""
    return "arabic" if is_arabic_speech(value) else "parakeet"


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
    speech_language: str = "parakeet"  # "parakeet" (25 European languages) | "arabic" (Cohere Arabic GGUF)
    threads: int = 0                  # 0 = auto
    diarize: bool = True              # background multi-speaker identification (Speaker 1, 2, ...)
    audio_source: str = "mic"         # "mic" | "system" | "both"
    device: Optional[int] = None      # None = system default microphone
    loopback_device: Optional[str] = None  # None = default Windows playback speakers
    silence: float = 0.5
    max_utterance: float = 8.0
    vad_threshold: float = 0.5
    partial_interval: float = 0.5
    ar_silence: float = 0.25          # tuned VAD silence pause (s) for Arabic Cohere
    ar_max_utterance: float = 4.0     # max chunk length (s) keeping Vulkan partials crisp
    ar_partial_interval: float = 0.8  # partial caption refresh interval (s) for Arabic Cohere
    # --- output ---
    save_audio: bool = False
    sessions_dir: str = ""            # empty -> <repo>/sessions
    models_dir: str = ""              # empty -> <repo>/models
    # --- interface ---
    ui: str = "widget"                # widget | terminal | web
    hotkey: str = "ctrl+shift+space"  # start/stop recording while the widget runs (empty = off)
    dictate_hotkey: str = "ctrl+alt+space"  # start/stop dictation & paste at cursor (empty = off)
    compose_hotkey: str = "ctrl+alt+a"      # start/stop Ask AI (speak prompt -> generate & paste, empty = off)
    switch_model_hotkey: str = "ctrl+alt+m" # quick toggle between Parakeet and Arabic (empty = off)
    launch_hotkey: str = "ctrl+shift+e"  # starts Eloquent from anywhere (background launcher)
    dictation_cleanup: bool = True    # clean dictated text with AI even if live transcript polishing is off
    compose_selection: bool = True    # read highlighted text on screen for Ask AI when available
    restore_clipboard: bool = True    # restore previous clipboard text after dictation paste
    dictation_sound: bool = True      # play subtle chimes on dictation start and paste
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
            s.speech_language = normalize_speech_language(s.speech_language)
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
    def __post_init__(self):
        # Transient in-session override toggled by the model-switch shortcut (never saved as default)
        object.__setattr__(self, "_active_speech_language", None)

    def set_active_speech_language(self, lang: Optional[str]):
        val = normalize_speech_language(lang) if lang else None
        object.__setattr__(self, "_active_speech_language", val)

    def resolved_speech_language(self) -> str:
        active = getattr(self, "_active_speech_language", None)
        val = active if active is not None else getattr(self, "speech_language", "parakeet")
        return normalize_speech_language(val)

    def resolved_timing(self) -> tuple[float, float, float]:
        """Return ``(silence, max_utterance, partial_interval)`` for the active speech backend."""
        if is_arabic_speech(self.resolved_speech_language()):
            return float(self.ar_silence), float(self.ar_max_utterance), float(self.ar_partial_interval)
        return float(self.silence), float(self.max_utterance), float(self.partial_interval)

    def resolved_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        # Diarizer runs at BELOW_NORMAL priority on 1 thread only between turns,
        # so Parakeet can use the full CPU allocation without hurting real-time latency.
        return default_threads()

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
