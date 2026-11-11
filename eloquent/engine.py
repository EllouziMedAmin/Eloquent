"""The Eloquent engine: microphone (or WAV file) -> Silero VAD -> Parakeet -> transcript state.

UIs never touch audio. They subscribe to events and read `engine.state`:

    engine.subscribe(lambda kind, data: ...)

Event kinds
    status          "loading" | "idle" | "listening" | "stopping"
    llm             {"state": "off"|"checking"|"ready"|"error", "label", "message"}
    log             (level, message)              level: info | warn | error
    session_started {"dir": str}
    session_ended   {"dir": str}
    reset / partial / segment / paragraph / error   (from State)

Models are loaded once; start()/stop() only open/close the mic and the session,
so a new recording starts instantly.
"""

import collections
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .cleaner import Cleaner
from .diarizer import Diarizer, build_speaker_extractor, ensure_speaker_model
from .llm import make_provider
from .llm.prompts import build_format_instruction
from .session import Session
from .speech import (SAMPLE_RATE, WINDOW, build_recognizer, build_vad, ensure_asr_model,
                     ensure_vad_model, read_wav_16k, transcribe, transcribe_detailed)
from .state import State, fmt_time


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.state = State()
        self.state.emit = self._emit
        self.status = "idle"
        self.llm = {"state": "off", "label": "", "message": ""}
        self.level = 0.0                 # mic RMS of the latest frame (for UI meters)
        self.session = None
        self.last_session_dir = None
        self.recognizer = None
        self.vad_path = None
        self.extractor = None
        self.diarizer = None
        self.provider = None
        self._diar_lock = threading.Lock()
        self._provider_key = None
        self._listeners = []
        self._stop = threading.Event()
        self._worker = None
        self._loaded = False

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def subscribe(self, fn):
        self._listeners.append(fn)

    def _emit(self, kind, data=None):
        for fn in list(self._listeners):
            try:
                fn(kind, data)
            except Exception as e:  # a broken UI must never stop transcription
                print(f"[engine] listener error on {kind}: {e}")

    def log(self, msg: str, level: str = "info"):
        self._emit("log", (level, msg))

    def _set_status(self, status: str):
        self.status = status
        self._emit("status", status)

    def _set_llm(self, state: str, message: str = ""):
        self.llm = {"state": state, "label": self.provider.label() if self.provider else "",
                    "message": message}
        self._emit("llm", self.llm)

    # ------------------------------------------------------------------ #
    # setup
    # ------------------------------------------------------------------ #
    def load(self):
        """Download (first run) and load the speech models. Blocking; call from a worker thread in GUIs."""
        if self._loaded:
            return
        self._set_status("loading")
        s = self.settings
        models = s.resolved_models_dir()
        model_dir = ensure_asr_model(models, self.log)
        self.vad_path = ensure_vad_model(models, self.log)
        threads = s.resolved_threads()
        self.log(f"Loading Parakeet on CPU with {threads} threads...")
        t0 = time.time()
        self.recognizer = build_recognizer(model_dir, threads)
        self._loaded = True
        self.log(f"Speech model ready in {time.time() - t0:.1f}s")
        if s.diarize:
            self._load_speaker_extractor(models)
        self.refresh_provider()
        self._set_status("idle")

    def _load_speaker_extractor(self, models_dir: Path = None):
        with self._diar_lock:
            if self.extractor is not None:
                return
            try:
                mdir = models_dir or self.settings.resolved_models_dir()
                spk_path = ensure_speaker_model(mdir, self.log)
                self.extractor = build_speaker_extractor(spk_path, threads=1)
                self.log("[diarize] Multi-speaker identification ready (1 background thread)")
            except Exception as e:
                self.log(f"[diarize] Disabled ({type(e).__name__}: {e})", "warn")

    def sync_diarizer(self):
        """Lazy-load speaker extractor in the background if diarization was toggled ON after startup."""
        if self.settings.diarize and self._loaded and self.extractor is None:
            threading.Thread(target=self._load_speaker_extractor, daemon=True, name="eloquent-diar-load").start()

    def refresh_provider(self, force: bool = False):
        """(Re)create the LLM provider when its settings changed. The key check runs in the background."""
        s = self.settings
        key = (s.cleanup, s.resolved_api_key(), s.llm_model, s.keep_fillers)
        if key == self._provider_key and not force:
            return
        self._provider_key = key
        self.provider, reason = make_provider(s)
        if self.provider is None:
            self._set_llm("off", reason)
            if s.cleanup:
                self.log(f"[clean] {reason}", "warn")
            return
        self._set_llm("checking")
        provider = self.provider

        def check():
            try:
                provider.check()
                if provider is self.provider:
                    self._set_llm("ready")
                    self.log(f"[clean] {provider.label()} ready")
            except Exception as e:
                if provider is self.provider:
                    msg = f"{type(e).__name__}: {str(e)[:140]}"
                    self._set_llm("error", msg)
                    self.log(f"[clean] check failed ({msg}). Raw text is kept when cleanup fails.", "warn")

        threading.Thread(target=check, daemon=True).start()

    # ------------------------------------------------------------------ #
    # start / stop
    # ------------------------------------------------------------------ #
    @property
    def busy(self) -> bool:
        return self.status in ("listening", "stopping")

    def start(self, wav_path: Path = None) -> bool:
        if not self._loaded or self.status != "idle":
            return False
        s = self.settings
        self.refresh_provider()
        self.state.reset()
        self.session = Session(s.resolved_sessions_dir(), save_audio=s.save_audio)
        self.last_session_dir = self.session.dir
        self.vad = build_vad(self.vad_path, s.vad_threshold, s.silence, s.max_utterance)
        if s.diarize and self.extractor is None:
            self._load_speaker_extractor()
        self.diarizer = Diarizer(self.extractor, self.state, self.session, log=self.log) if (
            s.diarize and self.extractor is not None
        ) else None
        if self.diarizer is not None:
            self.diarizer.start()
        self.cleaner = Cleaner(self.state, self.session, self.provider, s, log=self.log)
        self.cleaner.start()
        # per-session pipeline state
        self._buf = collections.deque()
        self._buf_len = 0
        self._since_partial = 0
        self._last_partial = ""
        self._samples = 0
        self._preroll = int(0.3 * SAMPLE_RATE)
        self._partial_every = int(s.partial_interval * SAMPLE_RATE)
        self._stop.clear()
        self._set_status("listening")
        self._emit("session_started", {"dir": str(self.session.dir)})
        self._worker = threading.Thread(target=self._run, args=(wav_path,), daemon=True,
                                        name="eloquent-audio")
        self._worker.start()
        return True

    def stop(self):
        """Ask the audio thread to stop. Returns immediately; status goes stopping -> idle."""
        if self.status == "listening":
            self._stop.set()

    def toggle(self):
        if self.status == "idle":
            self.start()
        elif self.status == "listening":
            self.stop()

    def wait_idle(self, timeout: float = None) -> bool:
        if self._worker is not None:
            self._worker.join(timeout)
            return not self._worker.is_alive()
        return True

    # ------------------------------------------------------------------ #
    # audio thread
    # ------------------------------------------------------------------ #
    def _run(self, wav_path):
        try:
            if wav_path is None:
                self._loop_mic()
            else:
                self._loop_file(Path(wav_path))
        except Exception as e:
            self.log(f"Audio error: {type(e).__name__}: {e}", "error")
        finally:
            self._finish()

    def _loop_mic(self):
        import sounddevice as sd
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(indata[:, 0].copy())

        pending = np.zeros(0, dtype=np.float32)
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE,
                            blocksize=WINDOW, device=self.settings.device, callback=callback):
            while not self._stop.is_set():
                try:
                    chunk = q.get(timeout=0.2)
                except queue.Empty:
                    continue
                pending = np.concatenate([pending, chunk])
                while len(pending) >= WINDOW:
                    frame, pending = pending[:WINDOW], pending[WINDOW:]
                    # skip live partials when we are falling behind: finals always win
                    self._process(frame, allow_partial=q.qsize() < 8)

    def _loop_file(self, path: Path):
        """Feed a recording through the exact same pipeline, as fast as the CPU allows."""
        self.log(f"Transcribing file: {path}")
        audio = read_wav_16k(path)
        for i in range(0, len(audio) - WINDOW + 1, WINDOW):
            if self._stop.is_set():
                break
            self._process(audio[i:i + WINDOW], allow_partial=False)

    def _process(self, frame: np.ndarray, allow_partial: bool):
        self.session.write_audio(frame)
        self._samples += len(frame)
        self.level = float(np.sqrt(np.mean(frame * frame)))
        vad = self.vad
        vad.accept_waveform(frame)
        self._buf.append(frame)
        self._buf_len += len(frame)
        speaking = vad.is_speech_detected()

        if not speaking and vad.empty():
            # keep a short pre-roll so the first syllable is not cut
            while self._buf_len - len(self._buf[0]) >= self._preroll and len(self._buf) > 1:
                self._buf_len -= len(self._buf.popleft())
            return

        self._since_partial += len(frame)
        if speaking and allow_partial and self._since_partial >= self._partial_every:
            self._since_partial = 0
            text = transcribe(self.recognizer, np.concatenate(self._buf))
            if text and text != self._last_partial:
                self._last_partial = text
                self.state.set_partial(text)

        self._drain()

    def _drain(self):
        """Transcribe every sentence the VAD has finished."""
        vad = self.vad
        while not vad.empty():
            seg = np.asarray(vad.front.samples, dtype=np.float32)
            vad.pop()
            t1 = time.time()
            text, words = transcribe_detailed(self.recognizer, seg)
            dur = len(seg) / SAMPLE_RATE
            decode = time.time() - t1
            if text:
                t0 = max(0.0, self._samples / SAMPLE_RATE - dur)
                stamp = fmt_time(t0)
                self.session.add_raw(stamp, text)
                seg_id = self.state.add_segment(
                    text, stamp, {"audio_sec": dur, "decode_sec": decode},
                    start_sec=t0, spk_ready=(self.diarizer is None),
                )
                if self.diarizer is not None:
                    self.diarizer.submit(seg_id, t0, stamp, seg, text, words)
            else:
                self.state.set_partial("")
            self._buf.clear()
            self._buf_len = 0
            self._since_partial = 0
            self._last_partial = ""

    def _finish(self):
        """Finish the sentence in progress, then diarization & cleanup, then close the files."""
        self._set_status("stopping")
        self.level = 0.0
        try:
            self.vad.flush()
            self._drain()
        except Exception as e:
            self.log(f"Flush error: {e}", "error")
        self.state.set_partial("")
        if self.diarizer is not None:
            self.diarizer.flush_and_wait()
        self.cleaner.flush_and_wait()
        if self.diarizer is not None:
            self.session.sync_from_state(self.state)
        self.diarizer = None
        self.session.close()
        d = str(self.session.dir)
        self._set_status("idle")
        self._emit("session_ended", {"dir": d})

    # ------------------------------------------------------------------ #
    # post-session document generation
    # ------------------------------------------------------------------ #
    def generate_report(self, format_id: str, custom_prompt: str = "", blocking: bool = False):
        """Format or summarize the current transcript with the LLM and save to the session folder."""
        text = self.state.full_text(raw=False).strip()
        if not text:
            self.state.set_report({
                "format_id": format_id, "label": "No transcript",
                "status": "error", "text": "", "file": "",
                "error": "No transcript text to format yet.",
            })
            return False

        provider = self.provider
        if provider is None:
            provider, reason = make_provider(self.settings, force=True)
            if provider is None:
                self.state.set_report({
                    "format_id": format_id, "label": "LLM unavailable",
                    "status": "error", "text": "", "file": "", "error": reason,
                })
                return False

        label, filename, instruction = build_format_instruction(format_id, custom_prompt)
        self.state.set_report({
            "format_id": format_id, "label": label,
            "status": "generating", "text": "", "file": "", "error": "",
        })
        self.log(f"Generating {label}...")

        def work():
            try:
                out = provider.generate(instruction, text)
                if not out:
                    raise RuntimeError("Empty response from LLM")
                out_path = ""
                if self.last_session_dir:
                    p = Path(self.last_session_dir) / filename
                    p.write_text(out + "\n", encoding="utf-8")
                    out_path = str(p)
                self.state.set_report({
                    "format_id": format_id, "label": label,
                    "status": "done", "text": out, "file": out_path, "error": "",
                })
                self.log(f"{label} saved to {out_path}" if out_path else f"{label} ready")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e)[:180]}"
                self.state.set_report({
                    "format_id": format_id, "label": label,
                    "status": "error", "text": "", "file": "", "error": msg,
                })
                self.log(f"Could not generate {label}: {msg}", "error")

        if blocking:
            work()
        else:
            threading.Thread(target=work, daemon=True, name="eloquent-report").start()
        return True
