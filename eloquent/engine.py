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

from .audio import LoopbackCapture, MicCapture
from .cleaner import Cleaner
from .diarizer import Diarizer, build_speaker_extractor, ensure_speaker_model
from .llm import make_provider
from .llm.prompts import build_format_instruction
from .session import Session, write_session_files
from .speech import (SAMPLE_RATE, WINDOW, build_recognizer, build_vad, ensure_asr_model,
                     ensure_vad_model, read_wav_16k, transcribe, transcribe_detailed)
from .state import State, fmt_time

_SILENCE_FRAME = np.zeros(WINDOW, dtype=np.float32)


class _StreamPipe:
    """Per-audio-source VAD + partial buffer + speaker routing state."""

    def __init__(self, tag: str, vad, preroll: int, partial_every: int,
                 use_diarizer: bool = True, fixed_speaker: str = "", partial_prefix: str = ""):
        self.tag = tag
        self.vad = vad
        self.preroll = preroll
        self.partial_every = partial_every
        self.use_diarizer = use_diarizer
        self.fixed_speaker = fixed_speaker
        self.partial_prefix = partial_prefix
        self.buf = collections.deque()
        self.buf_len = 0
        self.since_partial = 0
        self.last_partial = ""
        self.partial_cut = 0
        self.partial_prefix_text = ""
        self.last_partial_dt = 0.0
        self.samples = 0
        self.level = 0.0
        self.last_frame_t = time.monotonic()


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
        if s.diarize and self.extractor is None:
            self._load_speaker_extractor()
        self.diarizer = Diarizer(self.extractor, self.state, self.session, log=self.log) if (
            s.diarize and self.extractor is not None
        ) else None
        if self.diarizer is not None:
            self.diarizer.start()
        self.cleaner = Cleaner(self.state, self.session, self.provider, s, log=self.log)
        self.cleaner.start()

        # per-session pipeline state (one VAD pipe per live source, or single file pipe)
        preroll = int(0.3 * SAMPLE_RATE)
        partial_every = int(s.partial_interval * SAMPLE_RATE)
        src = s.audio_source if s.audio_source in ("mic", "system", "both") else "mic"
        max_utt = min(float(s.max_utterance), 8.0)
        self._is_file_run = wav_path is not None
        self._session_t0 = time.monotonic()
        self._pipes: dict[str, _StreamPipe] = {}
        if wav_path is not None:
            self._pipes["file"] = _StreamPipe(
                "file", build_vad(self.vad_path, s.vad_threshold, s.silence, max_utt),
                preroll, partial_every, use_diarizer=(self.diarizer is not None),
            )
        elif src == "both":
            self._pipes["mic"] = _StreamPipe(
                "mic", build_vad(self.vad_path, s.vad_threshold, s.silence, max_utt),
                preroll, partial_every, use_diarizer=False, fixed_speaker="You", partial_prefix="You: ",
            )
            self._pipes["system"] = _StreamPipe(
                "system", build_vad(self.vad_path, s.vad_threshold, s.silence, max_utt),
                preroll, partial_every, use_diarizer=(self.diarizer is not None),
                fixed_speaker="" if self.diarizer is not None else "Remote",
            )
        else:
            self._pipes[src] = _StreamPipe(
                src, build_vad(self.vad_path, s.vad_threshold, s.silence, max_utt),
                preroll, partial_every, use_diarizer=(self.diarizer is not None),
            )

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
                self._loop_live()
            else:
                self._loop_file(Path(wav_path))
        except Exception as e:
            self.log(f"Audio error: {type(e).__name__}: {e}", "error")
        finally:
            self._finish()

    def _loop_live(self):
        s = self.settings
        q: "queue.Queue[tuple[str, np.ndarray]]" = queue.Queue()
        captures = []
        dual = "mic" in self._pipes and "system" in self._pipes
        wav_sys = collections.deque() if (dual and s.save_audio) else None
        frame_dt = WINDOW / SAMPLE_RATE

        try:
            if "mic" in self._pipes:
                mic = MicCapture(s.device, q, tag="mic", on_status=lambda m: self.log(m, "warn"))
                mic.start()
                captures.append(mic)
            if "system" in self._pipes:
                sys_cap = LoopbackCapture(s.loopback_device, q, tag="system", on_log=self.log)
                sys_cap.start()
                captures.append(sys_cap)

            self._session_t0 = time.monotonic()
            for p in self._pipes.values():
                p.last_frame_t = self._session_t0

            while not self._stop.is_set():
                try:
                    tag, frame = q.get(timeout=0.04)
                except queue.Empty:
                    tag, frame = None, None

                now = time.monotonic()
                if tag is not None:
                    pipe = self._pipes[tag]
                    pipe.last_frame_t = now
                    pipe.level = float(np.sqrt(np.mean(frame * frame)))
                    if s.save_audio:
                        if dual:
                            if tag == "mic":
                                sf = wav_sys.popleft() if wav_sys else 0.0
                                self.session.write_audio(np.clip(frame + sf, -1.0, 1.0))
                            elif len(wav_sys) < 12:
                                wav_sys.append(frame)
                        else:
                            self.session.write_audio(frame)
                    self._process_pipe(pipe, frame, allow_partial=q.qsize() < 6, elapsed=now - self._session_t0)

                # Windows WASAPI loopback pauses callbacks during total system silence; inject trailing
                # quiet frames if speech was active so Silero VAD finalizes immediately on YouTube pause.
                sys_pipe = self._pipes.get("system")
                if sys_pipe is not None and (now - sys_pipe.last_frame_t) >= 0.08:
                    sys_pipe.level = 0.0
                    if sys_pipe.vad.is_speech_detected() or not sys_pipe.vad.empty():
                        sys_pipe.last_frame_t += frame_dt
                        self._process_pipe(sys_pipe, _SILENCE_FRAME, allow_partial=False,
                                           elapsed=now - self._session_t0)

                self.level = max((p.level for p in self._pipes.values()), default=0.0)
        finally:
            for cap in captures:
                cap.close()

    def _loop_file(self, path: Path):
        """Feed a recording through the exact same pipeline, as fast as the CPU allows."""
        self.log(f"Transcribing file: {path}")
        pipe = self._pipes["file"]
        audio = read_wav_16k(path)
        for i in range(0, len(audio) - WINDOW + 1, WINDOW):
            if self._stop.is_set():
                break
            frame = audio[i:i + WINDOW]
            self.session.write_audio(frame)
            pipe.level = float(np.sqrt(np.mean(frame * frame)))
            self.level = pipe.level
            self._process_pipe(pipe, frame, allow_partial=False, elapsed=None)

    def _process_pipe(self, pipe: _StreamPipe, frame: np.ndarray, allow_partial: bool, elapsed):
        pipe.samples += len(frame)
        vad = pipe.vad
        vad.accept_waveform(frame)
        pipe.buf.append(frame)
        pipe.buf_len += len(frame)
        speaking = vad.is_speech_detected()

        if not speaking and vad.empty():
            # keep a short pre-roll so the first syllable is not cut
            while pipe.buf_len - len(pipe.buf[0]) >= pipe.preroll and len(pipe.buf) > 1:
                pipe.buf_len -= len(pipe.buf.popleft())
            return

        pipe.since_partial += len(frame)
        buf_sec = pipe.buf_len / SAMPLE_RATE

        # Prefix-locked partial decoding: keep decoded tail <= ~5.2 s + bounded adaptive cooldown
        base_iv = pipe.partial_every if buf_sec <= 4.5 else int(pipe.partial_every * 1.6)
        min_iv = max(base_iv, min(int(1.4 * SAMPLE_RATE), int(pipe.last_partial_dt * 1.5 * SAMPLE_RATE)))
        if speaking and allow_partial and pipe.since_partial >= min_iv:
            pipe.since_partial = 0
            full = np.concatenate(pipe.buf)
            tail = full[pipe.partial_cut:]
            t_dec = time.monotonic()
            if len(tail) > int(4.4 * SAMPLE_RATE):
                _, pwords = transcribe_detailed(self.recognizer, tail)
                tail_dur = len(tail) / SAMPLE_RATE
                cut_idx = None
                for wi in range(len(pwords) - 1, -1, -1):
                    if pwords[wi]["t1"] <= tail_dur - 1.8:
                        if wi + 1 < len(pwords) and (pwords[wi + 1]["t0"] - pwords[wi]["t1"]) >= 0.06:
                            cut_idx = wi
                            break
                        if cut_idx is None:
                            cut_idx = wi
                if cut_idx is not None:
                    committed = " ".join(w["w"] for w in pwords[:cut_idx + 1]).strip()
                    rest = " ".join(w["w"] for w in pwords[cut_idx + 1:]).strip()
                    if committed:
                        pipe.partial_prefix_text = f"{pipe.partial_prefix_text} {committed}".strip()
                    t_adv = float(pwords[cut_idx]["t1"])
                    if cut_idx + 1 < len(pwords):
                        gap = float(pwords[cut_idx + 1]["t0"]) - t_adv
                        t_adv = t_adv + min(0.03, 0.5 * max(0.0, gap))
                    pipe.partial_cut += min(len(tail), max(0, int(t_adv * SAMPLE_RATE)))
                    text = f"{pipe.partial_prefix_text} {rest}".strip()
                else:
                    text = f"{pipe.partial_prefix_text} {' '.join(w['w'] for w in pwords)}".strip()
            else:
                tail_text = transcribe(self.recognizer, tail)
                text = f"{pipe.partial_prefix_text} {tail_text}".strip()
            pipe.last_partial_dt = time.monotonic() - t_dec
            if text and text != pipe.last_partial:
                pipe.last_partial = text
                self.state.set_partial(f"{pipe.partial_prefix}{text}" if pipe.partial_prefix else text)

        self._drain_pipe(pipe, elapsed)

    def _drain_pipe(self, pipe: _StreamPipe, elapsed):
        """Transcribe every finished utterance on a pipe and route to State (+ Diarizer if applicable)."""
        vad = pipe.vad
        while not vad.empty():
            seg = np.asarray(vad.front.samples, dtype=np.float32)
            vad.pop()
            t1 = time.time()
            text, words = transcribe_detailed(self.recognizer, seg)
            dur = len(seg) / SAMPLE_RATE
            decode = time.time() - t1
            if text:
                end_sec = (pipe.samples / SAMPLE_RATE) if elapsed is None else elapsed
                t0 = max(0.0, end_sec - dur)
                stamp = fmt_time(t0)
                self.session.add_raw(stamp, text)
                use_diar = pipe.use_diarizer and self.diarizer is not None
                seg_id = self.state.add_segment(
                    text, stamp, {"audio_sec": dur, "decode_sec": decode},
                    start_sec=t0, spk_ready=(not use_diar), speaker=pipe.fixed_speaker,
                )
                if use_diar:
                    self.diarizer.submit(seg_id, t0, stamp, seg, text, words)
            else:
                self.state.set_partial("")
            pipe.buf.clear()
            pipe.buf_len = 0
            pipe.since_partial = 0
            pipe.last_partial = ""
            pipe.partial_cut = 0
            pipe.partial_prefix_text = ""
            pipe.last_partial_dt = 0.0

    def _finish(self):
        """Flush all active streams, finish diarization & cleanup, then close session files."""
        self._set_status("stopping")
        self.level = 0.0
        try:
            elapsed = None if self._is_file_run else (time.monotonic() - self._session_t0)
            for pipe in self._pipes.values():
                pipe.vad.flush()
                self._drain_pipe(pipe, elapsed)
        except Exception as e:
            self.log(f"Flush error: {e}", "error")
        self.state.set_partial("")
        if self.diarizer is not None:
            self.diarizer.flush_and_wait()
        self.cleaner.flush_and_wait()
        if self.diarizer is not None or self.state.snapshot().get("multi_speaker"):
            self.session.sync_from_state(self.state)
        self.diarizer = None
        self.session.close()
        d = str(self.session.dir)
        self._set_status("idle")
        self._emit("session_ended", {"dir": d})

    # ------------------------------------------------------------------ #
    # speaker naming & post-session document generation
    # ------------------------------------------------------------------ #
    def rename_speakers(self, names: dict[str, str]):
        """Rename speakers in State and sync raw.txt / clean.txt on disk."""
        self.state.rename_speakers(names)
        if self.session is not None and not self.session.raw_f.closed:
            self.session.sync_from_state(self.state)
        elif self.last_session_dir:
            write_session_files(Path(self.last_session_dir), self.state)

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
