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
import gc
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .audio import LoopbackCapture, MicCapture
from .cleaner import Cleaner, DICTATION_PAUSE_SEC
from .config import config_dir, is_arabic_speech
from .diarizer import Diarizer, build_speaker_extractor, ensure_speaker_model
from .llm import make_provider
from .llm.prompts import build_format_instruction
from .overlap import build_overlap_segmenter
from .session import Session, load_session_data, write_session_files
from .speech import (SAMPLE_RATE, WINDOW, build_speech_recognizer, build_vad, close_recognizer,
                     ensure_vad_model, read_wav_16k, transcribe, transcribe_detailed)
from .state import State, fmt_time

_SILENCE_FRAME = np.zeros(WINDOW, dtype=np.float32)


class _StreamPipe:
    """Per-audio-source VAD + partial buffer + speaker routing state."""

    def __init__(self, tag: str, vad, preroll: int, partial_every: int,
                 max_samples: int = int(8.0 * SAMPLE_RATE),
                 use_diarizer: bool = True, fixed_speaker: str = "", partial_prefix: str = ""):
        self.tag = tag
        self.vad = vad
        self.preroll = preroll
        self.partial_every = partial_every
        self.max_samples = max(int(SAMPLE_RATE), int(max_samples))
        self.speech_samples = 0
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
        self.last_end_sec = -1.0
        self.level = 0.0
        self.last_frame_t = time.monotonic()


class Engine:
    def __init__(self, settings):
        self.settings = settings
        self.state = State()
        self.state.emit = self._emit
        self.status = "idle"
        self.session_mode = "meeting"
        self._paste_on_stop = False
        self.llm = {"state": "off", "label": "", "message": ""}
        self.level = 0.0                 # mic RMS of the latest frame (for UI meters)
        self.session = None
        self.last_session_dir = None
        self.recognizer = None
        self.vad_path = None
        self.extractor = None
        self.segmenter = None
        self._segmenter_tried = False
        self.diarizer = None
        self._last_clusterer = None
        self.provider = None
        self.dict_provider = None
        self.compose_provider = None
        self.compose_history = []
        self._diar_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._provider_key = None
        self._listeners = []
        self._stop = threading.Event()
        self._worker = None
        self._loaded = False
        self._loaded_speech_lang = None

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
        try:
            line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level.upper()}] {msg}\n"
            with self._log_lock:
                with open(config_dir() / "eloquent.log", "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass
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
    def _load_active_model(self, force_reload: bool = False):
        """Load the speech recognizer matching settings.speech_language, swapping cleanly if changed."""
        with self._model_lock:
            if self.busy:
                return
            s = self.settings
            lang = s.resolved_speech_language()
            if not force_reload and self._loaded and self._loaded_speech_lang == lang:
                return
            self._set_status("loading")
            try:
                if self.recognizer is not None:
                    close_recognizer(self.recognizer)
                    self.recognizer = None
                    self._loaded = False
                    self._loaded_speech_lang = None
                    gc.collect()
                models = s.resolved_models_dir()
                self.vad_path = ensure_vad_model(models, self.log)
                threads = s.resolved_threads()
                t0 = time.time()
                self.recognizer = build_speech_recognizer(lang, models, threads, self.log)
                self._loaded = True
                self._loaded_speech_lang = lang
                self.log(f"Speech model ready in {time.time() - t0:.1f}s")
                if s.diarize:
                    self._load_speaker_extractor(models)
                self.refresh_provider()
            except Exception:
                self._loaded = False
                self._loaded_speech_lang = None
                raise
            finally:
                if not self.busy:
                    self._set_status("idle")

    def load(self):
        """Download (first run) and load the speech models. Blocking; call from a worker thread in GUIs."""
        self._load_active_model(force_reload=False)

    def reload_speech_model(self):
        """Hot-swap the speech recognizer when settings.speech_language changes while idle."""
        lang = self.settings.resolved_speech_language()
        if self._loaded and self._loaded_speech_lang == lang:
            return
        try:
            self._load_active_model(force_reload=True)
        except Exception as e:
            self.log(f"Could not switch speech model ({type(e).__name__}: {e})", "error")

    def _load_speaker_extractor(self, models_dir: Path = None):
        with self._diar_lock:
            mdir = models_dir or self.settings.resolved_models_dir()
            if self.extractor is None:
                try:
                    spk_path = ensure_speaker_model(mdir, self.log)
                    self.extractor = build_speaker_extractor(spk_path, threads=1)
                    self.log("[diarize] Multi-speaker identification ready (1 background thread)")
                except Exception as e:
                    self.log(f"[diarize] Disabled ({type(e).__name__}: {e})", "warn")
                    return
            if not self._segmenter_tried:
                self._segmenter_tried = True
                self.segmenter = build_overlap_segmenter(mdir, threads=1, log=self.log)
                if self.segmenter is not None:
                    self.log("[diarize] Overlap segmentation ready (pyannote-3.0)")

    def sync_diarizer(self):
        """Lazy-load speaker extractor if enabled and propagate live max_speakers changes."""
        if self.settings.diarize and self._loaded and self.extractor is None:
            threading.Thread(target=self._load_speaker_extractor, daemon=True, name="eloquent-diar-load").start()
        self.set_max_speakers(getattr(self.settings, "max_speakers", 0))

    def clear_compose_history(self):
        self.compose_history.clear()

    def refresh_provider(self, force: bool = False):
        """(Re)create the LLM provider(s) when settings change. The key check runs in the background."""
        s = self.settings
        vocab = getattr(s, "custom_vocabulary", "")
        for p in (self.provider, self.dict_provider, self.compose_provider):
            if p is not None and hasattr(p, "custom_vocabulary"):
                p.custom_vocabulary = vocab
        key = (s.cleanup, s.dictation_cleanup, bool(s.compose_hotkey),
               s.resolved_api_key(), s.llm_model, s.keep_fillers, s.resolved_speech_language())
        if key == self._provider_key and not force:
            return
        self._provider_key = key
        self.provider, reason = make_provider(s)
        forced = None
        if self.provider is None and (s.dictation_cleanup or s.compose_hotkey):
            forced, _ = make_provider(s, force=True)
        self.dict_provider = (self.provider or forced) if s.dictation_cleanup else None
        self.compose_provider = (self.provider or forced) if s.compose_hotkey else None
        for p in (self.provider, self.dict_provider, self.compose_provider):
            if p is not None and hasattr(p, "log") and p.log is None:
                p.log = self.log

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

    def start(self, wav_path: Path = None, mode: str = "meeting", selected_text: str = "") -> bool:
        if not self._loaded or self.status != "idle":
            return False
        s = self.settings
        mode = mode if (mode in ("dictation", "compose") and wav_path is None) else "meeting"
        dictating = (mode == "dictation")
        composing = (mode == "compose")
        paste_mode = dictating or composing
        self.refresh_provider()
        if composing:
            cleaner_provider = self.compose_provider
            if cleaner_provider is None:
                cleaner_provider, reason = make_provider(s, force=True)
                self.compose_provider = cleaner_provider
            else:
                reason = ""
            if cleaner_provider is None:
                msg = reason or "Add a Gemini API key in Settings to use Ask AI."
                self.log(f"[compose] {msg}", "warn")
                self._emit("compose_unavailable", msg)
                return False
        elif dictating:
            cleaner_provider = self.dict_provider
        else:
            cleaner_provider = self.provider

        self.session_mode = mode
        self._paste_on_stop = paste_mode

        self.state.reset()
        self._last_clusterer = None
        if paste_mode:
            self.session = None
        else:
            self.session = Session(s.resolved_sessions_dir(), save_audio=s.save_audio)
            self.last_session_dir = self.session.dir
        use_diar = (not paste_mode) and s.diarize
        arabic = is_arabic_speech(s.resolved_speech_language())
        if use_diar and (self.extractor is None or not self._segmenter_tried):
            self._load_speaker_extractor()
        self.diarizer = Diarizer(
            self.extractor, self.state, self.session, log=self.log,
            arabic_mode=arabic,
            max_speakers=getattr(s, "max_speakers", 0),
            segmenter=self.segmenter,
        ) if (use_diar and self.extractor is not None) else None
        if self.diarizer is not None:
            self.diarizer.start()
        self.cleaner = Cleaner(
            self.state, self.session, cleaner_provider, s, mode=mode,
            selected_text=selected_text if composing else "",
            compose_history=list(self.compose_history) if composing else None,
            log=self.log,
        )
        self.cleaner.start()

        # per-session pipeline state (one VAD pipe per live source, or single file pipe)
        silence, max_utt_raw, partial_sec = s.resolved_timing()
        preroll = int(0.3 * SAMPLE_RATE)
        partial_every = int(partial_sec * SAMPLE_RATE)
        src = "mic" if paste_mode else (s.audio_source if s.audio_source in ("mic", "system", "both") else "mic")
        max_utt = min(float(max_utt_raw), 8.0)
        self._is_file_run = wav_path is not None
        self._session_t0 = time.monotonic()
        max_samples = int(max_utt * SAMPLE_RATE)
        self._pipes: dict[str, _StreamPipe] = {}
        if wav_path is not None:
            self._pipes["file"] = _StreamPipe(
                "file", build_vad(self.vad_path, s.vad_threshold, silence, max_utt),
                preroll, partial_every, max_samples=max_samples, use_diarizer=(self.diarizer is not None),
            )
        elif src == "both":
            self._pipes["mic"] = _StreamPipe(
                "mic", build_vad(self.vad_path, s.vad_threshold, silence, max_utt),
                preroll, partial_every, max_samples=max_samples, use_diarizer=False, fixed_speaker="You", partial_prefix="You: ",
            )
            self._pipes["system"] = _StreamPipe(
                "system", build_vad(self.vad_path, s.vad_threshold, silence, max_utt),
                preroll, partial_every, max_samples=max_samples, use_diarizer=(self.diarizer is not None),
                fixed_speaker="" if self.diarizer is not None else "Remote",
            )
        else:
            self._pipes[src] = _StreamPipe(
                src, build_vad(self.vad_path, s.vad_threshold, silence, max_utt),
                preroll, partial_every, max_samples=max_samples, use_diarizer=(self.diarizer is not None),
            )

        self._stop.clear()
        self._set_status("listening")
        self._emit("session_started", {
            "dir": str(self.session.dir) if self.session is not None else "",
            "mode": mode,
        })
        self._worker = threading.Thread(target=self._run, args=(wav_path,), daemon=True,
                                        name="eloquent-audio")
        self._worker.start()
        return True

    def stop(self, paste: bool = True):
        """Ask the audio thread to stop. Returns immediately; status goes stopping -> idle."""
        if self.status == "listening":
            if not paste:
                self._paste_on_stop = False
            self._stop.set()

    def toggle(self, mode: str = "meeting", selected_text: str = "") -> bool:
        if self.status == "idle":
            return self.start(mode=mode, selected_text=selected_text)
        if self.status == "listening":
            if mode == self.session_mode:
                self.stop(paste=True)
                return True
            if self.session_mode in ("dictation", "compose") and mode == "meeting":
                self.stop(paste=False)
                return True
        return False

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
                    if s.save_audio and self.session is not None:
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

        # sherpa-onnx Silero VAD suppresses max_speech_duration after its first forced cut until silence;
        # flush cleanly at max_samples so continuous fast speech still emits bounded segments and partials.
        if speaking:
            pipe.speech_samples += len(frame)
            if pipe.speech_samples >= pipe.max_samples and vad.empty():
                vad.flush()
                pipe.speech_samples = 0
                speaking = vad.is_speech_detected()
        else:
            pipe.speech_samples = 0

        if not speaking and vad.empty():
            # keep a short pre-roll so the first syllable is not cut
            while pipe.buf_len - len(pipe.buf[0]) >= pipe.preroll and len(pipe.buf) > 1:
                dropped = len(pipe.buf.popleft())
                pipe.buf_len -= dropped
                pipe.partial_cut = max(0, pipe.partial_cut - dropped)
            return

        max_buf = pipe.max_samples + pipe.preroll
        while pipe.buf_len - len(pipe.buf[0]) >= max_buf and len(pipe.buf) > 1:
            dropped = len(pipe.buf.popleft())
            pipe.buf_len -= dropped
            pipe.partial_cut = max(0, pipe.partial_cut - dropped)

        pipe.since_partial += len(frame)
        buf_sec = pipe.buf_len / SAMPLE_RATE

        # Prefix-locked partial decoding: keep decoded tail <= ~5.2 s + bounded adaptive cooldown
        base_iv = pipe.partial_every if buf_sec <= 4.5 else int(pipe.partial_every * 1.6)
        min_iv = max(base_iv, min(int(1.4 * SAMPLE_RATE), int(pipe.last_partial_dt * 1.5 * SAMPLE_RATE)))
        supports_words = bool(getattr(self.recognizer, "supports_word_timestamps", True))
        if speaking and vad.empty() and allow_partial and pipe.since_partial >= min_iv:
            pipe.since_partial = 0
            full = np.concatenate(pipe.buf)
            tail = full[pipe.partial_cut:]
            t_dec = time.monotonic()
            try:
                if supports_words and len(tail) > int(4.4 * SAMPLE_RATE):
                    tail_raw, pwords = transcribe_detailed(self.recognizer, tail)
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
                    elif pwords:
                        text = f"{pipe.partial_prefix_text} {' '.join(w['w'] for w in pwords)}".strip()
                    else:
                        text = f"{pipe.partial_prefix_text} {tail_raw}".strip()
                else:
                    if len(tail) > pipe.max_samples:
                        tail = tail[-pipe.max_samples:]
                    tail_text = transcribe(self.recognizer, tail)
                    text = f"{pipe.partial_prefix_text} {tail_text}".strip()
                pipe.last_partial_dt = time.monotonic() - t_dec
                if text and text != pipe.last_partial:
                    pipe.last_partial = text
                    self.state.set_partial(f"{pipe.partial_prefix}{text}" if pipe.partial_prefix else text)
            except Exception as e:
                self.log(f"[asr] partial skipped: {type(e).__name__}: {e}", "warn")

        self._drain_pipe(pipe, elapsed)

    def _drain_pipe(self, pipe: _StreamPipe, elapsed):
        """Transcribe every finished utterance on a pipe and route to State (+ Diarizer if applicable)."""
        vad = pipe.vad
        while not vad.empty():
            seg = np.asarray(vad.front.samples, dtype=np.float32)
            vad.pop()
            t1 = time.time()
            try:
                text, words = transcribe_detailed(self.recognizer, seg)
            except Exception as e:
                self.log(f"[asr] segment skipped: {type(e).__name__}: {e}", "warn")
                text, words = "", []
            dur = len(seg) / SAMPLE_RATE
            decode = time.time() - t1
            if text:
                end_sec = (pipe.samples / SAMPLE_RATE) if elapsed is None else elapsed
                t0 = max(0.0, end_sec - dur)
                gap_before = max(0.0, t0 - pipe.last_end_sec) if pipe.last_end_sec >= 0.0 else 0.0
                pipe.last_end_sec = end_sec
                stamp = fmt_time(t0)
                if self.session is not None:
                    self.session.add_raw(stamp, text)
                use_diar = pipe.use_diarizer and self.diarizer is not None
                seg_id = self.state.add_segment(
                    text, stamp, {"audio_sec": dur, "decode_sec": decode},
                    start_sec=t0, spk_ready=(not use_diar), speaker=pipe.fixed_speaker,
                    gap_before=gap_before,
                )
                if use_diar:
                    self.diarizer.submit(seg_id, t0, stamp, seg, text, words)
            else:
                self.state.set_partial("")
            pipe.buf.clear()
            pipe.buf_len = 0
            pipe.speech_samples = 0
            pipe.since_partial = 0
            pipe.last_partial = ""
            pipe.partial_cut = 0
            pipe.partial_prefix_text = ""
            pipe.last_partial_dt = 0.0

    def _dictation_text(self) -> str:
        with self.state.lock:
            parts = []
            for p in self.state.paragraphs:
                txt = (p["clean"] or p["raw"]).strip()
                if not txt:
                    continue
                if parts:
                    parts.append("\n\n" if p.get("break_before") else " ")
                parts.append(txt)
            for x in self.state.loose:
                txt = x["text"].strip()
                if not txt:
                    continue
                if parts:
                    parts.append("\n\n" if x.get("gap_before", 0.0) >= DICTATION_PAUSE_SEC else " ")
                parts.append(txt)
            return "".join(parts).strip()

    def _compose_result(self) -> tuple[str, str]:
        with self.state.lock:
            raw_parts = []
            clean_parts = []
            err = ""
            for p in self.state.paragraphs:
                raw = (p.get("raw") or "").strip()
                clean = (p.get("clean") or "").strip()
                note = (p.get("note") or "").strip()
                if raw:
                    raw_parts.append(raw)
                if clean:
                    clean_parts.append(clean)
                elif raw and not err:
                    err = note
            state_err = (self.state.error or "").strip()

        if not raw_parts and not clean_parts:
            return "", ""
        if len(clean_parts) < len(raw_parts) or not clean_parts:
            return "", err or state_err or "Ask AI unavailable"
        generated = "\n\n".join(clean_parts).strip()
        raw_prompt = "\n\n".join(raw_parts).strip()
        if raw_prompt and generated:
            self.compose_history.append((raw_prompt, generated))
            self.compose_history = self.compose_history[-5:]
        return generated, ""

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
        if self.session is not None:
            if self.diarizer is not None or self.state.snapshot().get("multi_speaker"):
                self.session.sync_from_state(self.state)
            self.session.close()
        if self.diarizer is not None:
            self._last_clusterer = self.diarizer.clusterer
        self.diarizer = None
        d = str(self.session.dir) if self.session is not None else ""
        mode = self.session_mode
        cancelled = (mode in ("dictation", "compose") and not self._paste_on_stop)
        paste_text = ""
        err = ""
        if not cancelled:
            if mode == "dictation":
                paste_text = self._dictation_text()
            elif mode == "compose":
                paste_text, err = self._compose_result()
        self._set_status("idle")
        self._emit("session_ended", {
            "dir": d, "mode": mode, "paste_text": paste_text,
            "cancelled": cancelled, "error": err,
        })

    # ------------------------------------------------------------------ #
    # speaker naming & post-session document generation
    # ------------------------------------------------------------------ #
    def _sync_session_files(self):
        if self.session is not None and not self.session.raw_f.closed:
            self.session.sync_from_state(self.state)
        elif self.last_session_dir:
            write_session_files(Path(self.last_session_dir), self.state)

    def rename_speakers(self, names: dict[str, str]):
        """Rename speakers in State (locking named clusters against merge) and sync raw.txt / clean.txt."""
        cur_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None:
            diar.set_speaker_names(names, current_names=cur_names)
        elif self._last_clusterer is not None:
            self._last_clusterer.set_speaker_names(names, current_names=cur_names)
        self.state.rename_speakers(names)
        self._sync_session_files()

    def can_dissolve_speaker(self, spk_cid: str) -> bool:
        """Return True when *spk_cid* has voiceprints and at least 2 diarized speakers exist."""
        if not spk_cid:
            return False
        diar = self.diarizer
        if diar is not None:
            return diar.has_speaker(spk_cid) and diar.speaker_count() >= 2
        cl = self._last_clusterer
        if cl is not None:
            return cl.has_speaker(spk_cid) and cl.speaker_count() >= 2
        return False

    def speaker_anchor_ids(self, spk_cid: str) -> list[int]:
        """Capture stable turn IDs for *spk_cid* so live UI clicks survive renumbering."""
        if not spk_cid:
            return []
        diar = self.diarizer
        if diar is not None:
            return diar.speaker_anchor_ids(spk_cid)
        cl = self._last_clusterer
        if cl is not None:
            return cl.speaker_anchor_ids(spk_cid)
        return []

    def resolve_speaker_from_anchors(self, fallback_cid: str, anchors: list[int]) -> str:
        diar = self.diarizer
        if diar is not None:
            return diar.resolve_speaker_from_anchors(fallback_cid, anchors)
        cl = self._last_clusterer
        if cl is not None:
            return cl.resolve_speaker_from_anchors(fallback_cid, anchors)
        return fallback_cid

    def merge_speakers(self, src_cid: str, dst_cid: str,
                       src_anchors: list[int] = None, dst_anchors: list[int] = None) -> bool:
        """Merge *src_cid* into *dst_cid* (live voiceprint merge when available, else State merge)."""
        if src_anchors:
            src_cid = self.resolve_speaker_from_anchors(src_cid, src_anchors)
        if dst_anchors:
            dst_cid = self.resolve_speaker_from_anchors(dst_cid, dst_anchors)
        if not src_cid or not dst_cid or src_cid == dst_cid:
            return False
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None and diar.has_speaker(src_cid) and diar.has_speaker(dst_cid):
            mapping, remapped = diar.merge_speakers(src_cid, dst_cid, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=diar.last_id_remap())
                self._sync_session_files()
                return True
        cl = self._last_clusterer
        if diar is None and cl is not None and cl.has_speaker(src_cid) and cl.has_speaker(dst_cid):
            mapping, remapped = cl.merge_speakers(src_cid, dst_cid, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=cl.last_id_remap)
                self._sync_session_files()
                return True
        if self.state.merge_speakers_fallback(src_cid, dst_cid):
            self._sync_session_files()
            return True
        return False

    def reassign_line_speaker(self, item_kind: str, item_id: int, dst_cid: str,
                              dst_anchors: list[int] = None) -> bool:
        """Reassign only one paragraph (`'p'`) or loose segment (`'l'`) to *dst_cid*."""
        if dst_anchors:
            dst_cid = self.resolve_speaker_from_anchors(dst_cid, dst_anchors)
        if not item_kind or item_id is None or not dst_cid:
            return False
        turn_ids = self.state.item_turn_ids(item_kind, item_id)
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None and turn_ids and diar.has_speaker(dst_cid) and diar.has_any_turn(turn_ids):
            mapping, remapped = diar.pin_turns_to_speaker(turn_ids, dst_cid, old_names=old_names)
            if mapping:
                id_remap = diar.last_id_remap()
                self.state.apply_cluster_remap(mapping, remapped, id_remap=id_remap)
                self.state.reassign_item_speaker(item_kind, item_id, id_remap.get(dst_cid, dst_cid))
                self._sync_session_files()
                return True
        cl = self._last_clusterer
        if diar is None and cl is not None and turn_ids and cl.has_speaker(dst_cid) and cl.has_any_turn(turn_ids):
            mapping, remapped = cl.pin_turns_to_speaker(turn_ids, dst_cid, old_names=old_names)
            if mapping:
                id_remap = cl.last_id_remap
                self.state.apply_cluster_remap(mapping, remapped, id_remap=id_remap)
                self.state.reassign_item_speaker(item_kind, item_id, id_remap.get(dst_cid, dst_cid))
                self._sync_session_files()
                return True
        if self.state.reassign_item_speaker(item_kind, item_id, dst_cid):
            self._sync_session_files()
            return True
        return False

    def dissolve_speaker(self, src_cid: str, src_anchors: list[int] = None) -> bool:
        """Reassign a clash/overlap speaker's turns individually to the closest remaining speakers."""
        if src_anchors:
            src_cid = self.resolve_speaker_from_anchors(src_cid, src_anchors)
        if not src_cid:
            return False
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None and diar.has_speaker(src_cid) and diar.speaker_count() >= 2:
            mapping, remapped = diar.dissolve_speaker(src_cid, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=diar.last_id_remap())
                self._sync_session_files()
                return True
        cl = self._last_clusterer
        if diar is None and cl is not None and cl.has_speaker(src_cid) and cl.speaker_count() >= 2:
            mapping, remapped = cl.dissolve_speaker(src_cid, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=cl.last_id_remap)
                self._sync_session_files()
                return True
        return False

    def speaker_edit_count(self) -> int:
        """Number of reversible manual speaker merges / overlap reassignments in the active or last session."""
        diar = self.diarizer
        if diar is not None:
            return diar.manual_edit_count()
        cl = self._last_clusterer
        if cl is not None:
            return cl.manual_edit_count()
        return 0

    def undo_last_speaker_edit(self) -> bool:
        """Undo the most recent manual speaker merge or overlap reassignment."""
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None and diar.manual_edit_count() > 0:
            mapping, remapped = diar.undo_last_speaker_edit(old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=diar.last_id_remap())
                self._sync_session_files()
                return True
            return False
        cl = self._last_clusterer
        if diar is None and cl is not None and cl.manual_edit_count() > 0:
            mapping, remapped = cl.undo_last_speaker_edit(old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=cl.last_id_remap)
                self._sync_session_files()
                return True
        return False

    def reset_speaker_edits(self) -> bool:
        """Reset all manual speaker merges and overlap reassignments back to automatic clustering."""
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None and diar.manual_edit_count() > 0:
            mapping, remapped = diar.reset_speaker_edits(old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=diar.last_id_remap())
                self._sync_session_files()
                return True
            return False
        cl = self._last_clusterer
        if diar is None and cl is not None and cl.manual_edit_count() > 0:
            mapping, remapped = cl.reset_speaker_edits(old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=cl.last_id_remap)
                self._sync_session_files()
                return True
        return False

    def set_max_speakers(self, max_speakers: int) -> bool:
        """Apply a new maximum speaker limit live (or to the just-finished session)."""
        cap = int(max_speakers) if int(max_speakers or 0) >= 2 else 0
        old_names = self.state.speaker_names_copy()
        diar = self.diarizer
        if diar is not None:
            mapping, remapped = diar.set_max_speakers(cap, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=diar.last_id_remap())
                self._sync_session_files()
                return True
            return False
        cl = self._last_clusterer
        if cl is not None:
            mapping, remapped = cl.set_max_speakers(cap, old_names=old_names)
            if mapping:
                self.state.apply_cluster_remap(mapping, remapped, id_remap=cl.last_id_remap)
                self._sync_session_files()
                return True
        return False

    def apply_speaker_actions(self, renames: dict[str, str],
                              merges: list[tuple[str, str]],
                              dissolves: list[str]) -> None:
        """Apply renames, merges, and overlap dissolves safely even when IDs renumber mid-batch."""
        if renames:
            self.rename_speakers(renames)
        if not merges and not dissolves:
            return
        diar = self.diarizer
        cl = diar.clusterer if diar is not None else self._last_clusterer
        anchors: dict[str, list[int]] = {}
        if cl is not None:
            if diar is not None:
                with diar._cluster_lock:
                    anchors = cl._speaker_turn_ids()
            else:
                anchors = cl._speaker_turn_ids()

        def _resolve_live(orig_cid: str, redirected: dict[str, str]) -> str:
            if anchors.get(orig_cid) and cl is not None:
                tids = set(anchors[orig_cid])
                if diar is not None:
                    with diar._cluster_lock:
                        labels = [t.get("speaker", "") for t in cl.turns if t["id"] in tids and t.get("speaker")]
                else:
                    labels = [t.get("speaker", "") for t in cl.turns if t["id"] in tids and t.get("speaker")]
                if labels:
                    return max(set(labels), key=labels.count)
            cur = orig_cid
            seen = set()
            while cur in redirected and cur not in seen:
                seen.add(cur)
                cur = redirected[cur]
            return cur

        redirected: dict[str, str] = {}
        for src_orig, dst_orig in merges:
            cur_src = _resolve_live(src_orig, redirected)
            cur_dst = _resolve_live(dst_orig, redirected)
            if cur_src and cur_dst and cur_src != cur_dst:
                if self.merge_speakers(cur_src, cur_dst):
                    redirected[src_orig] = dst_orig
        for src_orig in dissolves:
            cur_src = _resolve_live(src_orig, redirected)
            if cur_src:
                self.dissolve_speaker(cur_src)

    def apply_vocabulary_fix(self, wrong: str, right: str, remember: bool = True) -> int:
        """Fix a misheard word/phrase in State (+ session files) and save to custom vocabulary."""
        wrong, right = (wrong or "").strip(), (right or "").strip()
        count = self.state.replace_text(wrong, right) if (wrong and right and wrong != right) else 0
        if remember:
            entry = f"{wrong} -> {right}" if (wrong and right and wrong.lower() != right.lower()) else (right or wrong)
            if self.settings.add_vocabulary_entry(entry):
                self.settings.save()
                self.refresh_provider()
        if count > 0:
            if self.session is not None and not self.session.raw_f.closed:
                self.session.sync_from_state(self.state)
            elif self.last_session_dir:
                write_session_files(Path(self.last_session_dir), self.state)
        return count

    def load_session(self, session_dir: Path) -> Optional[dict]:
        """Load a saved session folder into State so it can be read, edited, or formatted."""
        if self.busy:
            return None
        d = Path(session_dir)
        if not d.exists() or not d.is_dir():
            return None
        data = load_session_data(d)
        self._last_clusterer = None
        self.session = None
        self.last_session_dir = d
        self.state.load_saved_session(data["paragraphs"], report=data.get("report"))
        return data

    def open_saved_report(self, report_meta: dict) -> bool:
        """Load an existing saved ``.md`` report from the active/loaded session into State."""
        if not report_meta:
            return False
        p = Path(report_meta["path"])
        if not p.exists():
            return False
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            return False
        if not text:
            return False
        self.state.set_report({
            "format_id": report_meta.get("format_id", "custom"),
            "label": report_meta.get("label", p.stem),
            "status": "done",
            "text": text,
            "file": str(p),
            "error": "",
        })
        return True

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
