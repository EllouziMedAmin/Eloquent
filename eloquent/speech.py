"""Speech models: download, load, decode.

Credits
  ASR  NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0), INT8 export run through sherpa-onnx
       Cohere Arabic Q4_K_M GGUF, run through transcribe.cpp (CPU + Vulkan GPU)
  VAD  Silero VAD, run through sherpa-onnx
"""

import os
import re
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import sherpa_onnx

from .config import is_arabic_speech

SAMPLE_RATE = 16000
WINDOW = 512  # Silero VAD frame (32 ms at 16 kHz)

MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
COHERE_GGUF_REL = Path("cohere-gguf") / "cohere-transcribe-arabic-07-2026-Q4_K_M.gguf"
COHERE_GGUF_URL = (
    "https://huggingface.co/handy-computer/cohere-transcribe-arabic-07-2026-gguf/"
    "resolve/main/cohere-transcribe-arabic-07-2026-Q4_K_M.gguf"
)
TRANSCRIBE_CPP_RELEASE = "https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.2/"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"


def ensure_asr_model(models_dir: Path, log=print) -> Path:
    target = models_dir / "parakeet-tdt-0.6b-v3-int8"
    if all((target / f).exists() for f in MODEL_FILES):
        return target
    log("Downloading Parakeet TDT 0.6B v3 INT8 (~670 MB, first run only)...")
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=MODEL_REPO, allow_patterns=["*.onnx", "tokens.txt"],
                                  local_dir=target))


def ensure_vad_model(models_dir: Path, log=print) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "silero_vad.onnx"
    if not path.exists():
        log("Downloading Silero VAD (~2 MB)...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(VAD_URL, tmp)
        tmp.replace(path)
    return path


def _download_http_resumable(url: str, dest: Path, log=print, chunk_bytes: int = 32 * 1024 * 1024):
    """Download a large binary file with HTTP Range resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as r:
        total = int(r.headers["Content-Length"])
    done = part.stat().st_size if part.exists() else 0
    with open(part, "ab") as f:
        while done < total:
            end = min(total - 1, done + chunk_bytes - 1)
            for attempt in range(8):
                try:
                    rr = urllib.request.Request(url, headers={"Range": f"bytes={done}-{end}", "User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(rr, timeout=25) as resp:
                        buf = resp.read(1024 * 1024)
                        while buf:
                            f.write(buf)
                            done += len(buf)
                            buf = resp.read(1024 * 1024)
                    f.flush()
                    log(f"Downloading Cohere Arabic GGUF ({done * 100 // total}%)...")
                    break
                except Exception:
                    if attempt == 7:
                        raise
                    time.sleep(2)
    part.replace(dest)


def ensure_cohere_model(models_dir: Path, log=print) -> Path:
    dest = models_dir / COHERE_GGUF_REL
    if dest.exists():
        return dest
    log("Downloading Cohere Arabic Q4_K_M GGUF (~1.45 GiB, first Arabic run only)...")
    _download_http_resumable(COHERE_GGUF_URL, dest, log=log)
    return dest


def ensure_transcribe_cpp(models_dir: Path, log=print):
    """Return imported ``transcribe_cpp`` module, fetching the runtime into models/_vendor if absent."""
    try:
        import transcribe_cpp
        return transcribe_cpp
    except Exception:
        pass

    vdir = models_dir / "_vendor" / "transcribe_cpp"
    vdir.mkdir(parents=True, exist_ok=True)
    whl = vdir / "transcribe_cpp-0.4.2-py3-none-any.whl"
    tgz = vdir / "transcribe-native-0.4.2-windows-x86_64-cpu-vulkan.tar.gz"
    nat = vdir / "native"
    dll = next(nat.rglob("transcribe.dll"), None) if nat.exists() else None
    if not (vdir / "transcribe_cpp").exists():
        log("Downloading transcribe.cpp Python wrapper...")
        urllib.request.urlretrieve(TRANSCRIBE_CPP_RELEASE + whl.name, whl)
        with zipfile.ZipFile(whl) as z:
            z.extractall(vdir)
    if dll is None:
        log("Downloading transcribe.cpp Windows CPU+Vulkan runtime (~17 MB)...")
        urllib.request.urlretrieve(TRANSCRIBE_CPP_RELEASE + tgz.name, tgz)
        nat.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(nat)
        dll = next(nat.rglob("transcribe.dll"))
    os.environ["TRANSCRIBE_LIBRARY"] = str(dll)
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll.parent))
    if str(vdir) not in sys.path:
        sys.path.insert(0, str(vdir))
    import transcribe_cpp
    return transcribe_cpp


def clean_arabic_asr_artifacts(text: str) -> str:
    """Strip Cohere silence/noise control tokens, stray truncation '#' markers, and repetition loops."""
    cleaned = re.sub(r"@+\s*فراغ|\S*@@@\S*|###\S*|#+", "", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\b(\S+(?:\s+\S+)?)(?:\s+\1){2,}\b", r"\1", cleaned).strip()
    return "" if cleaned in ("", ".", "،", "؟", "!", "...") else cleaned


class CohereRecognizer:
    """Thread-safe wrapper around ``transcribe_cpp`` for Cohere Arabic Q4_K_M GGUF."""

    supports_word_timestamps = False

    def __init__(self, tcpp, model_path: Path, threads: int, backend: str = "auto", language: str = "ar"):
        tcpp.set_log_callback(lambda *_: None)
        self._lock = threading.Lock()
        self._closed = False
        self._model = tcpp.Model(model_path, backend=backend)
        self._session = self._model.session(n_threads=max(1, int(threads)))
        self.language = language
        dev = getattr(self._model, "device", None)
        name = getattr(dev, "name", backend) or backend
        desc = getattr(dev, "description", "") or ""
        self.device_label = f"{name} ({desc})".strip() if desc else str(name)

    def transcribe(self, samples: np.ndarray) -> str:
        if len(samples) / SAMPLE_RATE <= 0.05 or float(np.max(np.abs(samples))) < 1e-4:
            return ""
        pcm = np.ascontiguousarray(samples, dtype=np.float32)
        with self._lock:
            if self._closed:
                return ""
            raw = self._session.run(pcm, language=self.language).text.strip()
        return clean_arabic_asr_artifacts(raw)

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._session.close()
            finally:
                self._model.close()


def build_speech_recognizer(speech_language: str, models_dir: Path, threads: int, log=print):
    """Download (if needed) and build the recognizer for ``'parakeet'`` (25 European languages) or ``'arabic'`` (Cohere)."""
    if is_arabic_speech(speech_language):
        tcpp = ensure_transcribe_cpp(models_dir, log)
        gguf = ensure_cohere_model(models_dir, log)
        log(f"Loading Cohere Arabic (auto backend, {threads} CPU threads)...")
        rec = CohereRecognizer(tcpp, gguf, threads, backend="auto", language="ar")
        log(f"Cohere Arabic ready on {rec.device_label}")
        return rec
    model_dir = ensure_asr_model(models_dir, log)
    log(f"Loading Parakeet on CPU with {threads} threads...")
    return build_recognizer(model_dir, threads)


def close_recognizer(recognizer):
    """Cleanly release native resources if the active recognizer owns explicit sessions."""
    if recognizer is not None and hasattr(recognizer, "close"):
        recognizer.close()


def build_recognizer(model_dir: Path, threads: int):
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        model_type="nemo_transducer", num_threads=threads,
        decoding_method="greedy_search", provider="cpu")


def build_vad(vad_path: Path, threshold: float, silence: float, max_utterance: float):
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(vad_path)
    cfg.silero_vad.threshold = threshold
    cfg.silero_vad.min_silence_duration = silence
    cfg.silero_vad.min_speech_duration = 0.15
    cfg.silero_vad.max_speech_duration = max_utterance
    cfg.silero_vad.window_size = WINDOW
    cfg.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


def _extract_words(result) -> list[dict]:
    """Group BPE tokens from OfflineRecognitionResult into word spans: [{'w', 't0', 't1'}]."""
    tokens = getattr(result, "tokens", None) or []
    times = getattr(result, "timestamps", None) or []
    durs = getattr(result, "durations", None) or []
    if not tokens or len(tokens) != len(times):
        return []
    words = []
    for i, tok in enumerate(tokens):
        t0 = float(times[i])
        dur = float(durs[i]) if i < len(durs) and durs[i] > 0 else 0.08
        t1 = t0 + dur
        starts_word = (not words) or tok.startswith((" ", "\u2581"))
        piece = tok.lstrip(" \u2581")
        if not piece:
            continue
        if starts_word:
            words.append({"w": piece, "t0": t0, "t1": t1})
        else:
            words[-1]["w"] += piece
            words[-1]["t1"] = max(words[-1]["t1"], t1)
    return words


def transcribe(recognizer, samples: np.ndarray) -> str:
    if hasattr(recognizer, "transcribe"):
        return recognizer.transcribe(samples)
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def transcribe_detailed(recognizer, samples: np.ndarray) -> tuple[str, list[dict]]:
    """Decode audio once and return (text, [{'w', 't0', 't1'}, ...]) with zero extra ASR cost."""
    if hasattr(recognizer, "transcribe"):
        return recognizer.transcribe(samples), []
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    r = stream.result
    return r.text.strip(), _extract_words(r)


def read_wav_16k(path: Path) -> np.ndarray:
    """Read a PCM WAV file as mono float32 at 16 kHz (simple linear resampling if needed)."""
    import wave
    with wave.open(str(path), "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        data = w.readframes(w.getnframes())
    if width == 2:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != SAMPLE_RATE:
        n = int(len(x) * SAMPLE_RATE / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x
