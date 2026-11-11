"""Speech models: download, load, decode.

Credits
  ASR  NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0), INT8 export run through sherpa-onnx
  VAD  Silero VAD, run through sherpa-onnx
"""

import urllib.request
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
WINDOW = 512  # Silero VAD frame (32 ms at 16 kHz)

MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
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


def transcribe(recognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


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
