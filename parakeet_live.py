#!/usr/bin/env python3
"""
Live, CPU-only speech-to-text with NVIDIA Parakeet TDT 0.6B v3 (25 languages).

- Model:   sherpa-onnx INT8 export of nvidia/parakeet-tdt-0.6b-v3 (runs on plain CPU)
- VAD:     Silero VAD (decides when you start/stop talking, skips silence)
- Streaming: Parakeet is an *offline* model, so partial words are produced by
             re-decoding the growing utterance every ~0.5 s. The final text for each
             sentence comes from the clean VAD-cut segment.

Install:
    pip install sherpa-onnx sounddevice numpy huggingface_hub

Run:
    python parakeet_live.py
    python parakeet_live.py --threads 8 --partial-interval 0.4
    python parakeet_live.py --list-devices
    python parakeet_live.py --device 3
"""

import argparse
import collections
import os
import queue
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import sherpa_onnx
import sounddevice as sd

SAMPLE_RATE = 16000
WINDOW = 512  # Silero VAD frame size at 16 kHz (32 ms)

MODEL_REPO = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
MODELS_DIR = Path(__file__).resolve().parent / "models"


# --------------------------------------------------------------------------- #
# Model download / loading
# --------------------------------------------------------------------------- #
def ensure_asr_model() -> Path:
    from huggingface_hub import snapshot_download

    print(f"[setup] ASR model: {MODEL_REPO} (first run downloads ~650 MB)")
    path = snapshot_download(
        repo_id=MODEL_REPO,
        allow_patterns=["*.onnx", "tokens.txt"],
        local_dir=MODELS_DIR / "parakeet-tdt-0.6b-v3-int8",
    )
    return Path(path)


def ensure_vad_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "silero_vad.onnx"
    if not path.exists():
        print("[setup] Downloading Silero VAD (~2 MB)")
        urllib.request.urlretrieve(VAD_URL, path)
    return path


def build_recognizer(model_dir: Path, threads: int) -> sherpa_onnx.OfflineRecognizer:
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=str(model_dir / "encoder.int8.onnx"),
        decoder=str(model_dir / "decoder.int8.onnx"),
        joiner=str(model_dir / "joiner.int8.onnx"),
        tokens=str(model_dir / "tokens.txt"),
        model_type="nemo_transducer",  # required for Parakeet TDT
        num_threads=threads,
        decoding_method="greedy_search",
        provider="cpu",
    )


def build_vad(vad_path: Path, args) -> sherpa_onnx.VoiceActivityDetector:
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(vad_path)
    cfg.silero_vad.threshold = args.vad_threshold
    cfg.silero_vad.min_silence_duration = args.silence  # pause length that ends a sentence
    cfg.silero_vad.min_speech_duration = 0.15
    cfg.silero_vad.max_speech_duration = args.max_utterance  # force-split very long speech
    cfg.silero_vad.window_size = WINDOW
    cfg.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def transcribe(recognizer, samples: np.ndarray) -> str:
    stream = recognizer.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    recognizer.decode_stream(stream)
    return stream.result.text.strip()


def show_partial(text: str):
    cols = shutil.get_terminal_size((100, 20)).columns
    line = "… " + text
    if len(line) > cols - 1:  # keep the newest words visible on one line
        line = "…" + line[-(cols - 2):]
    sys.stdout.write("\r\033[K\033[90m" + line + "\033[0m")
    sys.stdout.flush()


def show_final(text: str, audio_sec: float, decode_sec: float):
    sys.stdout.write("\r\033[K")
    if text:
        print(f"{text}   \033[90m[{audio_sec:.1f}s audio, decoded in {decode_sec:.2f}s]\033[0m")
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)),
                   help="CPU threads for the ASR model (default: half your cores, max 8)")
    p.add_argument("--partial-interval", type=float, default=0.5,
                   help="seconds between live partial updates (lower = snappier, more CPU)")
    p.add_argument("--silence", type=float, default=0.5,
                   help="seconds of silence that end a sentence")
    p.add_argument("--max-utterance", type=float, default=15.0,
                   help="force a sentence break after this many seconds of continuous speech")
    p.add_argument("--vad-threshold", type=float, default=0.5)
    p.add_argument("--device", type=int, default=None, help="input device index")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    os.system("")  # enables ANSI colours on Windows terminals

    model_dir = ensure_asr_model()
    vad_path = ensure_vad_model()

    print(f"[setup] Loading model on CPU with {args.threads} threads...")
    t0 = time.time()
    recognizer = build_recognizer(model_dir, args.threads)
    vad = build_vad(vad_path, args)
    print(f"[setup] Ready in {time.time() - t0:.1f}s. Speak! (Ctrl+C to stop)\n")

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    preroll_samples = int(0.3 * SAMPLE_RATE)      # keep a bit of audio before speech starts
    partial_every = int(args.partial_interval * SAMPLE_RATE)

    buf = collections.deque()   # recent audio chunks
    buf_len = 0                 # total samples in buf
    since_partial = 0
    last_partial_text = ""
    pending = np.zeros(0, dtype=np.float32)  # leftover samples < WINDOW

    try:
        with sd.InputStream(channels=1, dtype="float32", samplerate=SAMPLE_RATE,
                            blocksize=WINDOW, device=args.device, callback=callback):
            while True:
                chunk = audio_q.get()
                pending = np.concatenate([pending, chunk])

                while len(pending) >= WINDOW:
                    frame, pending = pending[:WINDOW], pending[WINDOW:]
                    vad.accept_waveform(frame)

                    buf.append(frame)
                    buf_len += len(frame)

                    speaking = vad.is_speech_detected()

                    # While idle, only keep a short pre-roll so memory stays tiny.
                    if not speaking and vad.empty():
                        while buf_len - len(buf[0]) >= preroll_samples and len(buf) > 1:
                            buf_len -= len(buf.popleft())
                        continue

                    # ---- live partial (only if we're not falling behind) ----
                    since_partial += len(frame)
                    if speaking and since_partial >= partial_every and audio_q.qsize() < 8:
                        since_partial = 0
                        text = transcribe(recognizer, np.concatenate(buf))
                        if text and text != last_partial_text:
                            last_partial_text = text
                            show_partial(text)

                    # ---- finished sentence from VAD ----
                    while not vad.empty():
                        seg = np.asarray(vad.front.samples, dtype=np.float32)
                        vad.pop()
                        t1 = time.time()
                        text = transcribe(recognizer, seg)
                        show_final(text, len(seg) / SAMPLE_RATE, time.time() - t1)
                        buf.clear()
                        buf_len = 0
                        since_partial = 0
                        last_partial_text = ""

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()