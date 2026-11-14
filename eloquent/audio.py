"""Audio capture sources: microphone, Windows WASAPI loopback, and streaming resampling."""

import queue
import sys
import threading
from typing import Callable, Optional

import numpy as np

from .speech import SAMPLE_RATE, WINDOW

AUDIO_SOURCES = (
    ("mic", "Microphone only"),
    ("system", "System audio (videos / calls)"),
    ("both", "Mic + System audio (meetings)"),
)

AUDIO_SOURCE_LABELS = dict(AUDIO_SOURCES)


def _make_lowpass_kernel(src_rate: float, dst_rate: int) -> Optional[np.ndarray]:
    if src_rate <= dst_rate * 1.01:
        return None
    cutoff = 0.45 * dst_rate / src_rate
    n = np.arange(-10, 11, dtype=np.float64)
    h = np.sinc(2.0 * cutoff * n) * np.hanning(len(n))
    h /= np.sum(h)
    return h.astype(np.float32)


class StreamResampler:
    """Anti-aliased stateful resampler that yields fixed-size 16 kHz mono float32 frames."""

    def __init__(self, src_rate: float, dst_rate: int = SAMPLE_RATE, frame_size: int = WINDOW):
        self.src_rate = float(src_rate)
        self.dst_rate = int(dst_rate)
        self.frame_size = int(frame_size)
        self.passthrough = abs(self.src_rate - self.dst_rate) < 1.0
        self.step = self.src_rate / self.dst_rate
        self._lpf = None if self.passthrough else _make_lowpass_kernel(self.src_rate, self.dst_rate)
        self._lpf_tail = np.zeros(0, dtype=np.float32)
        self._prev = np.zeros(0, dtype=np.float32)
        self._pos = 0.0
        self._out = np.zeros(0, dtype=np.float32)

    def push(self, mono: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(mono, dtype=np.float32).ravel()
        if len(x) == 0:
            return []

        if self.passthrough:
            resampled = x
        else:
            if self._lpf is not None:
                buf = np.concatenate([self._lpf_tail, x])
                taps = len(self._lpf)
                if len(buf) < taps:
                    self._lpf_tail = buf
                    return []
                filt = np.convolve(buf, self._lpf, mode="valid").astype(np.float32, copy=False)
                self._lpf_tail = buf[-(taps - 1):].copy()
            else:
                filt = x

            src = np.concatenate([self._prev, filt]) if len(self._prev) else filt
            limit = len(src) - 1
            if self._pos >= limit:
                self._prev = src
                return []
            est = int(np.floor((limit - self._pos) / self.step)) + 1
            pos = self._pos + np.arange(max(0, est), dtype=np.float64) * self.step
            while len(pos) and pos[-1] >= limit:
                pos = pos[:-1]
            if len(pos):
                i0 = np.floor(pos).astype(np.int64)
                frac = (pos - i0).astype(np.float32)
                resampled = src[i0] * (1.0 - frac) + src[i0 + 1] * frac
                next_pos = float(pos[-1] + self.step)
                drop = min(len(src) - 1, int(np.floor(next_pos)))
                self._prev = src[drop:].copy()
                self._pos = next_pos - drop
            else:
                resampled = np.zeros(0, dtype=np.float32)
                self._prev = src

        if len(resampled) == 0:
            return []
        self._out = np.concatenate([self._out, resampled]) if len(self._out) else resampled
        frames = []
        n = len(self._out) // self.frame_size
        if n > 0:
            end = n * self.frame_size
            block = self._out[:end].reshape(n, self.frame_size)
            frames = [block[i].copy() for i in range(n)]
            self._out = self._out[end:].copy()
        return frames


def to_mono(data: np.ndarray, channels: int) -> np.ndarray:
    """Convert raw callback audio `(frames * channels)` or `(frames, channels)` to float32 mono."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr[:, 0] if arr.shape[1] == 1 else arr.mean(axis=1)
    if channels <= 1:
        return arr
    usable = (len(arr) // channels) * channels
    return arr[:usable].reshape(-1, channels).mean(axis=1)


def list_input_devices() -> list[tuple[int, str]]:
    """Return `[(index, name), ...]` for default-host microphones."""
    out = []
    try:
        import sounddevice as sd
        default_api = sd.query_devices(kind="input")["hostapi"]
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0 and d["hostapi"] == default_api:
                out.append((i, d["name"]))
    except Exception:
        pass
    return out


def list_loopback_devices() -> list[tuple[int, str, bool]]:
    """Return `[(index, clean_name, is_default), ...]` for Windows WASAPI loopback outputs."""
    if sys.platform != "win32":
        return []
    out = []
    try:
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            default_idx = None
            try:
                default_idx = p.get_default_wasapi_loopback()["index"]
            except Exception:
                pass
            for d in p.get_loopback_device_info_generator():
                name = str(d.get("name", "")).removesuffix(" [Loopback]").strip()
                out.append((int(d["index"]), name, d["index"] == default_idx))
        finally:
            p.terminate()
    except Exception:
        pass
    return out


def _resolve_loopback_device(p, target_name: Optional[str]):
    default_dev = None
    try:
        default_dev = p.get_default_wasapi_loopback()
    except Exception:
        pass
    if not target_name:
        if default_dev is None:
            raise RuntimeError("No default Windows playback device found for system audio capture.")
        return default_dev

    req = target_name.removesuffix(" [Loopback]").strip().lower()
    for d in p.get_loopback_device_info_generator():
        cand = str(d.get("name", "")).removesuffix(" [Loopback]").strip().lower()
        if cand == req:
            return d
    if default_dev is not None:
        return default_dev
    raise RuntimeError(f"Playback device not found: {target_name}")


class MicCapture:
    """Capture mono 16 kHz frames from a microphone via sounddevice (resampling automatically if needed)."""

    def __init__(self, device: Optional[int], out_q: queue.Queue, tag: str = "mic",
                 on_status: Optional[Callable[[str], None]] = None):
        import sounddevice as sd
        self._sd = sd
        self.device = device
        self.out_q = out_q
        self.tag = tag
        self.on_status = on_status
        self.stream = None
        self.resampler = None

    def start(self):
        sd = self._sd
        try:
            dev_info = sd.query_devices(self.device, "input")
            native_rate = float(dev_info.get("default_samplerate") or SAMPLE_RATE)
        except Exception:
            native_rate = float(SAMPLE_RATE)

        rate = SAMPLE_RATE
        try:
            sd.check_input_settings(device=self.device, channels=1, dtype="float32", samplerate=rate)
            block = WINDOW
        except Exception:
            rate = int(native_rate)
            block = max(256, int(round(WINDOW * rate / SAMPLE_RATE)))

        self.resampler = StreamResampler(rate, SAMPLE_RATE, WINDOW)

        def callback(indata, frames, time_info, status):
            if status and self.on_status:
                self.on_status(f"[mic] {status}")
            mono = indata[:, 0].copy() if indata.shape[1] == 1 else indata.mean(axis=1)
            for f in self.resampler.push(mono):
                self.out_q.put((self.tag, f))

        self.stream = sd.InputStream(
            channels=1, dtype="float32", samplerate=rate,
            blocksize=block, device=self.device, callback=callback,
        )
        self.stream.start()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


class LoopbackCapture:
    """Capture 16 kHz mono frames from Windows WASAPI speaker loopback via PyAudioWPatch."""

    def __init__(self, target_name: Optional[str], out_q: queue.Queue, tag: str = "system",
                 on_log: Optional[Callable[[str], None]] = None):
        if sys.platform != "win32":
            raise RuntimeError("System audio loopback capture is currently only supported on Windows.")
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError(
                "System audio capture requires PyAudioWPatch (run setup.bat or pip install PyAudioWPatch)."
            ) from e
        self._pyaudio = pyaudio
        self.target_name = target_name
        self.out_q = out_q
        self.tag = tag
        self.on_log = on_log
        self.p = None
        self.stream = None
        self.resampler = None
        self.device_name = ""
        self._lock = threading.Lock()

    def start(self):
        pyaudio = self._pyaudio
        self.p = pyaudio.PyAudio()
        try:
            dev = _resolve_loopback_device(self.p, self.target_name)
            self.device_name = str(dev.get("name", "")).removesuffix(" [Loopback]").strip()
            channels = max(1, int(dev["maxInputChannels"]))
            rate = int(round(float(dev["defaultSampleRate"])))
            block = max(256, int(round(WINDOW * rate / SAMPLE_RATE)))
            self.resampler = StreamResampler(rate, SAMPLE_RATE, WINDOW)

            def callback(in_data, frame_count, time_info, status_flags):
                if in_data:
                    raw = np.frombuffer(in_data, dtype=np.float32)
                    mono = to_mono(raw, channels)
                    for f in self.resampler.push(mono):
                        self.out_q.put((self.tag, f))
                return (None, pyaudio.paContinue)

            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(dev["index"]),
                frames_per_buffer=block,
                stream_callback=callback,
            )
            self.stream.start_stream()
            if self.on_log:
                self.on_log(f"System audio loopback active: {self.device_name} ({rate} Hz, {channels}ch)")
        except Exception:
            self.close()
            raise

    def close(self):
        with self._lock:
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
            if self.p is not None:
                try:
                    self.p.terminate()
                except Exception:
                    pass
                self.p = None
