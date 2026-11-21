"""Lightweight frame-level speaker overlap and turn-boundary detection via pyannote-segmentation-3.0.

Runs the 5.9 MB FP32 ONNX powerset segmentation model (~17 ms frame resolution, ~60 ms per 10s
chunk on 1 CPU thread) strictly on the low-priority background diarizer thread for Parakeet mode:

1. Detects simultaneous 2-speaker overlap intervals (`overlap` badges in UI & saved transcripts).
2. Masks overlapping frames out of TitaNet voiceprint embeddings so crosstalk never blurs speaker
   clusters or bridges distinct voices.
3. Provides neural speaker-change timestamps (~17 ms) to sharpen mid-chunk handoffs alongside
   Parakeet's native word timestamps.

Gracefully degrades to a no-op if ``onnxruntime`` or the segmentation model is unavailable.
"""

from __future__ import annotations

import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .speech import SAMPLE_RATE

SEG_MODEL_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
SEG_MODEL_FILE = "model.onnx"
SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)

WIN_SEC = 10.0
WIN_SAMPLES = int(WIN_SEC * SAMPLE_RATE)
HOP_SAMPLES = int(5.0 * SAMPLE_RATE)
OVERLAP_FRAME_THRESH = 0.45
MIN_OVERLAP_SPAN = 0.35
MIN_CLEAN_EMBED_SEC = 1.0

_PAIR_TO_CLASS = {
    frozenset((0, 1)): 4,
    frozenset((0, 2)): 5,
    frozenset((1, 2)): 6,
}
_SPK_PERMS = (
    (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
)
_CLASS_PERMS = {
    p: [
        0,
        1 + p[0], 1 + p[1], 1 + p[2],
        _PAIR_TO_CLASS[frozenset((p[0], p[1]))],
        _PAIR_TO_CLASS[frozenset((p[0], p[2]))],
        _PAIR_TO_CLASS[frozenset((p[1], p[2]))],
    ]
    for p in _SPK_PERMS
}


def ensure_segmentation_model(models_dir: Path, log=print) -> Path:
    """Ensure pyannote-segmentation-3.0 INT8 ONNX (~1.47 MB, MIT) exists inside *models_dir*."""
    target_dir = models_dir / SEG_MODEL_DIR
    model_path = target_dir / SEG_MODEL_FILE
    if model_path.exists() and model_path.stat().st_size > 100_000:
        return model_path

    models_dir.mkdir(parents=True, exist_ok=True)
    archive = models_dir / f"{SEG_MODEL_DIR}.tar.bz2"
    part = archive.with_suffix(".bz2.part")
    log("Downloading pyannote overlap segmentation model (~7 MB, first run only)...")
    try:
        req = urllib.request.Request(SEG_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            while True:
                buf = resp.read(256 * 1024)
                if not buf:
                    break
                out.write(buf)
        part.replace(archive)
        base_resolved = models_dir.resolve()
        with tarfile.open(archive, "r:bz2") as tf:
            members = []
            for m in tf.getmembers():
                dest = (models_dir / m.name).resolve()
                try:
                    dest.relative_to(base_resolved)
                except ValueError:
                    raise ValueError(f"Unsafe tar path: {m.name}")
                members.append(m)
            tf.extractall(models_dir, members=members)
    finally:
        for p in (part, archive):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    if not model_path.exists():
        raise FileNotFoundError(f"Expected {model_path} after extracting {SEG_MODEL_URL}")
    return model_path


@dataclass
class SegmentAnalysis:
    """Frame-level powerset segmentation result for one audio chunk."""

    dur: float
    frame_dur: float
    probs: np.ndarray                     # (n_frames, 7) powerset probabilities
    ov_prob: np.ndarray                   # (n_frames,) sum of 2-speaker overlap classes [4:7]
    overlap_spans: list[tuple[float, float]]
    change_times: list[float]

    def has_overlap_in(self, t_a: float, t_b: float, min_overlap_sec: float = 0.40) -> tuple[bool, float]:
        """Return `(has_overlap, total_overlap_seconds)` inside `[t_a, t_b]`."""
        total = 0.0
        for oa, ob in self.overlap_spans:
            lo = max(t_a, oa)
            hi = min(t_b, ob)
            if hi > lo:
                total += (hi - lo)
        return (total >= min_overlap_sec), total

    def strip_overlap(self, slice_samples: np.ndarray, t_a: float, t_b: float
                      ) -> tuple[np.ndarray, float, bool]:
        """Return `(clean_audio, clean_voiced_sec, was_stripped)` for slice `[t_a, t_b]`.

        Removes frames with simultaneous 2-speaker overlap before TitaNet embedding whenever at
        least ``MIN_CLEAN_EMBED_SEC`` of clean single-speaker audio remains.
        """
        n_frames = len(self.ov_prob)
        if n_frames == 0 or len(slice_samples) == 0:
            return slice_samples, len(slice_samples) / SAMPLE_RATE, False

        fa = max(0, int(t_a / self.frame_dur))
        fb = min(n_frames, max(fa + 1, int(np.ceil(t_b / self.frame_dur))))
        ov_sub = self.ov_prob[fa:fb]
        single_sub = self.probs[fa:fb, 1:4].sum(axis=1)
        clean_mask = ov_sub < OVERLAP_FRAME_THRESH
        clean_voiced_sec = float(np.sum(clean_mask & (single_sub >= 0.40))) * self.frame_dur

        if bool(np.all(clean_mask)):
            return slice_samples, clean_voiced_sec, False
        if clean_voiced_sec < MIN_CLEAN_EMBED_SEC:
            return slice_samples, clean_voiced_sec, False

        # Collect contiguous non-overlapping audio spans inside slice_samples
        n_s = len(slice_samples)
        chunks: list[np.ndarray] = []
        i = 0
        m_len = len(clean_mask)
        while i < m_len:
            if not clean_mask[i]:
                i += 1
                continue
            j = i
            while j < m_len and clean_mask[j]:
                j += 1
            s0 = max(0, int(((fa + i) * self.frame_dur - t_a) * SAMPLE_RATE))
            s1 = min(n_s, int(((fa + j) * self.frame_dur - t_a) * SAMPLE_RATE))
            if s1 - s0 >= int(0.12 * SAMPLE_RATE):
                chunks.append(slice_samples[s0:s1])
            i = j

        if not chunks:
            return slice_samples, clean_voiced_sec, False
        clean_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        if len(clean_audio) < int(MIN_CLEAN_EMBED_SEC * SAMPLE_RATE):
            return slice_samples, clean_voiced_sec, False
        return clean_audio, clean_voiced_sec, True


class OverlapSegmenter:
    """Thread-safe wrapper around pyannote-segmentation-3.0 INT8 ONNX."""

    def __init__(self, model_path: Path, threads: int = 2):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self.in_name = self.session.get_inputs()[0].name

    def analyze(self, samples: np.ndarray) -> Optional[SegmentAnalysis]:
        """Run powerset segmentation on 16 kHz mono float32 *samples*."""
        n = len(samples)
        dur = n / SAMPLE_RATE
        if dur < 1.0:
            return None
        x = np.asarray(samples, dtype=np.float32)
        if float(np.max(np.abs(x))) < 1e-4:
            return None
        # Gently normalize quiet recordings (~-40 dBFS) for stable SincNet input without mutating samples
        rms = float(np.sqrt(np.mean(x * x)))
        if 1e-4 <= rms < 0.08:
            gain = min(10.0, 0.08 / rms)
            x = np.clip(x * gain, -1.0, 1.0)

        if n <= WIN_SAMPLES:
            chunk = np.pad(x, (0, WIN_SAMPLES - n)) if n < WIN_SAMPLES else x
            log_p = self.session.run(None, {self.in_name: chunk[None, None, :]})[0][0]
            probs_full = np.exp(log_p).astype(np.float32)
            frame_dur = WIN_SEC / len(probs_full)
            n_valid = max(1, min(len(probs_full), int(round(dur / frame_dur))))
            probs = probs_full[:n_valid]
        else:
            # Slide 10s windows with 5s hop and average overlapping frame probabilities
            first_chunk = x[:WIN_SAMPLES]
            first_probs = np.exp(
                self.session.run(None, {self.in_name: first_chunk[None, None, :]})[0][0]
            ).astype(np.float64)
            f_per_win = len(first_probs)
            frame_dur = WIN_SEC / f_per_win
            total_frames = max(f_per_win, int(round(dur / frame_dur)))
            acc = np.zeros((total_frames, 7), dtype=np.float64)
            cnt = np.zeros((total_frames, 1), dtype=np.float64)
            acc[:f_per_win] += first_probs
            cnt[:f_per_win] += 1.0

            pos = HOP_SAMPLES
            while pos < n:
                end = min(n, pos + WIN_SAMPLES)
                seg = x[pos:end]
                if len(seg) < WIN_SAMPLES:
                    seg = np.pad(seg, (0, WIN_SAMPLES - len(seg)))
                w_probs = np.exp(
                    self.session.run(None, {self.in_name: seg[None, None, :]})[0][0]
                ).astype(np.float64)
                f0 = int(round((pos / SAMPLE_RATE) / frame_dur))
                valid_w = max(1, min(f_per_win, int(round(((end - pos) / SAMPLE_RATE) / frame_dur))))
                f1 = min(total_frames, f0 + valid_w)
                take = f1 - f0
                if take > 0:
                    # Align permutation-invariant local speaker slots (1..3) to accumulated overlap zone
                    ov_mask = cnt[f0:f1, 0] > 0
                    if np.any(ov_mask):
                        ref_s = acc[f0:f1][ov_mask, 1:4] / cnt[f0:f1][ov_mask]
                        cur_s = w_probs[:take][ov_mask, 1:4]
                        best_p = max(_SPK_PERMS, key=lambda p: float(np.sum(ref_s * cur_s[:, p])))
                        w_probs = w_probs[:, _CLASS_PERMS[best_p]]
                    acc[f0:f1] += w_probs[:take]
                    cnt[f0:f1] += 1.0
                if end >= n:
                    break
                pos += HOP_SAMPLES
            probs = (acc / np.maximum(1.0, cnt)).astype(np.float32)

        ov_prob = probs[:, 4:7].sum(axis=1)
        overlap_spans = _extract_overlap_spans(ov_prob, frame_dur)
        change_times = _extract_change_times(probs, ov_prob, frame_dur)
        return SegmentAnalysis(
            dur=dur,
            frame_dur=frame_dur,
            probs=probs,
            ov_prob=ov_prob,
            overlap_spans=overlap_spans,
            change_times=change_times,
        )


def _extract_overlap_spans(ov_prob: np.ndarray, frame_dur: float) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    in_ov = False
    st = 0.0
    for i, p in enumerate(ov_prob):
        t = i * frame_dur
        if p >= OVERLAP_FRAME_THRESH and not in_ov:
            in_ov = True
            st = t
        elif p < OVERLAP_FRAME_THRESH and in_ov:
            in_ov = False
            raw.append((st, t))
    if in_ov:
        raw.append((st, len(ov_prob) * frame_dur))

    merged: list[tuple[float, float]] = []
    for a, b in raw:
        if merged and a - merged[-1][1] <= 0.15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if (b - a) >= MIN_OVERLAP_SPAN]


def _extract_change_times(probs: np.ndarray, ov_prob: np.ndarray, frame_dur: float) -> list[float]:
    """Extract clean neural speaker-change timestamps from powerset single-speaker runs."""
    single = probs[:, 1:4]
    max_s = single.max(axis=1)
    arg_s = single.argmax(axis=1)
    dom = np.where((max_s >= 0.50) & (ov_prob < OVERLAP_FRAME_THRESH), arg_s, -1)

    # Build contiguous single-speaker runs, bridging brief (<0.45s) silence/overlap gaps of same speaker
    runs: list[list] = []  # [spk, start_f, end_f, active_frames]
    bridge_f = max(1, int(round(0.45 / frame_dur)))
    for i, d in enumerate(dom):
        d_int = int(d)
        if d_int < 0:
            continue
        if runs and runs[-1][0] == d_int and (i - runs[-1][2]) <= bridge_f:
            runs[-1][2] = i + 1
            runs[-1][3] += 1
        else:
            runs.append([d_int, i, i + 1, 1])

    min_active_f = max(1, int(round(0.60 / frame_dur)))
    solid = [r for r in runs if r[3] >= min_active_f]
    if len(solid) < 2:
        return []

    cuts: list[float] = []
    for r1, r2 in zip(solid[:-1], solid[1:]):
        if r1[0] != r2[0]:
            t_cut = 0.5 * (r1[2] + r2[1]) * frame_dur
            cuts.append(t_cut)
    return cuts


def build_overlap_segmenter(models_dir: Path, threads: int = 2, log=print) -> Optional[OverlapSegmenter]:
    """Load pyannote-segmentation-3.0 if onnxruntime is available; return None on graceful fallback."""
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return None
    try:
        model_path = ensure_segmentation_model(models_dir, log=log)
        return OverlapSegmenter(model_path, threads=threads)
    except Exception as e:
        log(f"[diarize] Overlap segmentation unavailable ({type(e).__name__}: {e})", "warn")
        return None
