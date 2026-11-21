"""Zero-latency-impact multi-speaker diarization using sherpa-onnx (NVIDIA NeMo TitaNet-Small).

Architecture
------------
1. The audio/ASR thread (`Engine._drain`) emits finalized text to `State` immediately
   and pushes `(seg_id, start_sec, samples, text, words)` onto a queue for `Diarizer`.
2. `Diarizer` runs on a dedicated `BELOW_NORMAL` priority daemon thread with the speaker
   embedding extractor pinned to `num_threads=1`, so Parakeet never stalls.
3. For long VAD turns (> 4.8 s), sliding 1.8 s voiceprint windows detect mid-chunk speaker
   handoffs (when two speakers talk back-to-back without pausing) and snap splits onto
   Parakeet's native word timestamps and punctuation boundaries.
4. Duration-weighted UPGMA (average-linkage) clustering over confirmed utterance embeddings
   keeps speaker IDs (`Speaker 1`, `Speaker 2`, ...) accurate and prevents drift.
"""

import ctypes
import queue
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import sherpa_onnx

from .speech import SAMPLE_RATE

DIARIZE_MODEL_NAME = "nemo_en_titanet_small.onnx"
DIARIZE_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/" + DIARIZE_MODEL_NAME
)

MIN_CONFIRM_DUR = 1.5         # minimum voiced speech (s) required to seed a new speaker cluster
MERGE_THRESHOLD = 0.50        # UPGMA average-linkage cosine similarity merge threshold
CENTROID_MERGE = 0.51         # centroid cosine similarity merge threshold for drift consolidation
ESTABLISHED_MERGE = 0.58      # stricter centroid merge when both clusters already have >= 6s speech
ESTABLISHED_MULTI_MERGE = 0.64  # centroid merge when both have >= 6s and >= 2 confirmed turns
ESTABLISHED_MEDIAN_MIN = 0.47   # minimum median cross-turn similarity for >= 6s established merges
ESTABLISHED_MULTI_MEDIAN = 0.52 # minimum median cross-turn similarity for multi-turn established merges
SPLIT_CROSS_THRESHOLD = 0.36  # 2-window sliding acoustic change-point threshold inside a chunk
SLICE_MERGE_THRESHOLD = 0.45 # direct slice verification merge threshold
SHORT_ATTACH_THRESHOLD = 0.30
SENTENCE_END = (".", "?", "!", "…", "؟")
RESERVOIR_LIMIT = 160         # max confirmed turns fed to UPGMA (keeps O(1 ms) on multi-hour calls)
RESERVOIR_RECENT = 32         # always keep newest confirmed turns in UPGMA
RESERVOIR_LOCKED_PER_SPK = 16 # locked anchor turns reserved per user-named speaker
RESERVOIR_MAX_PER_SPK = 24    # first-pass stratified quota per speaker when > RESERVOIR_LIMIT


def ensure_speaker_model(models_dir: Path, log=print) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / DIARIZE_MODEL_NAME
    if not path.exists():
        log(f"Downloading speaker model ({DIARIZE_MODEL_NAME}, ~30 MB, first run only)...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(DIARIZE_MODEL_URL, tmp)
        tmp.replace(path)
    return path


def build_speaker_extractor(model_path: Path, threads: int = 1):
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=max(1, int(threads)),
        provider="cpu",
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def compute_embedding(extractor, samples: np.ndarray, min_rms: float = 0.002) -> Optional[np.ndarray]:
    """Extract an L2-normalized speaker embedding vector, or None if clip is silent/too short."""
    if len(samples) < int(0.35 * SAMPLE_RATE):
        return None
    if float(np.sqrt(np.mean(samples * samples))) < min_rms:
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(SAMPLE_RATE, samples)
    stream.input_finished()
    vec = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8 or not np.isfinite(norm):
        return None
    return vec / norm


def _lower_thread_priority() -> None:
    if sys.platform == "win32":
        try:
            # THREAD_PRIORITY_BELOW_NORMAL (-1) on GetCurrentThread (-2)
            ctypes.windll.kernel32.SetThreadPriority(-2, -1)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Mid-chunk turn splitter (pure NumPy + word spans)
# --------------------------------------------------------------------------- #
def _pick_word_cut(words: list[dict], start_idx: int, end_idx: int, t_target: float,
                   min_side_dur: float = 1.25) -> Optional[int]:
    """Choose best split index `cut` in `(start_idx, end_idx)` so parts are `[start:cut]` and `[cut:end]`."""
    best_idx, best_score = None, -1e9
    for idx in range(start_idx + 2, end_idx - 1):
        prev_w, next_w = words[idx - 1], words[idx]
        boundary_t = 0.5 * (prev_w["t1"] + next_w["t0"])
        dist = abs(boundary_t - t_target)
        if dist > 1.25:
            continue
        score = -dist * 0.7
        w_txt = prev_w["w"].rstrip("\"'”)")
        if w_txt.endswith(SENTENCE_END):
            score += 2.6
        elif w_txt.endswith((",", ";", ":")):
            score += 0.7
        gap = max(0.0, next_w["t0"] - prev_w["t1"])
        score += min(gap, 0.8) * 2.2
        # require each side to span enough audio to yield a stable embedding
        left_dur = prev_w["t1"] - words[start_idx]["t0"]
        right_dur = words[end_idx - 1]["t1"] - next_w["t0"]
        if left_dur < min_side_dur or right_dur < min_side_dur:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _find_window_changes(w_vecs: list[np.ndarray], w_centers: list[float],
                         cross_thresh: float = SPLIT_CROSS_THRESHOLD) -> list[float]:
    """Detect local minima in adjacent 2-window cosine similarity indicating speaker handoffs."""
    n = len(w_vecs)
    if n < 5:
        return []
    mat = np.stack(w_vecs)
    scores = []
    for k in range(2, n - 1):
        l_vec = mat[max(0, k - 2):k].mean(axis=0)
        r_vec = mat[k:min(n, k + 2)].mean(axis=0)
        ln, rn = float(np.linalg.norm(l_vec)), float(np.linalg.norm(r_vec))
        sim = float(np.dot(l_vec / max(ln, 1e-8), r_vec / max(rn, 1e-8)))
        t_mid = 0.5 * (w_centers[k - 1] + w_centers[k])
        scores.append((k, t_mid, sim))

    cuts = []
    for i, (k, t_mid, sim) in enumerate(scores):
        if sim >= cross_thresh:
            continue
        left_ok = (i == 0) or sim <= scores[i - 1][2]
        right_ok = (i == len(scores) - 1) or sim <= scores[i + 1][2]
        if not (left_ok and right_ok):
            continue
        if cuts and (t_mid - cuts[-1]) < 1.3:
            continue
        cuts.append(t_mid)
    return cuts


def _find_linguistic_cuts(words: list[dict], min_side_dur: float = 1.25,
                          max_cuts: int = 8, min_words: int = 6) -> list[int]:
    """Return candidate word split indices at sentence endings or pauses (`>= 0.14 s`)."""
    n = len(words)
    if n < min_words:
        return []
    candidates = []
    for idx in range(2, n - 1):
        prev_w, next_w = words[idx - 1], words[idx]
        left_dur = prev_w["t1"] - words[0]["t0"]
        right_dur = words[-1]["t1"] - next_w["t0"]
        if left_dur < min_side_dur or right_dur < min_side_dur:
            continue
        w_txt = prev_w["w"].rstrip("\"'”)")
        gap = max(0.0, next_w["t0"] - prev_w["t1"])
        if w_txt.endswith(SENTENCE_END) or gap >= 0.14:
            score = (2.0 if w_txt.endswith(SENTENCE_END) else 0.0) + min(gap, 0.8) * 2.0
            t_mid = 0.5 * (prev_w["t1"] + next_w["t0"])
            candidates.append((score, t_mid, idx))
    # Keep strongest distinct boundaries spaced cleanly apart
    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = []
    for _, t_mid, idx in candidates:
        if all(abs(t_mid - words[p]["t0"]) >= 1.10 for p in picked):
            picked.append(idx)
            if len(picked) >= max_cuts:
                break
    return sorted(picked)


def _detect_voiced_islands(samples: np.ndarray) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    """Strip low-energy silence when word timestamps are absent and return `(trimmed, voiced_dur, islands)`."""
    n = len(samples)
    dur = n / SAMPLE_RATE
    frame_n = int(0.02 * SAMPLE_RATE)
    n_frames = n // frame_n
    if n < int(0.06 * SAMPLE_RATE) or n_frames < 2:
        return samples, dur, [(0.0, dur)]

    frames = samples[:n_frames * frame_n].reshape(n_frames, frame_n)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    thr = max(0.0006, min(0.025, float(np.percentile(rms, 85)) * 0.20))
    active = (rms >= thr).copy()

    # Bridge brief consonant/micro pauses <= 60 ms (3 frames)
    false_run = 0
    seen_true = False
    for i in range(n_frames):
        if active[i]:
            if seen_true and 0 < false_run <= 3:
                active[i - false_run:i] = True
            seen_true = True
            false_run = 0
        elif seen_true:
            false_run += 1

    # Drop isolated single-frame clicks
    for i in range(n_frames):
        if active[i] and not ((i > 0 and active[i - 1]) or (i + 1 < n_frames and active[i + 1])):
            active[i] = False

    raw_islands: list[tuple[float, float]] = []
    i = 0
    while i < n_frames:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n_frames and active[j]:
            j += 1
        raw_islands.append((i * frame_n / SAMPLE_RATE, min(dur, j * frame_n / SAMPLE_RATE)))
        i = j

    islands: list[tuple[float, float]] = []
    for t0, t1 in raw_islands:
        if islands and (t0 - islands[-1][1]) < 0.12:
            islands[-1] = (islands[-1][0], max(islands[-1][1], t1))
        else:
            islands.append((t0, t1))

    if not islands:
        return samples, dur, [(0.0, dur)]

    voiced_dur = sum(t1 - t0 for t0, t1 in islands)
    pad = int(0.05 * SAMPLE_RATE)
    spans_s: list[list[int]] = []
    for t0, t1 in islands:
        s0 = max(0, int(t0 * SAMPLE_RATE) - pad)
        s1 = min(n, int(t1 * SAMPLE_RATE) + pad)
        if spans_s and s0 <= spans_s[-1][1]:
            spans_s[-1][1] = max(spans_s[-1][1], s1)
        else:
            spans_s.append([s0, s1])

    trimmed = np.concatenate([samples[a:b] for a, b in spans_s]) if len(spans_s) > 1 else samples[spans_s[0][0]:spans_s[0][1]]
    if voiced_dur >= 1.8:
        guard = int(0.08 * SAMPLE_RATE)
        if len(trimmed) > 2 * guard + int(1.2 * SAMPLE_RATE):
            trimmed = trimmed[guard:-guard]
    return trimmed, voiced_dur, islands


def _synthesize_arabic_word_spans(samples: np.ndarray, text: str) -> list[dict]:
    """Align Arabic words onto voiced audio islands so mid-line speaker handoffs can split accurately."""
    tokens = [w for w in (text or "").strip().split() if w]
    if not tokens:
        return []
    _, _, islands = _detect_voiced_islands(samples)
    while len(islands) > len(tokens):
        gaps = [islands[k + 1][0] - islands[k][1] for k in range(len(islands) - 1)]
        idx = int(np.argmin(gaps))
        islands[idx:idx + 2] = [(islands[idx][0], islands[idx + 1][1])]

    weights = [max(2, len(t)) for t in tokens]
    cum_w = [0]
    for w in weights:
        cum_w.append(cum_w[-1] + w)
    total_w = max(1, cum_w[-1])

    island_durs = [max(0.05, t1 - t0) for t0, t1 in islands]
    total_d = max(1e-6, sum(island_durs))

    # Map each acoustic pause boundary onto the nearest token split, preferring punctuation nearby
    cuts = [0]
    acc_d = 0.0
    punct = SENTENCE_END + ("،", ",", ";", ":")
    for k in range(len(islands) - 1):
        acc_d += island_durs[k]
        target_w = (acc_d / total_d) * total_w
        lo = cuts[-1] + 1
        hi = len(tokens) - (len(islands) - 1 - k)
        if lo >= hi:
            cuts.append(lo)
            continue
        best = min(range(lo, hi + 1), key=lambda c: abs(cum_w[c] - target_w))
        p_cands = [
            c for c in range(max(lo, best - 2), min(hi, best + 2) + 1)
            if tokens[c - 1].rstrip("\"'”)").endswith(punct)
        ]
        if p_cands:
            best = min(p_cands, key=lambda c: (abs(cum_w[c] - target_w), not tokens[c - 1].rstrip("\"'”)").endswith(SENTENCE_END)))
        cuts.append(best)
    cuts.append(len(tokens))

    spans: list[dict] = []
    for (a0, a1), (c0, c1) in zip(islands, zip(cuts[:-1], cuts[1:])):
        chunk = tokens[c0:c1]
        if not chunk:
            continue
        cw = [max(2, len(t)) for t in chunk]
        ctot = max(1, sum(cw))
        span_d = max(0.04, a1 - a0)
        cur_t = a0
        for tok, w in zip(chunk, cw):
            dt = span_d * (w / ctot)
            spans.append({"w": tok, "t0": cur_t, "t1": cur_t + dt})
            cur_t += dt
    return spans


def _voiced_samples(samples: np.ndarray, words: list[dict],
                    pad_sec: float = 0.08
                    ) -> tuple[np.ndarray, float]:
    """Trim leading/trailing VAD silence around ASR word spans and compute true voiced duration."""
    n = len(samples)
    if not words:
        trimmed, vdur, _ = _detect_voiced_islands(samples)
        return trimmed, vdur

    pad = int(pad_sec * SAMPLE_RATE)
    s0 = max(0, int(words[0]["t0"] * SAMPLE_RATE) - pad)
    s1 = min(n, int(words[-1]["t1"] * SAMPLE_RATE) + pad)
    trimmed = samples[s0:s1] if s1 > s0 else samples

    # Compute true voiced duration by merging word spans with gaps <= 0.35s
    spans: list[list[float]] = []
    for w in words:
        a = max(0.0, float(w["t0"]) - 0.04)
        b = min(n / SAMPLE_RATE, float(w["t1"]) + 0.04)
        if b <= a:
            continue
        if spans and a - spans[-1][1] <= 0.35:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    voiced_dur = sum(b - a for a, b in spans) if spans else len(trimmed) / SAMPLE_RATE
    return trimmed, voiced_dur


def _compute_sub_vecs(voiced: np.ndarray,
                      embed_fn: Callable[[np.ndarray], Optional[np.ndarray]],
                      win_sec: float = 2.0, hop_sec: float = 1.0) -> list[np.ndarray]:
    """Unused placeholder kept for signature compatibility."""
    return []


def _turn_similarity(ti: dict, tj: dict) -> float:
    """Pairwise vocal cosine similarity."""
    return float(np.dot(ti["vec"], tj["vec"]))


def _best_known_speaker(vec: np.ndarray, known: list[tuple[str, np.ndarray]]) -> tuple[str, float, float]:
    """Return `(best_label, best_sim, margin_over_second_best)`."""
    if not known:
        return "", -1.0, 0.0
    scored = sorted(((float(np.dot(vec, c)), lbl) for lbl, c in known), reverse=True)
    best_s, best_lbl = scored[0]
    second_s = scored[1][0] if len(scored) > 1 else -1.0
    return best_lbl, best_s, best_s - second_s


def _same_speaker_pair(va: np.ndarray, vb: np.ndarray,
                       known_speakers: Optional[list[tuple[str, np.ndarray]]],
                       dur_a: float = 2.0, dur_b: float = 2.0) -> bool:
    """Decide whether two adjacent turn slices come from the same voice."""
    sim = float(np.dot(va, vb))
    if known_speakers and len(known_speakers) >= 2:
        la, sa, ma = _best_known_speaker(va, known_speakers)
        lb, sb, mb = _best_known_speaker(vb, known_speakers)
        if la == lb and sa >= 0.44 and sb >= 0.44 and sim >= 0.32:
            return True
        if la != lb and min(sa, sb) >= 0.36 and min(ma, mb) >= 0.09:
            return sim >= 0.51
    elif known_speakers and len(known_speakers) == 1:
        _, sa, _ = _best_known_speaker(va, known_speakers)
        _, sb, _ = _best_known_speaker(vb, known_speakers)
        hi, lo = max(sa, sb), min(sa, sb)
        if hi >= 0.54 and lo <= 0.42 and (hi - lo) >= 0.14:
            return sim >= 0.51
    thresh = 0.40 if min(dur_a, dur_b) < 1.65 else SLICE_MERGE_THRESHOLD
    return sim >= thresh


def split_segment_by_speaker(samples: np.ndarray, words: list[dict],
                             embed_fn: Callable[[np.ndarray], Optional[np.ndarray]],
                             win_sec: float = 1.5, hop_sec: float = 0.75,
                             known_speakers: Optional[list[tuple[str, np.ndarray]]] = None,
                             text: str = "",
                             segmenter=None,
                             out_meta: Optional[list[dict]] = None,
                             ) -> list[tuple[int, int, str, np.ndarray, float, list[np.ndarray]]]:
    """Return `[(sample_start, sample_end, text, embedding_vec, voiced_dur, sub_vecs), ...]`.

    Tests sentence/pause boundaries and optional pyannote neural change points first (fast) and
    runs sliding acoustic change detection only across long unbroken spans, verified against pure
    slice embeddings (with overlapping speech frames stripped when *segmenter* is active) and
    known speaker centroids. When *out_meta* is provided, appends one dict
    ``{"overlap": bool, "heavy_overlap": bool}`` per returned slice.
    """
    dur = len(samples) / SAMPLE_RATE
    synthesized = False
    if not words and text and text.strip():
        words = _synthesize_arabic_word_spans(samples, text)
        synthesized = bool(words)
    full_text = (text.strip() if text and text.strip()
                 else (" ".join(w["w"] for w in words).strip() if words else ""))

    analysis = None
    if segmenter is not None and not synthesized and dur >= 1.2:
        try:
            analysis = segmenter.analyze(samples)
        except Exception:
            analysis = None

    def _slice_overlap_meta(sa: int, sb: int) -> dict:
        if analysis is None:
            return {"overlap": False, "heavy_overlap": False}
        t_a, t_b = sa / SAMPLE_RATE, sb / SAMPLE_RATE
        has_ov, _ = analysis.has_overlap_in(t_a, t_b, min_overlap_sec=0.40)
        if not has_ov:
            return {"overlap": False, "heavy_overlap": False}
        _, clean_voiced, _ = analysis.strip_overlap(samples[sa:sb], t_a, t_b)
        return {"overlap": True, "heavy_overlap": clean_voiced < 1.0}

    def _voiced_bounds(sa: int, sb: int, sub_words: list[dict], pad_sec: float = 0.08) -> tuple[float, float]:
        if not sub_words:
            return sa / SAMPLE_RATE, sb / SAMPLE_RATE
        n_sub = sb - sa
        pad = int(pad_sec * SAMPLE_RATE)
        s0 = max(0, int(sub_words[0]["t0"] * SAMPLE_RATE) - pad)
        s1 = min(n_sub, int(sub_words[-1]["t1"] * SAMPLE_RATE) + pad)
        if s1 <= s0:
            return sa / SAMPLE_RATE, sb / SAMPLE_RATE
        return (sa + s0) / SAMPLE_RATE, (sa + s1) / SAMPLE_RATE

    def _single_turn():
        use_words = [] if synthesized else words
        voiced, vdur = _voiced_samples(samples, use_words)
        embed_in = voiced
        if analysis is not None:
            vt_a, vt_b = _voiced_bounds(0, len(samples), use_words)
            clean_audio, _, was_stripped = analysis.strip_overlap(voiced, vt_a, vt_b)
            if was_stripped:
                embed_in = clean_audio
        vec = embed_fn(embed_in)
        if vec is None and embed_in is not voiced:
            vec = embed_fn(voiced)
        if vec is None:
            vec = embed_fn(samples)
        if vec is not None:
            if out_meta is not None:
                out_meta.append(_slice_overlap_meta(0, len(samples)))
            return [(0, len(samples), full_text, vec, vdur, [])]
        return []

    min_dur = 2.2 if synthesized else 2.6
    min_words = 4 if synthesized else 6
    if dur < min_dur or not words or len(words) < min_words:
        return _single_turn()

    min_side = 0.95 if synthesized else 1.25
    min_span_scan = 2.2 if synthesized else 3.9
    win_s = 1.25 if synthesized else win_sec
    hop_s = 0.55 if synthesized else hop_sec

    t_start = max(0.0, words[0]["t0"] - 0.10)
    t_end = min(dur, words[-1]["t1"] + 0.10)
    span_dur = t_end - t_start
    max_cuts = max(4, min(8, int(span_dur // 1.6) + 1))
    ling_cuts = _find_linguistic_cuts(words, min_side_dur=min_side, max_cuts=max_cuts, min_words=min_words)

    neural_cuts: list[int] = []
    if analysis is not None and analysis.change_times:
        for t_cut in analysis.change_times:
            idx = _pick_word_cut(words, 0, len(words), t_cut, min_side_dur=0.95)
            if idx is not None:
                t_c = 0.5 * (words[idx - 1]["t1"] + words[idx]["t0"])
                if all(abs(t_c - 0.5 * (words[c - 1]["t1"] + words[c]["t0"])) >= 0.85
                       for c in ling_cuts + neural_cuts):
                    neural_cuts.append(idx)

    # Fast path for single unbroken turns with no sentence/pause/neural boundaries
    if span_dur < (2.2 if synthesized else 4.0) and not ling_cuts and not neural_cuts:
        return _single_turn()

    # Only scan sliding acoustic windows across unbroken spans without punctuation/pauses/neural cuts
    cut_set = set(ling_cuts) | set(neural_cuts)
    bounds = [0] + sorted(cut_set) + [len(words)]
    w_vecs_all: list[np.ndarray] = []
    w_centers_all: list[float] = []
    win_n = int(win_s * SAMPLE_RATE)
    hop_n = int(hop_s * SAMPLE_RATE)

    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = words[a:b]
        if len(sub) < min_words:
            continue
        sp_t0 = max(0.0, sub[0]["t0"] - 0.08)
        sp_t1 = min(dur, sub[-1]["t1"] + 0.08)
        if sp_t1 - sp_t0 < min_span_scan:
            continue
        s0, s1 = int(sp_t0 * SAMPLE_RATE), int(sp_t1 * SAMPLE_RATE)
        w_vecs: list[np.ndarray] = []
        w_centers: list[float] = []
        pos = s0
        while pos + int(0.8 * SAMPLE_RATE) <= s1:
            end = min(s1, pos + win_n)
            vec = embed_fn(samples[pos:end])
            if vec is not None:
                w_vecs.append(vec)
                w_centers.append(0.5 * (pos + end) / SAMPLE_RATE)
            if end >= s1:
                break
            pos += hop_n
        w_vecs_all.extend(w_vecs)
        w_centers_all.extend(w_centers)
        prev_c = a
        for t_cut in _find_window_changes(w_vecs, w_centers):
            idx = _pick_word_cut(words, prev_c, b, t_cut, min_side_dur=min_side)
            if idx is not None and idx > prev_c:
                t_c = 0.5 * (words[idx - 1]["t1"] + words[idx]["t0"])
                if all(abs(t_c - 0.5 * (words[c - 1]["t1"] + words[c]["t0"])) >= 0.85 for c in cut_set):
                    cut_set.add(idx)
                prev_c = idx

    if not cut_set:
        return _single_turn()

    word_cuts = [0] + sorted(cut_set) + [len(words)]

    def _embed_slice(sa: int, sb: int, a: int, b: int, trim_edges: bool = True):
        sub_words = [
            {"w": w["w"], "t0": max(0.0, w["t0"] - sa / SAMPLE_RATE), "t1": max(0.0, w["t1"] - sa / SAMPLE_RATE)}
            for w in words[a:b]
        ]
        voiced_slice, vdur = _voiced_samples(samples[sa:sb], sub_words)
        core = voiced_slice
        was_stripped = False
        if analysis is not None:
            vt_a, vt_b = _voiced_bounds(sa, sb, sub_words)
            clean_audio, _, was_stripped = analysis.strip_overlap(voiced_slice, vt_a, vt_b)
            if was_stripped:
                core = clean_audio
        # Keep brief boundary crosstalk out of voiceprints when slice is long enough
        if not was_stripped and trim_edges and len(voiced_slice) >= int(1.9 * SAMPLE_RATE):
            guard = int(0.10 * SAMPLE_RATE)
            l_off = guard if a > 0 else 0
            r_off = len(voiced_slice) - (guard if b < len(words) else 0)
            if r_off - l_off >= int(1.2 * SAMPLE_RATE):
                core = voiced_slice[l_off:r_off]
        pvec = embed_fn(core)
        if pvec is None and core is not voiced_slice:
            pvec = embed_fn(voiced_slice)
        if pvec is None:
            pvec = embed_fn(samples[sa:sb])
        if pvec is None and w_vecs_all:
            t_a, t_b = sa / SAMPLE_RATE, sb / SAMPLE_RATE
            near = [v for t, v in zip(w_centers_all, w_vecs_all) if t_a <= t <= t_b] or [w_vecs_all[0]]
            pvec = np.mean(np.stack(near), axis=0)
            pvec /= max(1e-8, float(np.linalg.norm(pvec)))
            pvec = pvec.astype(np.float32)
        return pvec, vdur

    # Build slices and merge back any parts that match the same speaker
    parts: list[tuple[int, int, int, int, np.ndarray, float, list[np.ndarray]]] = []
    for a, b in zip(word_cuts[:-1], word_cuts[1:]):
        sa = 0 if a == 0 else max(0, int((words[a]["t0"] - 0.05) * SAMPLE_RATE))
        sb = len(samples) if b == len(words) else min(len(samples), int((words[b - 1]["t1"] + 0.05) * SAMPLE_RATE))
        pvec, vdur = _embed_slice(sa, sb, a, b)
        if pvec is None:
            continue
        if parts and _same_speaker_pair(parts[-1][4], pvec, known_speakers, parts[-1][5], vdur):
            prev_sa, _, prev_wa, _, _, _, _ = parts.pop()
            merged_vec, m_vdur = _embed_slice(prev_sa, sb, prev_wa, b)
            if merged_vec is None:
                merged_vec = pvec
            parts.append((prev_sa, sb, prev_wa, b, merged_vec, m_vdur, []))
        else:
            parts.append((sa, sb, a, b, pvec, vdur, []))

    # Heal brief (< 2.0 s) isolated middle fragments inside `[A, tiny_B, A]`
    i = 1
    while i < len(parts) - 1:
        if parts[i][5] < 2.0:
            is_distinct_known = False
            if known_speakers and len(known_speakers) >= 2:
                lm, sm, mm = _best_known_speaker(parts[i][4], known_speakers)
                ll, _, _ = _best_known_speaker(parts[i - 1][4], known_speakers)
                lr, _, _ = _best_known_speaker(parts[i + 1][4], known_speakers)
                if sm >= 0.42 and mm >= 0.08 and lm != ll and lm != lr:
                    is_distinct_known = True
            if not is_distinct_known and float(np.dot(parts[i - 1][4], parts[i + 1][4])) >= 0.48:
                sa, sb, wa, wb = parts[i - 1][0], parts[i + 1][1], parts[i - 1][2], parts[i + 1][3]
                mv, md = _embed_slice(sa, sb, wa, wb)
                if mv is not None:
                    parts[i - 1:i + 2] = [(sa, sb, wa, wb, mv, md, [])]
                    continue
        i += 1

    out = []
    for sa, sb, a, b, vec, vdur, subs in parts:
        txt = " ".join(w["w"] for w in words[a:b]).strip()
        if txt:
            out.append((sa, sb, txt, vec, vdur, subs))
            if out_meta is not None:
                out_meta.append(_slice_overlap_meta(sa, sb))
    return out


# --------------------------------------------------------------------------- #
# Incremental UPGMA speaker clustering
# --------------------------------------------------------------------------- #
class SpeakerClusterer:
    """Duration-weighted agglomerative clustering over utterance embeddings.

    Confirmed utterances (>= MIN_CONFIRM_DUR) define clusters via UPGMA average linkage.
    Ultra-short backchannels (< MIN_CONFIRM_DUR) attach to the closest confirmed cluster
    or inherit the adjacent speaker so brief replies never spawn phantom speakers.
    """

    def __init__(self, merge_threshold: float = MERGE_THRESHOLD,
                 min_confirm_dur: float = MIN_CONFIRM_DUR,
                 arabic_mode: bool = False,
                 max_speakers: int = 0):
        self.merge_threshold = merge_threshold
        self.min_confirm_dur = min_confirm_dur
        self.arabic_mode = arabic_mode
        self.max_speakers = int(max_speakers) if int(max_speakers or 0) >= 2 else 0
        self.turns: list[dict] = []  # {"id", "t0", "dur", "vec", "subs", "speaker", "bridge"}
        self._centroids: list[tuple[str, np.ndarray]] = []
        self._established: list[tuple[str, np.ndarray, float, int]] = []
        self._must_link_sets: list[set[int]] = []
        self._merged_protos: list[tuple[np.ndarray, np.ndarray, set[int]]] = []
        self._dissolved_prev_bridge: dict[int, bool] = {}
        self._pinned_turns: dict[int, tuple[set[int], Optional[np.ndarray]]] = {}
        # Persistent immutable speaker identity & user-named speaker locks
        self._next_spk_num: int = 1
        self._spk_names: dict[str, str] = {}
        self._locked_anchors: dict[str, set[int]] = {}
        self._retired_cids: set[str] = set()
        self._spk_centroids: dict[str, np.ndarray] = {}
        self._turn_prev_spk: dict[int, str] = {}
        self.last_id_remap: dict[str, str] = {}
        self._edit_history: list[tuple[
            list[set[int]],
            list[tuple[np.ndarray, np.ndarray, set[int]]],
            dict[int, bool],
            dict[int, tuple[set[int], Optional[np.ndarray]]],
            dict[str, list[int]],
            dict[str, str],
            dict[str, set[int]],
            dict[str, str],
            set[str],
        ]] = []

    @staticmethod
    def _spk_num(cid: str) -> int:
        if cid and cid.startswith("Speaker "):
            tail = cid[8:].strip()
            if tail.isdigit():
                return int(tail)
        return 10**9

    def set_speaker_names(self, updates: dict[str, str],
                          current_names: Optional[dict[str, str]] = None) -> None:
        """Record custom speaker names and lock each named speaker's confirmed anchors against merge."""
        if not updates:
            return
        known_cids = {t.get("speaker", "") for t in self.turns if t.get("speaker")}
        known_cids |= set(self._spk_names.keys()) | set(self._locked_anchors.keys())
        if current_names:
            known_cids |= set(current_names.keys())
        reverse: dict[str, str] = {}
        for cid, nm in self._spk_names.items():
            if nm:
                reverse[nm] = cid
        if current_names:
            for cid, nm in current_names.items():
                if nm:
                    reverse[nm] = cid

        for key, raw_name in updates.items():
            cid = key if key in known_cids else reverse.get(key, key)
            if not cid:
                continue
            new_name = (raw_name or "").strip()
            if not new_name or new_name == cid:
                self._spk_names.pop(cid, None)
                self._locked_anchors.pop(cid, None)
                continue
            self._spk_names[cid] = new_name
            cur_confirmed = {
                t["id"] for t in self.turns
                if t.get("speaker") == cid
                and t["dur"] >= self.min_confirm_dur
                and not t.get("bridge")
                and t["id"] not in self._pinned_turns
            }
            anchors = self._locked_anchors.setdefault(cid, set())
            anchors.update(cur_confirmed)
            if cur_confirmed:
                for other_cid, other_set in list(self._locked_anchors.items()):
                    if other_cid != cid:
                        other_set.difference_update(cur_confirmed)

    def _sync_locks_to_names(self, names: dict[str, str]) -> None:
        """Keep internal `_spk_names` and `_locked_anchors` aligned with post-edit *names*."""
        self._spk_names = {
            cid: nm.strip() for cid, nm in (names or {}).items()
            if cid and nm and nm.strip() and nm.strip() != cid
        }
        for cid in list(self._locked_anchors.keys()):
            if cid not in self._spk_names:
                del self._locked_anchors[cid]
        for cid in self._spk_names:
            cur_confirmed = {
                t["id"] for t in self.turns
                if t.get("speaker") == cid
                and t["dur"] >= self.min_confirm_dur
                and not t.get("bridge")
                and t["id"] not in self._pinned_turns
            }
            if cur_confirmed:
                self._locked_anchors.setdefault(cid, set()).update(cur_confirmed)

    def known_centroids(self) -> list[tuple[str, np.ndarray]]:
        return list(self._centroids)

    def has_speaker(self, label: str) -> bool:
        return any(t.get("speaker") == label for t in self.turns)

    def has_any_turn(self, turn_ids: list[int]) -> bool:
        if not turn_ids:
            return False
        tset = set(turn_ids)
        return any(t["id"] in tset for t in self.turns)

    def speaker_count(self) -> int:
        return len({t.get("speaker") for t in self.turns if t.get("speaker")})

    def manual_edit_count(self) -> int:
        if self._edit_history:
            return len(self._edit_history)
        return 1 if (
            self._must_link_sets or self._merged_protos
            or self._dissolved_prev_bridge or self._pinned_turns
        ) else 0

    def speaker_anchor_ids(self, label: str) -> list[int]:
        return [t["id"] for t in self.turns if t.get("speaker") == label]

    def resolve_speaker_from_anchors(self, fallback_label: str, anchors: list[int]) -> str:
        if anchors:
            aset = set(anchors)
            labels = [t.get("speaker", "") for t in self.turns if t["id"] in aset and t.get("speaker")]
            if labels:
                return max(set(labels), key=labels.count)
        return fallback_label

    @staticmethod
    def _compute_id_remap(old_by_spk: dict[str, list[int]],
                          new_mapping: dict[int, str],
                          merged: Optional[tuple[str, str]] = None) -> dict[str, str]:
        id_remap: dict[str, str] = {}
        merged_src = merged[0] if merged is not None else None
        for old_cid, tids in old_by_spk.items():
            counts: dict[str, int] = {}
            for tid in tids:
                ncid = new_mapping.get(tid)
                if ncid:
                    counts[ncid] = counts.get(ncid, 0) + 1
            if counts:
                # Persistent IDs: if old_cid still exists (and wasn't explicitly merged away), it was NOT renumbered
                if old_cid != merged_src and counts.get(old_cid, 0) > 0:
                    continue
                best = max(counts.items(), key=lambda kv: (kv[1], kv[0] == old_cid))[0]
                if best != old_cid:
                    id_remap[old_cid] = best
        if merged is not None:
            src, dst = merged
            dst_new = id_remap.get(dst, dst)
            if src != dst_new:
                id_remap[src] = dst_new
        return id_remap

    def _push_edit_snapshot(self, old_by_spk: dict[str, list[int]], old_names: dict[str, str]) -> None:
        snap = (
            [set(s) for s in self._must_link_sets],
            [(sv, dv, set(ids)) for sv, dv, ids in self._merged_protos],
            dict(self._dissolved_prev_bridge),
            {tid: (set(dids), dvec) for tid, (dids, dvec) in self._pinned_turns.items()},
            {k: list(v) for k, v in old_by_spk.items()},
            dict(old_names),
            {k: set(v) for k, v in self._locked_anchors.items()},
            dict(self._spk_names),
            set(self._retired_cids),
        )
        self._edit_history.append(snap)
        if len(self._edit_history) > 25:
            del self._edit_history[:-25]

    def _speaker_turn_ids(self) -> dict[str, list[int]]:
        by_spk: dict[str, list[int]] = {}
        for t in self.turns:
            sp = t.get("speaker")
            if sp:
                by_spk.setdefault(sp, []).append(t["id"])
        return by_spk

    @staticmethod
    def _remap_names(old_by_spk: dict[str, list[int]],
                     new_mapping: dict[int, str],
                     old_names: dict[str, str],
                     merged: Optional[tuple[str, str]] = None,
                     dissolved: Optional[str] = None) -> dict[str, str]:
        """Keep custom display names aligned with their real clusters after renumbering."""
        if not old_names:
            return {}
        new_names: dict[str, str] = {}
        items = list(old_names.items())
        if merged is not None:
            src, dst = merged
            dst_name = old_names.get(dst, "") or old_names.get(src, "")
            items = [(cid, nm) for cid, nm in items if cid not in (src, dst)]
            if dst_name:
                items.insert(0, (dst, dst_name))
        if dissolved is not None:
            items = [(cid, nm) for cid, nm in items if cid != dissolved]

        tracked = []
        untracked = []
        for old_cid, custom_name in items:
            if not custom_name:
                continue
            tids = old_by_spk.get(old_cid, [])
            if tids:
                tracked.append((old_cid, custom_name, tids))
            else:
                untracked.append((old_cid, custom_name))

        for old_cid, custom_name, tids in tracked:
            counts: dict[str, int] = {}
            for tid in tids:
                ncid = new_mapping.get(tid)
                if ncid:
                    counts[ncid] = counts.get(ncid, 0) + 1
            if counts:
                # Persistent IDs: keep custom name on old_cid whenever old_cid still exists
                if counts.get(old_cid, 0) > 0:
                    best_ncid = old_cid
                else:
                    best_ncid = max(counts.items(), key=lambda kv: (kv[1], kv[0] == old_cid))[0]
                if best_ncid not in new_names and custom_name != best_ncid:
                    new_names[best_ncid] = custom_name

        for old_cid, custom_name in untracked:
            if old_cid not in new_names and custom_name != old_cid:
                new_names[old_cid] = custom_name
        return new_names

    def _vec_centroid_for_ids(self, turn_ids: set[int]) -> Optional[np.ndarray]:
        wv = [t["vec"] * min(4.0, max(0.5, t["dur"])) for t in self.turns if t["id"] in turn_ids]
        if not wv:
            return None
        c = np.sum(np.stack(wv), axis=0)
        nrm = float(np.linalg.norm(c))
        if nrm < 1e-8:
            return None
        return (c / nrm).astype(np.float32)

    def merge_speakers(self, src_label: str, dst_label: str,
                       old_names: Optional[dict[str, str]] = None
                       ) -> tuple[dict[int, str], dict[str, str]]:
        """Permanently merge *src_label* into *dst_label* across past and future turns."""
        if not src_label or not dst_label or src_label == dst_label:
            return {}, dict(old_names or {})
        old_by_spk = self._speaker_turn_ids()
        src_ids = set(old_by_spk.get(src_label, ()))
        dst_ids = set(old_by_spk.get(dst_label, ()))
        if not src_ids or not dst_ids:
            return {}, dict(old_names or {})
        self._push_edit_snapshot(old_by_spk, old_names or {})

        src_vec = self._vec_centroid_for_ids(src_ids)
        dst_vec = self._vec_centroid_for_ids(dst_ids)
        if src_vec is not None and dst_vec is not None:
            self._merged_protos.append((src_vec, dst_vec, set(dst_ids)))

        combined = src_ids | dst_ids
        kept_sets: list[set[int]] = []
        for existing in self._must_link_sets:
            if not existing.isdisjoint(combined):
                combined |= existing
            else:
                kept_sets.append(existing)
        kept_sets.append(combined)
        self._must_link_sets = kept_sets
        for tid in src_ids:
            self._pinned_turns.pop(tid, None)
        # Fuse any locked anchors/names from src into dst so hard cannot-link never fights an explicit merge
        src_locks = self._locked_anchors.pop(src_label, set())
        src_name = self._spk_names.pop(src_label, "")
        self._retired_cids.add(src_label)
        if src_name and dst_label not in self._spk_names:
            self._spk_names[dst_label] = src_name
        if src_locks or dst_label in self._locked_anchors or dst_label in self._spk_names:
            dst_locks = self._locked_anchors.setdefault(dst_label, set())
            dst_locks.update(src_locks)

        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(old_by_spk, mapping, merged=(src_label, dst_label))
        remapped = self._remap_names(old_by_spk, mapping, old_names or {}, merged=(src_label, dst_label))
        self._sync_locks_to_names(remapped)
        return mapping, remapped

    def pin_turns_to_speaker(self, turn_ids: list[int], dst_label: str,
                             old_names: Optional[dict[str, str]] = None
                             ) -> tuple[dict[int, str], dict[str, str]]:
        """Reassign only the specified *turn_ids* to *dst_label* without merging whole speakers or polluting centroids."""
        if not turn_ids or not dst_label:
            return {}, dict(old_names or {})
        old_by_spk = self._speaker_turn_ids()
        dst_ids = set(old_by_spk.get(dst_label, ()))
        known_ids = {t["id"] for t in self.turns}
        target_tids = [tid for tid in turn_ids if tid in known_ids]
        if not target_tids or not dst_ids:
            return {}, dict(old_names or {})
        self._push_edit_snapshot(old_by_spk, old_names or {})
        target_set = set(target_tids)
        dst_anchors = dst_ids - target_set
        if not dst_anchors:
            dst_anchors = set(dst_ids)
        dst_vec = self._vec_centroid_for_ids(dst_anchors)
        for tid in target_tids:
            self._pinned_turns[tid] = (set(dst_anchors), dst_vec)
        for lock_set in self._locked_anchors.values():
            lock_set.difference_update(target_set)
        self._must_link_sets = [s - target_set for s in self._must_link_sets if len(s - target_set) >= 2]
        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(old_by_spk, mapping)
        remapped = self._remap_names(old_by_spk, mapping, old_names or {})
        self._sync_locks_to_names(remapped)
        return mapping, remapped

    def dissolve_speaker(self, src_label: str,
                         old_names: Optional[dict[str, str]] = None
                         ) -> tuple[dict[int, str], dict[str, str]]:
        """Mark *src_label* as overlap/clash turns and reassign each turn to its closest remaining speaker."""
        if not src_label or self.speaker_count() < 2:
            return {}, dict(old_names or {})
        old_by_spk = self._speaker_turn_ids()
        src_ids = set(old_by_spk.get(src_label, ()))
        if not src_ids:
            return {}, dict(old_names or {})
        self._push_edit_snapshot(old_by_spk, old_names or {})
        for t in self.turns:
            if t["id"] in src_ids:
                if t["id"] not in self._dissolved_prev_bridge:
                    self._dissolved_prev_bridge[t["id"]] = bool(t.get("bridge", False))
                t["bridge"] = True
        for tid in src_ids:
            self._pinned_turns.pop(tid, None)
        self._locked_anchors.pop(src_label, None)
        self._spk_names.pop(src_label, None)
        self._retired_cids.add(src_label)
        self._must_link_sets = [s - src_ids for s in self._must_link_sets if len(s - src_ids) >= 2]
        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(old_by_spk, mapping)
        remapped = self._remap_names(old_by_spk, mapping, old_names or {}, dissolved=src_label)
        self._sync_locks_to_names(remapped)
        return mapping, remapped

    def undo_last_speaker_edit(self, old_names: Optional[dict[str, str]] = None
                               ) -> tuple[dict[int, str], dict[str, str]]:
        """Revert the most recent manual line change, merge, or overlap reassignment."""
        if not self._edit_history:
            return self.reset_speaker_edits(old_names=old_names)
        cur_by_spk = self._speaker_turn_ids()
        (prev_links, prev_protos, prev_dissolved, prev_pinned,
         prev_by_spk, prev_names, prev_locks, prev_spk_names, prev_retired) = self._edit_history.pop()
        for t in self.turns:
            tid = t["id"]
            if tid in self._dissolved_prev_bridge and tid not in prev_dissolved:
                t["bridge"] = self._dissolved_prev_bridge[tid]
        self._must_link_sets = prev_links
        self._merged_protos = prev_protos
        self._dissolved_prev_bridge = prev_dissolved
        self._pinned_turns = prev_pinned
        self._locked_anchors = prev_locks
        self._spk_names = prev_spk_names
        self._retired_cids = prev_retired
        for cid, tids in prev_by_spk.items():
            for tid in tids:
                self._turn_prev_spk[tid] = cid
        if not self.turns:
            self.last_id_remap = {}
            return {}, dict(prev_names)
        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(cur_by_spk, mapping)
        remapped = self._remap_names(prev_by_spk, mapping, prev_names)
        cur_remapped = self._remap_names(cur_by_spk, mapping, old_names or {})
        for k, v in cur_remapped.items():
            if k not in remapped:
                remapped[k] = v
        self._sync_locks_to_names(remapped)
        return mapping, remapped

    def reset_speaker_edits(self, old_names: Optional[dict[str, str]] = None
                            ) -> tuple[dict[int, str], dict[str, str]]:
        """Clear all manual line changes, merges, and overlap reassignments back to automatic clustering."""
        if not (
            self._must_link_sets or self._merged_protos
            or self._dissolved_prev_bridge or self._pinned_turns or self._edit_history
        ):
            return {}, dict(old_names or {})
        cur_by_spk = self._speaker_turn_ids()
        if self._edit_history:
            first_by_spk = self._edit_history[0][4]
            first_names = self._edit_history[0][5]
            first_locks = {k: set(v) for k, v in self._edit_history[0][6].items()}
            first_spk_names = dict(self._edit_history[0][7])
            first_retired = set(self._edit_history[0][8])
        else:
            first_by_spk = cur_by_spk
            first_names = dict(old_names or {})
            first_locks = {k: set(v) for k, v in self._locked_anchors.items()}
            first_spk_names = dict(self._spk_names)
            first_retired = set()
        for t in self.turns:
            tid = t["id"]
            if tid in self._dissolved_prev_bridge:
                t["bridge"] = self._dissolved_prev_bridge[tid]
        self._must_link_sets.clear()
        self._merged_protos.clear()
        self._dissolved_prev_bridge.clear()
        self._pinned_turns.clear()
        self._edit_history.clear()
        self._locked_anchors = first_locks
        self._spk_names = first_spk_names
        self._retired_cids = first_retired
        for cid, tids in first_by_spk.items():
            for tid in tids:
                self._turn_prev_spk[tid] = cid
        if not self.turns:
            self.last_id_remap = {}
            return {}, dict(first_names)
        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(cur_by_spk, mapping)
        remapped = self._remap_names(first_by_spk, mapping, first_names)
        cur_remapped = self._remap_names(cur_by_spk, mapping, old_names or {})
        for k, v in cur_remapped.items():
            if k not in remapped:
                remapped[k] = v
        self._sync_locks_to_names(remapped)
        return mapping, remapped

    def set_max_speakers(self, max_speakers: int,
                         old_names: Optional[dict[str, str]] = None
                         ) -> tuple[dict[int, str], dict[str, str]]:
        """Update max speaker cap (0 = Auto, >= 2 caps to top K) and immediately recluster."""
        cap = int(max_speakers) if int(max_speakers or 0) >= 2 else 0
        if cap == self.max_speakers:
            return {}, dict(old_names or {})
        old_by_spk = self._speaker_turn_ids()
        self.max_speakers = cap
        if not self.turns:
            self.last_id_remap = {}
            return {}, dict(old_names or {})
        mapping = self._recluster()
        self.last_id_remap = self._compute_id_remap(old_by_spk, mapping)
        remapped = self._remap_names(old_by_spk, mapping, old_names or {})
        return mapping, remapped

    def _is_overlap_bridge(self, vec: np.ndarray, dur: float) -> bool:
        """Return True when an incoming turn strongly matches two distinct established speakers at once."""
        if dur < self.min_confirm_dur or len(self._established) < 2:
            return False
        scored = sorted(
            ((float(np.dot(vec, c)), c) for _, c, _, _ in self._established),
            key=lambda x: x[0],
            reverse=True,
        )
        s1, c1 = scored[0]
        s2, c2 = scored[1]
        if s1 < self.merge_threshold or float(np.dot(c1, c2)) >= ESTABLISHED_MERGE:
            return False
        margin = s1 - s2
        return (s2 >= 0.44 and margin < 0.10) or (s2 >= 0.48 and margin < 0.14)

    def add_and_cluster(self, turn_id: int, start_sec: float, dur: float,
                        vec: np.ndarray, sub_vecs: Optional[list[np.ndarray]] = None,
                        bridge_hint: bool = False,
                        ) -> tuple[str, dict[int, str]]:
        """Add an utterance embedding and return `(assigned_speaker, full_id_to_speaker_map)`."""
        v = np.asarray(vec, dtype=np.float32)
        self.turns.append({
            "id": turn_id, "t0": start_sec, "dur": dur,
            "vec": v,
            "subs": list(sub_vecs) if sub_vecs else [],
            "speaker": "",
            "bridge": bool(bridge_hint) or self._is_overlap_bridge(v, dur),
        })
        mapping = self._recluster()
        return mapping[turn_id], mapping

    @staticmethod
    def _pick_stratified(idxs: list[int], turns: list[dict], quota: int) -> list[int]:
        """Select up to *quota* diverse indices: earliest 2 + longest + evenly spaced across time."""
        if quota <= 0 or not idxs:
            return []
        if len(idxs) <= quota:
            return list(idxs)
        picked: set[int] = set()
        for m in idxs[:min(2, quota)]:
            picked.add(m)
        n_long = max(1, min(6, quota // 3))
        for m in sorted(idxs, key=lambda idx: turns[idx]["dur"], reverse=True):
            if len(picked) >= min(quota, 2 + n_long):
                break
            picked.add(m)
        rem = [m for m in idxs if m not in picked]
        need = quota - len(picked)
        if need > 0 and rem:
            if len(rem) <= need:
                picked.update(rem)
            else:
                step = (len(rem) - 1) / max(1, need - 1) if need > 1 else 0.0
                for k in range(need):
                    picked.add(rem[min(len(rem) - 1, int(round(k * step)))])
        return sorted(picked)

    def _select_reservoir(self, confirmed: list[int]) -> tuple[list[int], list[int]]:
        """Return `(confirmed_sub, older)` preserving every active/locked speaker within `RESERVOIR_LIMIT`."""
        if len(confirmed) <= RESERVOIR_LIMIT:
            return confirmed, []

        picked: set[int] = set()
        # 1. Locked anchor turns for user-named speakers (never evicted even on multi-hour monologues)
        locked_cids = [cid for cid, tids in sorted(self._locked_anchors.items()) if tids]
        if locked_cids:
            lock_budget = max(len(locked_cids), RESERVOIR_LIMIT - RESERVOIR_RECENT)
            per_locked = min(RESERVOIR_LOCKED_PER_SPK, max(4, lock_budget // len(locked_cids)))
            for cid in locked_cids:
                tids = self._locked_anchors[cid]
                c_idxs = [m for m in confirmed if self.turns[m]["id"] in tids]
                for m in self._pick_stratified(c_idxs, self.turns, per_locked):
                    if len(picked) < lock_budget:
                        picked.add(m)

        # 2. Keep representative turns from manual merge sets so merges persist past 160 turns
        if self._must_link_sets and len(picked) < RESERVOIR_LIMIT - RESERVOIR_RECENT:
            for link_set in self._must_link_sets:
                c_idxs = [m for m in confirmed if self.turns[m]["id"] in link_set]
                if c_idxs:
                    for m in self._pick_stratified(c_idxs, self.turns, 4):
                        if len(picked) < RESERVOIR_LIMIT - RESERVOIR_RECENT:
                            picked.add(m)

        # 3. Always include the latest confirmed turns so new voices cluster immediately
        for m in confirmed[-RESERVOIR_RECENT:]:
            if len(picked) >= RESERVOIR_LIMIT:
                break
            picked.add(m)

        # 4. Stratified exemplars per previously assigned speaker (prevents Speaker-1 eviction)
        if len(picked) < RESERVOIR_LIMIT:
            by_spk: dict[str, list[int]] = {}
            for m in confirmed:
                sp = self._turn_prev_spk.get(self.turns[m]["id"], "")
                by_spk.setdefault(sp, []).append(m)
            spks = sorted(
                by_spk.keys(),
                key=lambda s: (sum(1 for m in by_spk[s] if m in picked), self._spk_num(s)),
            )
            for idx_s, sp in enumerate(spks):
                rem_budget = RESERVOIR_LIMIT - len(picked)
                if rem_budget <= 0:
                    break
                rem_spks = len(spks) - idx_s
                grp = by_spk[sp]
                unpicked = [m for m in grp if m not in picked]
                if not unpicked:
                    continue
                already = len(grp) - len(unpicked)
                target_total = min(RESERVOIR_MAX_PER_SPK, already + max(4, rem_budget // max(1, rem_spks)))
                need = min(rem_budget, max(0, target_total - already))
                if need > 0:
                    picked.update(self._pick_stratified(unpicked, self.turns, need))

            # Fill any remaining capacity evenly across unpicked confirmed turns
            if len(picked) < RESERVOIR_LIMIT:
                unpicked_all = [m for m in confirmed if m not in picked]
                need = RESERVOIR_LIMIT - len(picked)
                if unpicked_all and need > 0:
                    picked.update(self._pick_stratified(unpicked_all, self.turns, need))

        confirmed_sub = [m for m in confirmed if m in picked]
        older = [m for m in confirmed if m not in picked]
        return confirmed_sub, older

    def _recluster(self) -> dict[int, str]:
        confirmed = [
            i for i, t in enumerate(self.turns)
            if t["dur"] >= self.min_confirm_dur and not t.get("bridge")
            and t["id"] not in self._pinned_turns
        ]
        if not confirmed:
            # Nothing long enough to confirm yet: treat initial short speech as Speaker 1
            for t in self.turns:
                t["speaker"] = "Speaker 1"
            self._centroids = []
            self._established = []
            return {t["id"]: t["speaker"] for t in self.turns}

        # Stratified exemplar reservoir capped at RESERVOIR_LIMIT (identical to full list when <= 160)
        confirmed_sub, older = self._select_reservoir(confirmed)

        # Build locked-speaker ownership map for hard cannot-link protection
        locked_owner: dict[int, str] = {}
        for cid, tids in self._locked_anchors.items():
            for tid in tids:
                locked_owner[tid] = cid

        def _locked_owners_of(member_indices: list[int]) -> set[str]:
            if not locked_owner:
                return set()
            owners: set[str] = set()
            for m in member_indices:
                own = locked_owner.get(self.turns[m]["id"])
                if own:
                    owners.add(own)
            return owners

        durs = np.array([self.turns[i]["dur"] for i in confirmed_sub], dtype=np.float32)
        weights = np.minimum(4.0, np.maximum(0.75, durs))
        cent_weights = np.minimum(4.0, durs)
        n = len(confirmed_sub)
        vecs = np.stack([self.turns[i]["vec"] for i in confirmed_sub]).astype(np.float32, copy=False)
        sim_mat = vecs @ vecs.T
        np.fill_diagonal(sim_mat, 1.0)

        # Clusters stored as lists of local indices into `confirmed_sub`
        clusters: list[list[int]] = [[i] for i in range(n)]
        cluster_dur = durs.copy()
        cluster_wsum = vecs * cent_weights[:, None]
        cluster_locked: list[set[str]] = [
            ({locked_owner[self.turns[m]["id"]]} if self.turns[m]["id"] in locked_owner else set())
            for m in confirmed_sub
        ]
        # Maintain pairwise weighted average similarity between active clusters (symmetric vectorized UPGMA)
        pair_wt = np.outer(weights, weights)
        pair_sum = sim_mat * pair_wt
        np.fill_diagonal(pair_wt, 0.0)
        np.fill_diagonal(pair_sum, 0.0)

        avg_mat = sim_mat.copy()
        np.fill_diagonal(avg_mat, -2.0)

        def _centroid(members_idx):
            wv = [self.turns[m]["vec"] * min(4.0, self.turns[m]["dur"]) for m in members_idx]
            c = np.sum(np.stack(wv), axis=0)
            c /= max(1e-8, float(np.dot(c, c)) ** 0.5)
            return c.astype(np.float32)

        def _can_merge_fast(ca_idx: int, cb_idx: int) -> bool:
            # Hard cannot-link: never merge two clusters owning anchors of different locked speakers
            if locked_owner:
                oa, ob = cluster_locked[ca_idx], cluster_locked[cb_idx]
                if oa and ob and oa != ob:
                    return False
            da = float(cluster_dur[ca_idx])
            db = float(cluster_dur[cb_idx])
            if da < 4.0 or db < 4.0:
                return True
            va = cluster_wsum[ca_idx]
            vb = cluster_wsum[cb_idx]
            denom = (float(np.dot(va, va)) * float(np.dot(vb, vb))) ** 0.5
            csim = float(np.dot(va, vb) / max(1e-8, denom))
            if da >= 6.0 and db >= 6.0:
                idxs_a, idxs_b = clusters[ca_idx], clusters[cb_idx]
                multi = len(idxs_a) >= 2 and len(idxs_b) >= 2
                need_c = ESTABLISHED_MULTI_MERGE if multi else ESTABLISHED_MERGE
                if csim < need_c:
                    return False
                need_m = ESTABLISHED_MULTI_MEDIAN if multi else ESTABLISHED_MEDIAN_MIN
                if len(idxs_a) == 1:
                    sub = sim_mat[idxs_a[0], idxs_b]
                elif len(idxs_b) == 1:
                    sub = sim_mat[idxs_a, idxs_b[0]]
                else:
                    sub = sim_mat[np.ix_(idxs_a, idxs_b)]
                return float(np.median(sub)) >= need_m
            return csim >= CENTROID_MERGE

        def _can_merge(idxs_a: list[int], idxs_b: list[int], require_established_only: bool = False) -> bool:
            ma = [confirmed_sub[idx] for idx in idxs_a]
            mb = [confirmed_sub[idx] for idx in idxs_b]
            # Hard cannot-link: checked BEFORE the < 4.0s early return
            if locked_owner:
                oa = _locked_owners_of(ma)
                ob = _locked_owners_of(mb)
                if oa and ob and oa != ob:
                    return False
            da = sum(self.turns[m]["dur"] for m in ma)
            db = sum(self.turns[m]["dur"] for m in mb)
            if require_established_only and (da < 4.0 or db < 4.0):
                return True
            csim = float(np.dot(_centroid(ma), _centroid(mb)))
            if da >= 6.0 and db >= 6.0:
                multi = len(idxs_a) >= 2 and len(idxs_b) >= 2
                need_c = ESTABLISHED_MULTI_MERGE if multi else ESTABLISHED_MERGE
                if csim < need_c:
                    return False
                need_m = ESTABLISHED_MULTI_MEDIAN if multi else ESTABLISHED_MEDIAN_MIN
                med = float(np.median(sim_mat[np.ix_(idxs_a, idxs_b)]))
                return med >= need_m
            return csim >= CENTROID_MERGE

        active_mask = np.ones(n, dtype=bool)
        blocked_mask = np.zeros((n, n), dtype=bool)
        n_active = n
        while n_active > 1:
            flat = int(avg_mat.argmax())
            best_a, best_b = divmod(flat, n)
            best_sim = float(avg_mat[best_a, best_b])
            if best_sim < self.merge_threshold:
                break
            if best_a > best_b:
                best_a, best_b = best_b, best_a
            if not _can_merge_fast(best_a, best_b):
                blocked_mask[best_a, best_b] = blocked_mask[best_b, best_a] = True
                avg_mat[best_a, best_b] = avg_mat[best_b, best_a] = -2.0
                continue
            clusters[best_a].extend(clusters[best_b])
            clusters[best_b] = []
            cluster_dur[best_a] += cluster_dur[best_b]
            cluster_wsum[best_a] += cluster_wsum[best_b]
            if cluster_locked[best_b]:
                cluster_locked[best_a].update(cluster_locked[best_b])
                cluster_locked[best_b].clear()
            active_mask[best_b] = False
            n_active -= 1
            avg_mat[best_b, :] = -2.0
            avg_mat[:, best_b] = -2.0
            pair_sum[best_a, :] += pair_sum[best_b, :]
            pair_sum[:, best_a] = pair_sum[best_a, :]
            pair_wt[best_a, :] += pair_wt[best_b, :]
            pair_wt[:, best_a] = pair_wt[best_a, :]
            valid = active_mask & (~blocked_mask[best_a])
            valid[best_a] = False
            new_avg = pair_sum[best_a, valid] / np.maximum(1e-8, pair_wt[best_a, valid])
            avg_mat[best_a, valid] = new_avg
            avg_mat[valid, best_a] = new_avg
        active = [i for i in range(n) if active_mask[i]]

        # Consolidate clusters whose weighted speaker centroids match closely, while shielding
        # established independent speakers (and user-named locked speakers) against collapse.
        local_groups = [clusters[cid] for cid in active]
        group_cents = [_centroid([confirmed_sub[idx] for idx in g]) for g in local_groups]
        merged_c = True
        while merged_c and len(local_groups) > 1:
            merged_c = False
            cmat = np.stack(group_cents).astype(np.float64, copy=False)
            csims = cmat @ cmat.T
            np.fill_diagonal(csims, -2.0)
            cand_pairs = np.argwhere(np.triu(csims, k=1) >= CENTROID_MERGE)
            for ia, ib in cand_pairs:
                ia_i, ib_i = int(ia), int(ib)
                if _can_merge(local_groups[ia_i], local_groups[ib_i]):
                    local_groups[ia_i].extend(local_groups[ib_i])
                    del local_groups[ib_i]
                    group_cents[ia_i] = _centroid([confirmed_sub[idx] for idx in local_groups[ia_i]])
                    del group_cents[ib_i]
                    merged_c = True
                    break
        groups = [[confirmed_sub[idx] for idx in g] for g in local_groups]

        # Enforce user-confirmed speaker merges (keeps merged clusters united on future turns)
        if self._must_link_sets and len(groups) > 1:
            changed = True
            while changed and len(groups) > 1:
                changed = False
                group_id_sets = [{self.turns[m]["id"] for m in g} for g in groups]
                for link_set in self._must_link_sets:
                    hits = [gi for gi, gids in enumerate(group_id_sets) if not gids.isdisjoint(link_set)]
                    if len(hits) >= 2:
                        base = hits[0]
                        for gi in reversed(hits[1:]):
                            groups[base].extend(groups[gi])
                            del groups[gi]
                        changed = True
                        break
        if self._merged_protos and len(groups) > 1:
            changed = True
            while changed and len(groups) > 1:
                changed = False
                group_id_sets = [{self.turns[m]["id"] for m in g} for g in groups]
                group_cents = [_centroid(g) for g in groups]
                for src_vec, dst_vec, dst_ids in self._merged_protos:
                    dst_gi = next(
                        (gi for gi, gids in enumerate(group_id_sets) if not gids.isdisjoint(dst_ids)),
                        None,
                    )
                    if dst_gi is None:
                        sims_d = [float(np.dot(gv, dst_vec)) for gv in group_cents]
                        best_d = int(np.argmax(sims_d))
                        if sims_d[best_d] >= self.merge_threshold:
                            dst_gi = best_d
                    if dst_gi is None:
                        continue
                    dst_own = _locked_owners_of(groups[dst_gi])
                    for gi, gv in enumerate(group_cents):
                        if gi == dst_gi:
                            continue
                        g_own = _locked_owners_of(groups[gi])
                        if dst_own and g_own and dst_own != g_own:
                            continue
                        if float(np.dot(gv, src_vec)) >= self.merge_threshold:
                            groups[dst_gi].extend(groups[gi])
                            del groups[gi]
                            changed = True
                            break
                    if changed:
                        break

        if self.arabic_mode:
            multi_major = [
                g for g in groups
                if len(g) >= 2 and sum(self.turns[m]["dur"] for m in g) >= 2.8
            ]
            if len(multi_major) >= 2:
                major = multi_major
            else:
                major = [g for g in groups if sum(self.turns[m]["dur"] for m in g) >= 2.8]
        else:
            major = [g for g in groups if sum(self.turns[m]["dur"] for m in g) >= 2.8]

        # Never fold away a cluster that owns anchors of a user-named locked speaker
        if locked_owner:
            major_set = {id(g) for g in major}
            for g in groups:
                if id(g) not in major_set and _locked_owners_of(g):
                    major.append(g)
                    major_set.add(id(g))

        attached: list[list[int]] = [[] for _ in groups]
        if major:
            major_ids = {id(g) for g in major}
            kept = [list(g) for g in major]
            kept_vecs = [_centroid(g) for g in kept]
            kept_owners = [_locked_owners_of(g) for g in kept]
            attached = [[] for _ in kept]
            for g in groups:
                if id(g) in major_ids:
                    continue
                gdur = sum(self.turns[m]["dur"] for m in g)
                gv = _centroid(g)
                g_own = _locked_owners_of(g)
                sims = [
                    (-2.0 if (g_own and ko and g_own != ko) else float(np.dot(gv, kv)))
                    for kv, ko in zip(kept_vecs, kept_owners)
                ]
                bi = int(np.argmax(sims))
                # Fold brief or isolated single-turn fragments into the closest established speaker
                if self.arabic_mode and len(major) >= 2 and len(g) == 1:
                    need = 0.20
                    if sims[bi] >= need:
                        # Attach label without polluting the real speaker's centroid
                        attached[bi].extend(g)
                    else:
                        kept.append(list(g))
                        kept_vecs.append(gv)
                        kept_owners.append(g_own)
                        attached.append([])
                else:
                    need = 0.35 if gdur < 1.8 else 0.42
                    if sims[bi] >= need:
                        kept[bi].extend(g)
                        kept_vecs[bi] = _centroid(kept[bi])
                        if g_own:
                            kept_owners[bi] |= g_own
                    else:
                        kept.append(list(g))
                        kept_vecs.append(gv)
                        kept_owners.append(g_own)
                        attached.append([])
            groups = kept

        # Optional user-configured speaker cap: prioritize locked/renamed speakers first, then top K strongest
        if self.max_speakers >= 2 and len(groups) > self.max_speakers:
            ranked = sorted(
                range(len(groups)),
                key=lambda gi: (
                    1 if _locked_owners_of(groups[gi]) else 0,
                    sum(self.turns[m]["dur"] for m in groups[gi]),
                    len(groups[gi]),
                    -min(self.turns[m]["t0"] for m in groups[gi]),
                ),
                reverse=True,
            )
            keep_set = set(ranked[:self.max_speakers])
            kept_g = [groups[gi] for gi in range(len(groups)) if gi in keep_set]
            kept_a = [attached[gi] for gi in range(len(groups)) if gi in keep_set]
            kept_c = [_centroid(g) for g in kept_g]
            for gi in range(len(groups)):
                if gi in keep_set:
                    continue
                for m in groups[gi] + attached[gi]:
                    v = self.turns[m]["vec"]
                    bi = int(np.argmax([float(np.dot(v, kv)) for kv in kept_c]))
                    kept_a[bi].append(m)
            groups, attached = kept_g, kept_a

        # Build cluster centroids and match 1-to-1 to persistent immutable speaker IDs
        comp = []
        for members, extra in zip(groups, attached):
            first_t0 = min(self.turns[m]["t0"] for m in members)
            comp.append((first_t0, members, extra, _centroid(members)))
        comp.sort(key=lambda x: x[0])

        n_comp = len(comp)
        assigned_cid: list[Optional[str]] = [None] * n_comp
        used_cids: set[str] = set()
        member_tids = [{self.turns[m]["id"] for m in members} for _, members, _, _ in comp]
        all_tids = [
            member_tids[ci] | {self.turns[m]["id"] for m in extra}
            for ci, (_, _, extra, _) in enumerate(comp)
        ]

        # Pass 1: Match locked (user-named) speakers by locked anchor overlap
        if self._locked_anchors:
            p1_cands = []
            for cid, anchor_tids in self._locked_anchors.items():
                if not anchor_tids:
                    continue
                prev_cent = self._spk_centroids.get(cid)
                for ci in range(n_comp):
                    lh_mem = len(member_tids[ci] & anchor_tids)
                    lh_all = len(all_tids[ci] & anchor_tids)
                    if lh_mem == 0 and lh_all == 0:
                        continue
                    prev_hits = sum(1 for tid in all_tids[ci] if self._turn_prev_spk.get(tid) == cid)
                    csim = float(np.dot(comp[ci][3], prev_cent)) if prev_cent is not None else 0.0
                    p1_cands.append(((lh_mem, lh_all, prev_hits, csim, -self._spk_num(cid), -ci), cid, ci))
            p1_cands.sort(key=lambda x: x[0], reverse=True)
            for _, cid, ci in p1_cands:
                if assigned_cid[ci] is None and cid not in used_cids:
                    assigned_cid[ci] = cid
                    used_cids.add(cid)

        # Pass 2: Match remaining clusters to their previous canonical IDs by turn-duration overlap
        p2_cands = []
        for ci, (_, members, extra, centroid) in enumerate(comp):
            if assigned_cid[ci] is not None:
                continue
            dur_by_cid: dict[str, float] = {}
            cnt_by_cid: dict[str, int] = {}
            for m in members + extra:
                prev_cid = self._turn_prev_spk.get(self.turns[m]["id"])
                if prev_cid and prev_cid not in used_cids:
                    dur_by_cid[prev_cid] = dur_by_cid.get(prev_cid, 0.0) + self.turns[m]["dur"]
                    cnt_by_cid[prev_cid] = cnt_by_cid.get(prev_cid, 0) + 1
            for cid, cnt in cnt_by_cid.items():
                prev_cent = self._spk_centroids.get(cid)
                csim = float(np.dot(centroid, prev_cent)) if prev_cent is not None else 0.0
                named_pri = 1 if cid in self._spk_names else 0
                p2_cands.append(((named_pri, dur_by_cid[cid], cnt, csim, -self._spk_num(cid), -ci), cid, ci))
        p2_cands.sort(key=lambda x: x[0], reverse=True)
        for _, cid, ci in p2_cands:
            if assigned_cid[ci] is None and cid not in used_cids:
                assigned_cid[ci] = cid
                used_cids.add(cid)

        # Pass 2b: Centroid fallback only for still-unmatched named/locked speakers
        if self._spk_names or self._locked_anchors:
            p2b_cands = []
            for cid in set(self._spk_names.keys()) | set(self._locked_anchors.keys()):
                if cid in used_cids or cid not in self._spk_centroids:
                    continue
                prev_cent = self._spk_centroids[cid]
                for ci in range(n_comp):
                    if assigned_cid[ci] is not None:
                        continue
                    csim = float(np.dot(comp[ci][3], prev_cent))
                    if csim >= 0.55:
                        p2b_cands.append(((csim, -self._spk_num(cid), -ci), cid, ci))
            p2b_cands.sort(key=lambda x: x[0], reverse=True)
            for _, cid, ci in p2b_cands:
                if assigned_cid[ci] is None and cid not in used_cids:
                    assigned_cid[ci] = cid
                    used_cids.add(cid)

        # Pass 3: Allocate smallest unused, unnamed, unlocked, non-retired ID for genuinely new clusters
        # (keeps active speaker IDs 100% immutable while avoiding ugly gaps from transient warmup clusters)
        reserved_cids = (
            used_cids
            | set(self._spk_names.keys())
            | set(self._locked_anchors.keys())
            | self._retired_cids
        )
        for ci in range(n_comp):
            if assigned_cid[ci] is None:
                k = 1
                while f"Speaker {k}" in reserved_cids:
                    k += 1
                cid = f"Speaker {k}"
                assigned_cid[ci] = cid
                used_cids.add(cid)
                reserved_cids.add(cid)

        max_num = max((self._spk_num(cid) for cid in used_cids if self._spk_num(cid) < 10**8), default=0)
        self._next_spk_num = max(1, max_num + 1)

        id_to_spk: dict[int, str] = {}
        centroids = []
        for ci, (_, members, extra, centroid) in enumerate(comp):
            label = assigned_cid[ci]
            centroids.append((label, centroid, members))
            self._spk_centroids[label] = centroid
            for m in members:
                self.turns[m]["speaker"] = label
                id_to_spk[self.turns[m]["id"]] = label
            for m in extra:
                self.turns[m]["speaker"] = label
                id_to_spk[self.turns[m]["id"]] = label

        # Prune any outlier anchor turn that legitimately split into another cluster while cid kept core anchors
        if self._locked_anchors:
            for cid, anchors in list(self._locked_anchors.items()):
                kept_here = {tid for tid in anchors if id_to_spk.get(tid) == cid}
                if kept_here and len(kept_here) < len(anchors):
                    anchors.intersection_update(kept_here)

        # Conservatively grow anchor sets for already-locked speakers on high-confidence, high-margin turns
        if self._locked_anchors and centroids:
            c_vecs = np.stack([cent for _, cent, _ in centroids]).astype(np.float32, copy=False)
            for ci, (lbl, _, members) in enumerate(centroids):
                if lbl not in self._locked_anchors:
                    continue
                anchors = self._locked_anchors[lbl]
                for m in members:
                    t = self.turns[m]
                    tid = t["id"]
                    if (tid in anchors or t.get("bridge")
                            or t["dur"] < self.min_confirm_dur or tid in self._pinned_turns):
                        continue
                    sims = c_vecs @ t["vec"]
                    my_sim = float(sims[ci])
                    if len(centroids) >= 2:
                        other_max = float(np.max(np.delete(sims, ci)))
                        if my_sim >= 0.55 and (my_sim - other_max) >= 0.12:
                            anchors.add(tid)
                    elif my_sim >= 0.58:
                        anchors.add(tid)

        self._centroids = [
            (lbl, cent) for lbl, cent, members in centroids
            if sum(self.turns[m]["dur"] for m in members) >= 2.8
        ]
        self._established = []
        for lbl, cent, members in centroids:
            tot_dur = sum(self.turns[m]["dur"] for m in members)
            if tot_dur >= 5.0 or (tot_dur >= 4.0 and len(members) >= 2):
                self._established.append((lbl, cent, tot_dur, len(members)))

        # Assign older turns outside the reservoir via vectorized matmul (preserving locked anchors)
        if older and centroids:
            cent_labels = [c[0] for c in centroids]
            cent_label_set = set(cent_labels)
            cent_mat = np.stack([c[1] for c in centroids]).astype(np.float32, copy=False)
            older_vecs = np.stack([self.turns[m]["vec"] for m in older]).astype(np.float32, copy=False)
            best_idxs = np.argmax(older_vecs @ cent_mat.T, axis=1)
            for m, bi in zip(older, best_idxs):
                tid = self.turns[m]["id"]
                own = locked_owner.get(tid)
                best_lbl = own if (own and own in cent_label_set) else cent_labels[int(bi)]
                self.turns[m]["speaker"] = best_lbl
                id_to_spk[tid] = best_lbl

        # Ensure short turns explicitly merged by the user stay with their merged cluster
        if self._must_link_sets:
            for link_set in self._must_link_sets:
                linked_labels = [id_to_spk[tid] for tid in link_set if tid in id_to_spk]
                if linked_labels:
                    target_lbl = max(set(linked_labels), key=linked_labels.count)
                    for t in self.turns:
                        if t["id"] in link_set and t["id"] not in id_to_spk:
                            t["speaker"] = target_lbl
                            id_to_spk[t["id"]] = target_lbl

        # Apply user single-line speaker pins (keeps those specific turns on target speaker without centroid drift)
        if self._pinned_turns and centroids:
            for t in self.turns:
                tid = t["id"]
                if tid not in self._pinned_turns:
                    continue
                dst_ids, dst_vec = self._pinned_turns[tid]
                counts: dict[str, int] = {}
                for did in dst_ids:
                    lbl = id_to_spk.get(did)
                    if lbl:
                        counts[lbl] = counts.get(lbl, 0) + 1
                if counts:
                    target_lbl = max(counts.items(), key=lambda kv: kv[1])[0]
                elif dst_vec is not None:
                    target_lbl = max(centroids, key=lambda c: float(np.dot(dst_vec, c[1])))[0]
                else:
                    continue
                t["speaker"] = target_lbl
                id_to_spk[tid] = target_lbl

        # Attach ultra-short utterances (< min_confirm_dur) to closest cluster or neighbor
        last_spk = centroids[0][0] if centroids else "Speaker 1"
        for t in self.turns:
            if t["id"] in id_to_spk:
                last_spk = id_to_spk[t["id"]]
                continue
            best_lbl, best_s = last_spk, -1.0
            last_s = -1.0
            for lbl, c, members in centroids:
                s_c = float(np.dot(t["vec"], c))
                s_m = max((_turn_similarity(t, self.turns[m]) for m in members), default=s_c)
                s = max(s_c, s_m)
                if lbl == last_spk:
                    last_s = s
                if s > best_s:
                    best_s = s
                    best_lbl = lbl
            if best_s >= SHORT_ATTACH_THRESHOLD:
                lbl = best_lbl
            elif self.arabic_mode and best_lbl != last_spk and best_s >= 0.22 and (best_s - last_s) >= 0.10:
                lbl = best_lbl
            else:
                lbl = last_spk
            t["speaker"] = lbl
            id_to_spk[t["id"]] = lbl
            last_spk = lbl

        for tid, lbl in id_to_spk.items():
            self._turn_prev_spk[tid] = lbl

        return id_to_spk


# --------------------------------------------------------------------------- #
# Background worker thread integrated with Engine / State / Session
# --------------------------------------------------------------------------- #
class Diarizer(threading.Thread):
    def __init__(self, extractor, state, session=None, log=print,
                 arabic_mode: bool = False, max_speakers: int = 0,
                 segmenter=None):
        super().__init__(daemon=True, name="eloquent-diarizer")
        self.extractor = extractor
        self.state = state
        self.session = session
        self.log = log
        self.arabic_mode = bool(arabic_mode)
        self.segmenter = None if self.arabic_mode else segmenter
        self.clusterer = SpeakerClusterer(arabic_mode=arabic_mode, max_speakers=max_speakers)
        self._cluster_lock = threading.Lock()
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._stop_flag = False
        self._sync_needed = False

    def set_speaker_names(self, updates: dict[str, str],
                          current_names: Optional[dict[str, str]] = None) -> None:
        with self._cluster_lock:
            self.clusterer.set_speaker_names(updates, current_names=current_names)

    def submit(self, seg_id: int, start_sec: float, stamp: str, samples: np.ndarray,
               text: str, words: list[dict]) -> None:
        if self.extractor is None:
            return
        self.q.put((seg_id, start_sec, stamp, samples, text, words))

    def has_speaker(self, label: str) -> bool:
        with self._cluster_lock:
            return self.clusterer.has_speaker(label)

    def has_any_turn(self, turn_ids: list[int]) -> bool:
        with self._cluster_lock:
            return self.clusterer.has_any_turn(turn_ids)

    def speaker_count(self) -> int:
        with self._cluster_lock:
            return self.clusterer.speaker_count()

    def manual_edit_count(self) -> int:
        with self._cluster_lock:
            return self.clusterer.manual_edit_count()

    def speaker_anchor_ids(self, label: str) -> list[int]:
        with self._cluster_lock:
            return self.clusterer.speaker_anchor_ids(label)

    def resolve_speaker_from_anchors(self, fallback_label: str, anchors: list[int]) -> str:
        with self._cluster_lock:
            return self.clusterer.resolve_speaker_from_anchors(fallback_label, anchors)

    def last_id_remap(self) -> dict[str, str]:
        with self._cluster_lock:
            return dict(self.clusterer.last_id_remap)

    def merge_speakers(self, src_label: str, dst_label: str,
                       old_names: Optional[dict[str, str]] = None
                       ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.merge_speakers(src_label, dst_label, old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def pin_turns_to_speaker(self, turn_ids: list[int], dst_label: str,
                             old_names: Optional[dict[str, str]] = None
                             ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.pin_turns_to_speaker(turn_ids, dst_label, old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def dissolve_speaker(self, src_label: str,
                         old_names: Optional[dict[str, str]] = None
                         ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.dissolve_speaker(src_label, old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def undo_last_speaker_edit(self, old_names: Optional[dict[str, str]] = None
                               ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.undo_last_speaker_edit(old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def reset_speaker_edits(self, old_names: Optional[dict[str, str]] = None
                            ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.reset_speaker_edits(old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def set_max_speakers(self, max_speakers: int,
                         old_names: Optional[dict[str, str]] = None
                         ) -> tuple[dict[int, str], dict[str, str]]:
        with self._cluster_lock:
            mapping, remapped = self.clusterer.set_max_speakers(max_speakers, old_names=old_names)
            if mapping:
                self._sync_needed = True
            return mapping, remapped

    def run(self) -> None:
        _lower_thread_priority()
        while True:
            item = self.q.get()
            if item is None:
                break
            try:
                self._process_item(*item)
            except Exception as e:
                self.log(f"[diarize] {type(e).__name__}: {e}", "warn")
            finally:
                self.q.task_done()

    def _embed(self, seg: np.ndarray) -> Optional[np.ndarray]:
        return compute_embedding(self.extractor, seg, min_rms=0.0008)

    def _process_item(self, seg_id: int, start_sec: float, stamp: str,
                      samples: np.ndarray, text: str, words: list[dict]) -> None:
        with self._cluster_lock:
            known = self.clusterer.known_centroids()
        meta_list: list[dict] = []
        parts = split_segment_by_speaker(
            samples, words, self._embed, known_speakers=known, text=text,
            segmenter=(None if self.arabic_mode else self.segmenter),
            out_meta=meta_list,
        )
        if not parts:
            with self._cluster_lock:
                fallback = self.clusterer.turns[-1]["speaker"] if self.clusterer.turns else "Speaker 1"
            self.state.apply_diarization(seg_id, [(seg_id, stamp, start_sec, text, fallback)], {})
            return

        overlap_ids: set[int] = set()
        if len(parts) == 1:
            sa, sb, ptxt, vec, vdur, subs = parts[0]
            meta = meta_list[0] if meta_list else {}
            if meta.get("overlap"):
                overlap_ids.add(seg_id)
            with self._cluster_lock:
                old_by_spk = self.clusterer._speaker_turn_ids()
                spk, mapping = self.clusterer.add_and_cluster(
                    seg_id, start_sec, vdur, vec, subs,
                    bridge_hint=bool(meta.get("heavy_overlap")),
                )
                id_remap = self.clusterer._compute_id_remap(old_by_spk, mapping)
            entries = [(seg_id, stamp, start_sec, ptxt or text, spk)]
        else:
            from .state import fmt_time
            entries = []
            mapping = {}
            with self._cluster_lock:
                old_by_spk = self.clusterer._speaker_turn_ids()
                for idx, (sa, sb, ptxt, vec, vdur, subs) in enumerate(parts):
                    meta = meta_list[idx] if idx < len(meta_list) else {}
                    sub_start = start_sec + sa / SAMPLE_RATE
                    sub_stamp = stamp if idx == 0 else fmt_time(sub_start)
                    turn_id = seg_id if idx == 0 else self.state.alloc_turn_id()
                    if meta.get("overlap"):
                        overlap_ids.add(turn_id)
                    spk, mapping = self.clusterer.add_and_cluster(
                        turn_id, sub_start, vdur, vec, subs,
                        bridge_hint=bool(meta.get("heavy_overlap")),
                    )
                    entries.append((turn_id, sub_stamp, sub_start, ptxt, spk))
                id_remap = self.clusterer._compute_id_remap(old_by_spk, mapping)

        self.state.apply_diarization(
            seg_id, entries, mapping, old_by_spk=old_by_spk, id_remap=id_remap,
            overlap_ids=overlap_ids,
        )
        self._sync_needed = True

    def flush_and_wait(self, timeout: float = 30.0) -> None:
        if not self.is_alive():
            return
        self.q.put(None)
        self.join(timeout)
