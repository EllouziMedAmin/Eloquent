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
SENTENCE_END = (".", "?", "!", "…")


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


def compute_embedding(extractor, samples: np.ndarray) -> Optional[np.ndarray]:
    """Extract an L2-normalized speaker embedding vector, or None if clip is silent/too short."""
    if len(samples) < int(0.35 * SAMPLE_RATE):
        return None
    if float(np.sqrt(np.mean(samples * samples))) < 0.002:
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
                          max_cuts: int = 8) -> list[int]:
    """Return candidate word split indices at sentence endings or pauses (`>= 0.14 s`)."""
    n = len(words)
    if n < 6:
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


def _voiced_samples(samples: np.ndarray, words: list[dict],
                    pad_sec: float = 0.08
                    ) -> tuple[np.ndarray, float]:
    """Trim leading/trailing VAD silence around ASR word spans and compute true voiced duration."""
    n = len(samples)
    if not words:
        return samples, n / SAMPLE_RATE

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
                             known_speakers: Optional[list[tuple[str, np.ndarray]]] = None
                             ) -> list[tuple[int, int, str, np.ndarray, float, list[np.ndarray]]]:
    """Return `[(sample_start, sample_end, text, embedding_vec, voiced_dur, sub_vecs), ...]`.

    Tests sentence/pause boundaries first (fast) and runs sliding `1.5 s / 0.75 s` acoustic
    change detection only across long unbroken spans (`>= 3.9 s`), verified against pure slice
    embeddings and known speaker centroids.
    """
    dur = len(samples) / SAMPLE_RATE
    full_text = " ".join(w["w"] for w in words).strip() if words else ""

    def _single_turn():
        voiced, vdur = _voiced_samples(samples, words)
        vec = embed_fn(voiced)
        if vec is None:
            vec = embed_fn(samples)
        return [(0, len(samples), full_text, vec, vdur, [])] if vec is not None else []

    if dur < 2.6 or not words or len(words) < 6:
        return _single_turn()

    t_start = max(0.0, words[0]["t0"] - 0.10)
    t_end = min(dur, words[-1]["t1"] + 0.10)
    span_dur = t_end - t_start
    max_cuts = max(4, min(8, int(span_dur // 1.6) + 1))
    ling_cuts = _find_linguistic_cuts(words, min_side_dur=1.25, max_cuts=max_cuts)

    # Fast path for single unbroken turns under 4.0s with no sentence/pause boundaries
    if span_dur < 4.0 and not ling_cuts:
        return _single_turn()

    # Only scan sliding acoustic windows across unbroken spans (>= 3.9s) without punctuation/pauses
    bounds = [0] + ling_cuts + [len(words)]
    cut_set = set(ling_cuts)
    w_vecs_all: list[np.ndarray] = []
    w_centers_all: list[float] = []
    win_n = int(win_sec * SAMPLE_RATE)
    hop_n = int(hop_sec * SAMPLE_RATE)

    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = words[a:b]
        if len(sub) < 6:
            continue
        sp_t0 = max(0.0, sub[0]["t0"] - 0.08)
        sp_t1 = min(dur, sub[-1]["t1"] + 0.08)
        if sp_t1 - sp_t0 < 3.9:
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
            idx = _pick_word_cut(words, prev_c, b, t_cut)
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
        # Keep brief boundary crosstalk out of voiceprints when slice is long enough
        core = voiced_slice
        if trim_edges and len(voiced_slice) >= int(1.9 * SAMPLE_RATE):
            guard = int(0.10 * SAMPLE_RATE)
            l_off = guard if a > 0 else 0
            r_off = len(voiced_slice) - (guard if b < len(words) else 0)
            if r_off - l_off >= int(1.2 * SAMPLE_RATE):
                core = voiced_slice[l_off:r_off]
        pvec = embed_fn(core)
        if pvec is None:
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
                 min_confirm_dur: float = MIN_CONFIRM_DUR):
        self.merge_threshold = merge_threshold
        self.min_confirm_dur = min_confirm_dur
        self.turns: list[dict] = []  # {"id", "t0", "dur", "vec", "subs", "speaker", "bridge"}
        self._centroids: list[tuple[str, np.ndarray]] = []
        self._established: list[tuple[str, np.ndarray, float, int]] = []

    def known_centroids(self) -> list[tuple[str, np.ndarray]]:
        return list(self._centroids)

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
                        vec: np.ndarray, sub_vecs: Optional[list[np.ndarray]] = None
                        ) -> tuple[str, dict[int, str]]:
        """Add an utterance embedding and return `(assigned_speaker, full_id_to_speaker_map)`."""
        v = np.asarray(vec, dtype=np.float32)
        self.turns.append({
            "id": turn_id, "t0": start_sec, "dur": dur,
            "vec": v,
            "subs": list(sub_vecs) if sub_vecs else [],
            "speaker": "",
            "bridge": self._is_overlap_bridge(v, dur),
        })
        mapping = self._recluster()
        return mapping[turn_id], mapping

    def _recluster(self) -> dict[int, str]:
        confirmed = [
            i for i, t in enumerate(self.turns)
            if t["dur"] >= self.min_confirm_dur and not t.get("bridge")
        ]
        if not confirmed:
            # Nothing long enough to confirm yet: treat initial short speech as Speaker 1
            for t in self.turns:
                t["speaker"] = "Speaker 1"
            self._centroids = []
            self._established = []
            return {t["id"]: t["speaker"] for t in self.turns}

        # Cap UPGMA size to most recent 240 confirmed turns to keep O(1 ms) on multi-hour recordings
        if len(confirmed) > 240:
            confirmed_sub = confirmed[-240:]
            older = confirmed[:-240]
        else:
            confirmed_sub = confirmed
            older = []

        weights = np.array([min(4.0, max(0.75, self.turns[i]["dur"])) for i in confirmed_sub], dtype=np.float64)
        n = len(confirmed_sub)
        sim_mat = np.eye(n, dtype=np.float64)
        for i in range(n):
            ti = self.turns[confirmed_sub[i]]
            for j in range(i + 1, n):
                tj = self.turns[confirmed_sub[j]]
                s = _turn_similarity(ti, tj)
                sim_mat[i, j] = sim_mat[j, i] = s

        # Clusters stored as lists of local indices into `confirmed_sub`
        clusters: list[list[int]] = [[i] for i in range(n)]
        # Maintain pairwise weighted average similarity between active clusters
        pair_sum = np.zeros((n, n), dtype=np.float64)
        pair_wt = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[i] * weights[j]
                pair_wt[i, j] = pair_wt[j, i] = w
                pair_sum[i, j] = pair_sum[j, i] = sim_mat[i, j] * w

        def _centroid(members_idx):
            wv = [self.turns[m]["vec"] * min(4.0, self.turns[m]["dur"]) for m in members_idx]
            c = np.sum(np.stack(wv), axis=0)
            c /= max(1e-8, float(np.linalg.norm(c)))
            return c.astype(np.float32)

        def _can_merge(idxs_a: list[int], idxs_b: list[int], require_established_only: bool = False) -> bool:
            ma = [confirmed_sub[idx] for idx in idxs_a]
            mb = [confirmed_sub[idx] for idx in idxs_b]
            da = sum(self.turns[m]["dur"] for m in ma)
            db = sum(self.turns[m]["dur"] for m in mb)
            if require_established_only and (da < 4.0 or db < 4.0):
                return True
            csim = float(np.dot(_centroid(ma), _centroid(mb)))
            if da >= 6.0 and db >= 6.0:
                med = float(np.median(sim_mat[np.ix_(idxs_a, idxs_b)]))
                if len(idxs_a) >= 2 and len(idxs_b) >= 2:
                    return csim >= ESTABLISHED_MULTI_MERGE and med >= ESTABLISHED_MULTI_MEDIAN
                return csim >= ESTABLISHED_MERGE and med >= ESTABLISHED_MEDIAN_MIN
            return csim >= CENTROID_MERGE

        active = list(range(n))
        blocked: set[tuple[int, int]] = set()
        while len(active) > 1:
            best_sim = -1.0
            best_a = best_b = -1
            for ia in range(len(active)):
                a = active[ia]
                for ib in range(ia + 1, len(active)):
                    b = active[ib]
                    if (a, b) in blocked:
                        continue
                    avg = pair_sum[a, b] / max(1e-8, pair_wt[a, b])
                    if avg > best_sim:
                        best_sim = avg
                        best_a, best_b = a, b
            if best_sim < self.merge_threshold or best_a < 0:
                break
            if not _can_merge(clusters[best_a], clusters[best_b], require_established_only=True):
                blocked.add((best_a, best_b))
                blocked.add((best_b, best_a))
                continue
            clusters[best_a].extend(clusters[best_b])
            clusters[best_b] = []
            active.remove(best_b)
            for c in active:
                if c == best_a:
                    continue
                pair_sum[best_a, c] = pair_sum[c, best_a] = pair_sum[best_a, c] + pair_sum[best_b, c]
                pair_wt[best_a, c] = pair_wt[c, best_a] = pair_wt[best_a, c] + pair_wt[best_b, c]

        # Consolidate clusters whose weighted speaker centroids match closely, while shielding
        # established independent speakers against cross-talk / late-bridge collapse.
        local_groups = [clusters[cid] for cid in active]
        merged_c = True
        while merged_c and len(local_groups) > 1:
            merged_c = False
            for ia in range(len(local_groups)):
                for ib in range(ia + 1, len(local_groups)):
                    if _can_merge(local_groups[ia], local_groups[ib]):
                        local_groups[ia].extend(local_groups[ib])
                        del local_groups[ib]
                        merged_c = True
                        break
                if merged_c:
                    break
        groups = [[confirmed_sub[idx] for idx in g] for g in local_groups]

        major = [g for g in groups if sum(self.turns[m]["dur"] for m in g) >= 2.8]
        if major:
            kept = [list(g) for g in major]
            kept_vecs = [_centroid(g) for g in kept]
            for g in groups:
                gdur = sum(self.turns[m]["dur"] for m in g)
                if gdur >= 2.8:
                    continue
                gv = _centroid(g)
                sims = [float(np.dot(gv, kv)) for kv in kept_vecs]
                bi = int(np.argmax(sims))
                # Fold brief isolated fragments into the closest established speaker only when similar
                need = 0.35 if gdur < 1.8 else 0.42
                if sims[bi] >= need:
                    kept[bi].extend(g)
                    kept_vecs[bi] = _centroid(kept[bi])
                else:
                    kept.append(list(g))
                    kept_vecs.append(gv)
            groups = kept

        # Build cluster centroids and order stably by earliest turn appearance
        comp = []
        for members in groups:
            first_t0 = min(self.turns[m]["t0"] for m in members)
            comp.append((first_t0, members, _centroid(members)))
        comp.sort(key=lambda x: x[0])

        id_to_spk: dict[int, str] = {}
        centroids = []
        for spk_num, (_, members, centroid) in enumerate(comp, 1):
            label = f"Speaker {spk_num}"
            centroids.append((label, centroid, members))
            for m in members:
                self.turns[m]["speaker"] = label
                id_to_spk[self.turns[m]["id"]] = label
        self._centroids = [
            (lbl, cent) for lbl, cent, members in centroids
            if sum(self.turns[m]["dur"] for m in members) >= 2.8
        ]
        self._established = []
        for lbl, cent, members in centroids:
            tot_dur = sum(self.turns[m]["dur"] for m in members)
            if tot_dur >= 5.0 or (tot_dur >= 4.0 and len(members) >= 2):
                self._established.append((lbl, cent, tot_dur, len(members)))

        # Preserve labels for older turns beyond the sliding window by matching centroids
        for m in older:
            v = self.turns[m]["vec"]
            best_lbl = max(centroids, key=lambda c: float(np.dot(v, c[1])))[0]
            self.turns[m]["speaker"] = best_lbl
            id_to_spk[self.turns[m]["id"]] = best_lbl

        # Attach ultra-short utterances (< min_confirm_dur) to closest cluster or neighbor
        last_spk = "Speaker 1"
        for t in self.turns:
            if t["id"] in id_to_spk:
                last_spk = id_to_spk[t["id"]]
                continue
            best_lbl, best_s = last_spk, -1.0
            for lbl, c, members in centroids:
                s_c = float(np.dot(t["vec"], c))
                s_m = max((_turn_similarity(t, self.turns[m]) for m in members), default=s_c)
                s = max(s_c, s_m)
                if s > best_s:
                    best_s = s
                    best_lbl = lbl
            lbl = best_lbl if best_s >= SHORT_ATTACH_THRESHOLD else last_spk
            t["speaker"] = lbl
            id_to_spk[t["id"]] = lbl
            last_spk = lbl

        return id_to_spk


# --------------------------------------------------------------------------- #
# Background worker thread integrated with Engine / State / Session
# --------------------------------------------------------------------------- #
class Diarizer(threading.Thread):
    def __init__(self, extractor, state, session=None, log=print):
        super().__init__(daemon=True, name="eloquent-diarizer")
        self.extractor = extractor
        self.state = state
        self.session = session
        self.log = log
        self.clusterer = SpeakerClusterer()
        self.q: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._stop_flag = False
        self._sync_needed = False

    def submit(self, seg_id: int, start_sec: float, stamp: str, samples: np.ndarray,
               text: str, words: list[dict]) -> None:
        if self.extractor is None:
            return
        self.q.put((seg_id, start_sec, stamp, samples, text, words))

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
        return compute_embedding(self.extractor, seg)

    def _process_item(self, seg_id: int, start_sec: float, stamp: str,
                      samples: np.ndarray, text: str, words: list[dict]) -> None:
        parts = split_segment_by_speaker(
            samples, words, self._embed, known_speakers=self.clusterer.known_centroids(),
        )
        if not parts:
            fallback = self.clusterer.turns[-1]["speaker"] if self.clusterer.turns else "Speaker 1"
            self.state.apply_diarization(seg_id, [(seg_id, stamp, start_sec, text, fallback)], {})
            return

        if len(parts) == 1:
            sa, sb, ptxt, vec, vdur, subs = parts[0]
            spk, mapping = self.clusterer.add_and_cluster(seg_id, start_sec, vdur, vec, subs)
            entries = [(seg_id, stamp, start_sec, ptxt or text, spk)]
        else:
            from .state import fmt_time
            entries = []
            mapping = {}
            for idx, (sa, sb, ptxt, vec, vdur, subs) in enumerate(parts):
                sub_start = start_sec + sa / SAMPLE_RATE
                sub_stamp = stamp if idx == 0 else fmt_time(sub_start)
                turn_id = seg_id if idx == 0 else self.state.alloc_turn_id()
                spk, mapping = self.clusterer.add_and_cluster(turn_id, sub_start, vdur, vec, subs)
                entries.append((turn_id, sub_stamp, sub_start, ptxt, spk))

        self.state.apply_diarization(seg_id, entries, mapping)
        self._sync_needed = True

    def flush_and_wait(self, timeout: float = 30.0) -> None:
        if not self.is_alive():
            return
        self.q.put(None)
        self.join(timeout)
