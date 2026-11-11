"""Zero-latency-impact multi-speaker diarization using sherpa-onnx (3D-Speaker CAM++).

Architecture
------------
1. The audio/ASR thread (`Engine._drain`) emits finalized text to `State` immediately
   and pushes `(seg_id, start_sec, samples, text, words)` onto a queue for `Diarizer`.
2. `Diarizer` runs on a dedicated `BELOW_NORMAL` priority daemon thread with the speaker
   embedding extractor pinned to `num_threads=1`, so Parakeet never stalls.
3. For long VAD turns (> 3 s), sliding 1.6 s voiceprint windows detect mid-chunk speaker
   handoffs (when two speakers talk back-to-back without pausing) and snap splits onto
   Parakeet's native word timestamps and punctuation boundaries.
4. Incremental UPGMA (average-linkage) clustering over confirmed utterance embeddings keeps
   speaker IDs (`Speaker 1`, `Speaker 2`, ...) accurate and prevents single-speaker drift.
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

DIARIZE_MODEL_NAME = "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
DIARIZE_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/" + DIARIZE_MODEL_NAME
)

MIN_CONFIRM_DUR = 1.1        # shorter backchannels attach instead of spawning new clusters
MERGE_THRESHOLD = 0.48       # average-linkage cosine similarity merge threshold
SPLIT_CROSS_THRESHOLD = 0.45 # sliding-window change-point threshold inside one VAD chunk
SHORT_ATTACH_THRESHOLD = 0.28
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
def _pick_word_cut(words: list[dict], start_idx: int, end_idx: int, t_target: float) -> Optional[int]:
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
        if left_dur < 1.0 or right_dur < 1.0:
            continue
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _find_window_changes(w_vecs: list[np.ndarray], w_centers: list[float],
                         cross_thresh: float = SPLIT_CROSS_THRESHOLD) -> list[float]:
    """Detect local minima in adjacent block cosine similarity indicating speaker handoffs."""
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


def split_segment_by_speaker(samples: np.ndarray, words: list[dict],
                             embed_fn: Callable[[np.ndarray], Optional[np.ndarray]],
                             win_sec: float = 1.6, hop_sec: float = 0.8
                             ) -> list[tuple[int, int, str, np.ndarray]]:
    """Return `[(sample_start, sample_end, text, embedding_vec), ...]` for a VAD segment.

    Fast path embeds once when duration <= 4.2 s or word timestamps are absent.
    Longer turns use sliding 1.6 s windows to detect zero-pause speaker switches inside one
    VAD segment and snap splits cleanly onto word/sentence boundaries.
    """
    dur = len(samples) / SAMPLE_RATE
    full_text = " ".join(w["w"] for w in words).strip() if words else ""

    if dur <= 4.2 or not words or len(words) < 8:
        vec = embed_fn(samples)
        if vec is None:
            return []
        return [(0, len(samples), full_text, vec)]

    # Trim leading/trailing silence using ASR word spans for clean ASTP embeddings
    t_start = max(0.0, words[0]["t0"] - 0.15)
    t_end = min(dur, words[-1]["t1"] + 0.15)
    win_n = int(win_sec * SAMPLE_RATE)
    hop_n = int(hop_sec * SAMPLE_RATE)
    s0 = int(t_start * SAMPLE_RATE)
    s1 = int(t_end * SAMPLE_RATE)

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

    if not w_vecs:
        vec = embed_fn(samples)
        return [(0, len(samples), full_text, vec)] if vec is not None else []

    raw_cuts = _find_window_changes(w_vecs, w_centers)
    if not raw_cuts:
        # Always compute the full-slice ASTP embedding so vectors stay in one consistent space
        vec = embed_fn(samples[s0:s1])
        if vec is None:
            vec = embed_fn(samples)
        return [(0, len(samples), full_text, vec)] if vec is not None else []

    # Map acoustic cut times to clean word boundaries
    word_cuts = [0]
    for t_cut in raw_cuts:
        idx = _pick_word_cut(words, word_cuts[-1], len(words), t_cut)
        if idx is not None and idx > word_cuts[-1]:
            word_cuts.append(idx)
    word_cuts.append(len(words))

    # Build slices and verify adjacent parts actually differ vocally
    parts: list[tuple[int, int, int, int, np.ndarray]] = []
    for a, b in zip(word_cuts[:-1], word_cuts[1:]):
        sa = 0 if a == 0 else max(0, int((words[a]["t0"] - 0.05) * SAMPLE_RATE))
        sb = len(samples) if b == len(words) else min(len(samples), int((words[b - 1]["t1"] + 0.05) * SAMPLE_RATE))
        pvec = embed_fn(samples[sa:sb])
        if pvec is None:
            t_a, t_b = sa / SAMPLE_RATE, sb / SAMPLE_RATE
            near = [v for t, v in zip(w_centers, w_vecs) if t_a <= t <= t_b] or [w_vecs[0]]
            pvec = np.mean(np.stack(near), axis=0)
            pvec /= max(1e-8, float(np.linalg.norm(pvec)))
            pvec = pvec.astype(np.float32)
        if parts and float(np.dot(parts[-1][4], pvec)) >= 0.50:
            prev_sa, _, prev_wa, _, _ = parts.pop()
            merged_vec = embed_fn(samples[prev_sa:sb])
            if merged_vec is None:
                merged_vec = pvec
            parts.append((prev_sa, sb, prev_wa, b, merged_vec))
        else:
            parts.append((sa, sb, a, b, pvec))

    out = []
    for sa, sb, a, b, vec in parts:
        txt = " ".join(w["w"] for w in words[a:b]).strip()
        if txt:
            out.append((sa, sb, txt, vec))
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
        self.turns: list[dict] = []  # {"id", "t0", "dur", "vec", "speaker"}

    def add_and_cluster(self, turn_id: int, start_sec: float, dur: float,
                        vec: np.ndarray) -> tuple[str, dict[int, str]]:
        """Add an utterance embedding and return `(assigned_speaker, full_id_to_speaker_map)`."""
        self.turns.append({
            "id": turn_id, "t0": start_sec, "dur": dur,
            "vec": np.asarray(vec, dtype=np.float32), "speaker": "",
        })
        mapping = self._recluster()
        return mapping[turn_id], mapping

    def _recluster(self) -> dict[int, str]:
        confirmed = [i for i, t in enumerate(self.turns) if t["dur"] >= self.min_confirm_dur]
        if not confirmed:
            # Nothing long enough to confirm yet: treat initial short speech as Speaker 1
            for t in self.turns:
                t["speaker"] = "Speaker 1"
            return {t["id"]: t["speaker"] for t in self.turns}

        # Cap UPGMA size to most recent 240 confirmed turns to keep O(1 ms) on multi-hour recordings
        if len(confirmed) > 240:
            confirmed_sub = confirmed[-240:]
            older = confirmed[:-240]
        else:
            confirmed_sub = confirmed
            older = []

        vecs = np.stack([self.turns[i]["vec"] for i in confirmed_sub])
        weights = np.array([min(4.0, max(0.75, self.turns[i]["dur"])) for i in confirmed_sub], dtype=np.float64)
        sim_mat = (vecs @ vecs.T).astype(np.float64)

        # Clusters stored as lists of local indices into `confirmed_sub`
        clusters: list[list[int]] = [[i] for i in range(len(confirmed_sub))]
        # Maintain pairwise weighted average similarity between active clusters
        n = len(clusters)
        pair_sum = np.zeros((n, n), dtype=np.float64)
        pair_wt = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                w = weights[i] * weights[j]
                pair_wt[i, j] = pair_wt[j, i] = w
                pair_sum[i, j] = pair_sum[j, i] = sim_mat[i, j] * w

        active = list(range(n))
        while len(active) > 1:
            best_sim = -1.0
            best_a = best_b = -1
            for ia in range(len(active)):
                a = active[ia]
                for ib in range(ia + 1, len(active)):
                    b = active[ib]
                    avg = pair_sum[a, b] / max(1e-8, pair_wt[a, b])
                    if avg > best_sim:
                        best_sim = avg
                        best_a, best_b = a, b
            if best_sim < self.merge_threshold or best_a < 0:
                break
            clusters[best_a].extend(clusters[best_b])
            clusters[best_b] = []
            active.remove(best_b)
            for c in active:
                if c == best_a:
                    continue
                pair_sum[best_a, c] = pair_sum[c, best_a] = pair_sum[best_a, c] + pair_sum[best_b, c]
                pair_wt[best_a, c] = pair_wt[c, best_a] = pair_wt[best_a, c] + pair_wt[best_b, c]

        # Fold noisy tiny singletons (<1.9s total, e.g. brief coughs or fragments) into established
        # clusters if weakly similar (>=0.30), while preserving clearly distinct short speakers (<0.30).
        def _centroid(members_idx):
            wv = [self.turns[m]["vec"] * min(4.0, self.turns[m]["dur"]) for m in members_idx]
            c = np.sum(np.stack(wv), axis=0)
            c /= max(1e-8, float(np.linalg.norm(c)))
            return c.astype(np.float32)

        groups = [[confirmed_sub[idx] for idx in clusters[cid]] for cid in active]
        major = [g for g in groups if sum(self.turns[m]["dur"] for m in g) >= 2.2]
        if major:
            kept = [list(g) for g in major]
            kept_vecs = [_centroid(g) for g in kept]
            for g in groups:
                gdur = sum(self.turns[m]["dur"] for m in g)
                if gdur >= 2.2:
                    continue
                gv = _centroid(g)
                sims = [float(np.dot(gv, kv)) for kv in kept_vecs]
                bi = int(np.argmax(sims))
                if sims[bi] >= 0.30 or gdur < 1.35:
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
            centroids.append((label, centroid))
            for m in members:
                self.turns[m]["speaker"] = label
                id_to_spk[self.turns[m]["id"]] = label

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
            v = t["vec"]
            best_lbl, best_s = max(((lbl, float(np.dot(v, c))) for lbl, c in centroids), key=lambda x: x[1])
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
        parts = split_segment_by_speaker(samples, words, self._embed)
        if not parts:
            fallback = self.clusterer.turns[-1]["speaker"] if self.clusterer.turns else "Speaker 1"
            self.state.apply_diarization(seg_id, [(seg_id, stamp, start_sec, text, fallback)], {})
            return

        if len(parts) == 1:
            sa, sb, ptxt, vec = parts[0]
            dur = (sb - sa) / SAMPLE_RATE
            spk, mapping = self.clusterer.add_and_cluster(seg_id, start_sec, dur, vec)
            entries = [(seg_id, stamp, start_sec, ptxt or text, spk)]
        else:
            from .state import fmt_time
            entries = []
            mapping = {}
            for idx, (sa, sb, ptxt, vec) in enumerate(parts):
                sub_start = start_sec + sa / SAMPLE_RATE
                sub_stamp = stamp if idx == 0 else fmt_time(sub_start)
                turn_id = seg_id if idx == 0 else self.state.alloc_turn_id()
                dur = (sb - sa) / SAMPLE_RATE
                spk, mapping = self.clusterer.add_and_cluster(turn_id, sub_start, dur, vec)
                entries.append((turn_id, sub_stamp, sub_start, ptxt, spk))

        self.state.apply_diarization(seg_id, entries, mapping)
        self._sync_needed = True

    def flush_and_wait(self, timeout: float = 30.0) -> None:
        if not self.is_alive():
            return
        self.q.put(None)
        self.join(timeout)
