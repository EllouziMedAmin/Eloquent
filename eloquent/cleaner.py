"""Background proofreading: groups finished sentences into paragraphs and sends them to the LLM.

The recognizer never waits for this thread, so speech latency is unaffected.
If the LLM is slow, offline or rewrites too much, the raw text is kept.
"""

import threading
import time

from .config import is_arabic_speech
from .textutil import accept_bilingual_fix, accept_cleaned, ends_sentence, mark_changes


DICTATION_PAUSE_SEC = 2.8


def _join_raw_segments(chunk: list[dict], preserve_pauses: bool) -> str:
    if not preserve_pauses:
        return " ".join(s["text"] for s in chunk)
    parts = []
    for i, s in enumerate(chunk):
        txt = s["text"].strip()
        if not txt:
            continue
        if parts:
            parts.append("\n\n" if (i > 0 and s.get("gap_before", 0.0) >= DICTATION_PAUSE_SEC) else " ")
        parts.append(txt)
    return "".join(parts)


class Cleaner(threading.Thread):
    def __init__(self, state, session, provider, settings, mode: str = "meeting",
                 selected_text: str = "", compose_history: list = None, log=print):
        super().__init__(daemon=True, name="eloquent-cleaner")
        self.state, self.session, self.provider, self.log = state, session, provider, log
        self.compose = (mode == "compose")
        self.dictation = (mode == "dictation")
        self.selected_text = selected_text
        self.compose_history = list(compose_history or ())
        self.speech_language = (settings.resolved_speech_language()
                                if hasattr(settings, "resolved_speech_language")
                                else getattr(settings, "speech_language", "parakeet"))
        self.min_sim = settings.min_similarity
        if self.compose:
            # Hold the complete spoken prompt until stop so it generates as a single request
            self.n_sent, self.min_words, self.max_words = 999, 99999, 99999
            self.idle_flush, self.max_seg = 99999.0, 256
        elif self.dictation:
            # Keep full context together for prompts/posts; only split mid-stream on very long dictations
            self.n_sent, self.min_words, self.max_words = 12, 180, 220
            self.idle_flush, self.max_seg = 9999.0, 24
        else:
            self.n_sent, self.min_words, self.max_words = settings.chunk_sentences, 25, 90
            self.idle_flush, self.max_seg = settings.idle_flush, 8
        self.flushing = False
        self.abort = False
        self.fail_streak, self.cooldown_until = 0, 0.0
        self.last_error = ""

    # -- chunking ----------------------------------------------------------
    def _take(self):
        st = self.state
        with st.lock:
            segs = st.loose
            if not segs:
                return None
            now = time.time()
            # Wait briefly (~6.0s max) for background diarizer before locking a chunk into a paragraph
            ready_n = 0
            for s in segs:
                if not self.flushing and not s.get("spk_ready", True) and (now - s["ts"]) < 6.0:
                    break
                ready_n += 1
            if ready_n == 0:
                return None

            first_spk = segs[0].get("speaker", "")
            same_n = 0
            for s in segs[:ready_n]:
                if same_n > 0 and s.get("speaker", "") != first_spk:
                    break
                same_n += 1
            speaker_switch = same_n < ready_n

            if self.compose and not self.flushing:
                return None

            take = same_n
            if self.flushing or speaker_switch:
                take = take if self.compose else min(take, self.max_seg)
            else:
                grp = segs[:same_n]
                words = sum(len(s["text"].split()) for s in grp)
                idle = now - grp[-1]["ts"]
                ready = ((len(grp) >= self.n_sent and words >= self.min_words)
                         or words >= self.max_words or idle >= self.idle_flush)
                if not ready:
                    return None
                # A last sentence without end punctuation was probably cut by the recorder:
                # keep it for the next chunk so it can be joined with its continuation.
                if take > self.max_seg:
                    take = self.max_seg
                elif idle < self.idle_flush and not ends_sentence(grp[-1]["text"]):
                    take -= 1
                if take == 0:
                    return None
            chunk = segs[:take]
            del segs[:take]
            raw = _join_raw_segments(chunk, preserve_pauses=(self.dictation or self.compose))
            break_before = bool(chunk and chunk[0].get("gap_before", 0.0) >= DICTATION_PAUSE_SEC)
            para = {
                "id": st.next_id, "time": chunk[0]["t"], "raw": raw, "clean": None,
                "tokens": None, "status": "cleaning" if self.provider else "raw",
                "note": "", "speaker": first_spk, "break_before": break_before,
                "seg_ids": [s["id"] for s in chunk],
                "segs": [{"id": s["id"], "t": s["t"], "text": s["text"], "speaker": s.get("speaker", "")}
                         for s in chunk],
            }
            st.next_id += 1
            st.paragraphs.append(para)
        st.touch("paragraph", para["id"])
        return para

    def _context(self, current) -> str:
        with self.state.lock:
            prev = [p for p in self.state.paragraphs if p is not current and p["status"] != "cleaning"][-3:]
            multi = self.state._multi_speaker_locked()
            parts = []
            for p in prev:
                t = p["clean"] or p["raw"]
                spk = self.state._disp_spk(p.get("speaker", ""))
                parts.append(f"{spk}: {t}" if (multi and spk) else t)
            text = " ".join(parts)
        return text[-1000:]

    # -- LLM call ----------------------------------------------------------
    def _clean(self, raw: str, ctx: str):
        if time.time() < self.cooldown_until:      # LLM keeps failing: do not even try for a while
            return None, f"{self.provider.name} unavailable, kept raw"
        prompt_mode = "dictation" if self.dictation else "proofread"
        text, err = None, ""
        for _ in range(2):
            try:
                if self.compose:
                    text = self.provider.compose(raw, self.selected_text, self.compose_history)
                else:
                    text = self.provider.proofread(raw, ctx, mode=prompt_mode)
                if text:
                    break
                err = "empty reply"
            except Exception as e:  # network, quota, bad key...
                err = f"{type(e).__name__}: {str(e)[:160]}"
                time.sleep(1.0)
        if not text:
            self._set_error(err)
            self.fail_streak += 1
            if self.fail_streak >= 3:              # 3 failed paragraphs in a row -> pause 30 s
                self.cooldown_until = time.time() + 30
                self.fail_streak = 0
            return None, (err or f"{self.provider.name} unavailable") if self.compose else f"{self.provider.name} unavailable, kept raw"
        self.fail_streak = 0
        if not self.compose:
            if is_arabic_speech(self.speech_language):
                ok, sim = accept_bilingual_fix(
                    raw, text, self.min_sim, context=ctx, dictation=self.dictation,
                )
            else:
                ok, sim = accept_cleaned(raw, text, self.min_sim, dictation=self.dictation)
            if not ok:
                return None, f"kept raw (model changed too much, {sim:.0%} similar)"
        self._set_error("")
        return text, ""

    def _set_error(self, err: str):
        self.state.set_error(err)
        if err and err != self.last_error:
            self.log(f"[clean] {err}", "warn")
        self.last_error = err

    # -- main loop ---------------------------------------------------------
    def run(self):
        while True:
            para = self._take()
            if para is None:
                with self.state.lock:
                    done = self.flushing and not self.state.loose
                if done:
                    break
                time.sleep(0.2)
                continue
            raw = para["raw"]
            if self.provider is None:
                result, note = None, ""
            else:
                result, note = self._clean(raw, self._context(para))
            tokens = mark_changes(raw, result) if result else None   # outside the lock
            with self.state.lock:
                if para["raw"] != raw:
                    if para.get("status") == "cleaning":
                        para["status"] = "raw"
                elif result:
                    para.update(clean=result, tokens=tokens, status="done")
                else:
                    para.update(status="raw", note=note)
            self.state.touch("paragraph", para["id"])
            if not self.abort and self.session is not None:
                self.session.add_clean(result or raw)

    def flush_and_wait(self, timeout: float = 30.0):
        self.flushing = True
        self.join(timeout)
        if self.is_alive():  # LLM too slow: stop waiting and save what is left as raw text
            self.abort = True
            if self.session is not None:
                with self.state.lock:
                    left = [p["raw"] for p in self.state.paragraphs if p["status"] == "cleaning"]
                    left += [x["text"] for x in self.state.loose]
                for t in left:
                    self.session.add_clean(t)
