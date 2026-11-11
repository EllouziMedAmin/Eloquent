"""Background proofreading: groups finished sentences into paragraphs and sends them to the LLM.

The recognizer never waits for this thread, so speech latency is unaffected.
If the LLM is slow, offline or rewrites too much, the raw text is kept.
"""

import threading
import time

from .textutil import accept_cleaned, ends_sentence, mark_changes


class Cleaner(threading.Thread):
    def __init__(self, state, session, provider, settings, log=print):
        super().__init__(daemon=True, name="eloquent-cleaner")
        self.state, self.session, self.provider, self.log = state, session, provider, log
        self.n_sent, self.min_words, self.max_words = settings.chunk_sentences, 25, 90
        self.idle_flush, self.min_sim = settings.idle_flush, settings.min_similarity
        self.flushing = False
        self.abort = False
        self.max_seg = 8                # cap per paragraph, even after a backlog
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
            # Wait briefly (~2.5s max) for background diarizer before locking a chunk into a paragraph
            ready_n = 0
            for s in segs:
                if not self.flushing and not s.get("spk_ready", True) and (now - s["ts"]) < 2.5:
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

            take = same_n
            if self.flushing or speaker_switch:
                take = min(take, self.max_seg)
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
            raw = " ".join(s["text"] for s in chunk)
            para = {
                "id": st.next_id, "time": chunk[0]["t"], "raw": raw, "clean": None,
                "tokens": None, "status": "cleaning" if self.provider else "raw",
                "note": "", "speaker": first_spk, "seg_ids": [s["id"] for s in chunk],
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
                spk = p.get("speaker", "")
                parts.append(f"{spk}: {t}" if (multi and spk) else t)
            text = " ".join(parts)
        return text[-1000:]

    # -- LLM call ----------------------------------------------------------
    def _clean(self, raw: str, ctx: str):
        if time.time() < self.cooldown_until:      # LLM keeps failing: do not even try for a while
            return None, f"{self.provider.name} unavailable, kept raw"
        text, err = None, ""
        for _ in range(2):
            try:
                text = self.provider.proofread(raw, ctx)
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
            return None, f"{self.provider.name} unavailable, kept raw"
        self.fail_streak = 0
        ok, sim = accept_cleaned(raw, text, self.min_sim)
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
                if result:
                    para.update(clean=result, tokens=tokens, status="done")
                else:
                    para.update(status="raw", note=note)
            self.state.touch("paragraph", para["id"])
            if not self.abort:
                self.session.add_clean(result or raw)

    def flush_and_wait(self, timeout: float = 30.0):
        self.flushing = True
        self.join(timeout)
        if self.is_alive():  # LLM too slow: stop waiting and save what is left as raw text
            self.abort = True
            with self.state.lock:
                left = [p["raw"] for p in self.state.paragraphs if p["status"] == "cleaning"]
                left += [x["text"] for x in self.state.loose]
            for t in left:
                self.session.add_clean(t)
