"""Transcript state shared by the recognizer, the cleaner and every UI.

Every change bumps `rev` and calls `emit(kind)` *outside* the lock, so a UI
listener can safely call `snapshot()` from its callback.

Event kinds emitted from here: "partial", "segment", "paragraph", "error".
"""

import json
import threading


def fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.emit = lambda kind, data=None: None
        self.rev = 0
        self.reset(notify=False)

    def reset(self, notify: bool = True):
        with self.lock:
            self.paragraphs = []   # {"id","time","raw","clean","tokens","status","note"}
            self.loose = []        # finalized sentences not yet in a paragraph: {"t","ts","text"}
            self.partial = ""
            self.error = ""
            self.next_id = 1
            self.rev += 1
        if notify:
            self.emit("reset")

    # -- recognizer side --------------------------------------------------
    def set_partial(self, text: str):
        with self.lock:
            if text == self.partial:
                return
            self.partial = text
            self.rev += 1
        self.emit("partial", text)

    def add_segment(self, text: str, stamp: str, info: dict):
        import time
        with self.lock:
            self.loose.append({"t": stamp, "ts": time.time(), "text": text})
            self.partial = ""
            self.rev += 1
        self.emit("segment", dict(info, text=text, t=stamp))

    # -- cleaner side -----------------------------------------------------
    def touch(self, kind: str, data=None):
        """Call after mutating paragraphs/error under `lock` (and after releasing it)."""
        with self.lock:
            self.rev += 1
        self.emit(kind, data)

    def set_error(self, err: str):
        with self.lock:
            if err == self.error:
                return
            self.error = err
            self.rev += 1
        self.emit("error", err)

    # -- readers ----------------------------------------------------------
    def _build(self) -> dict:
        return {
            "rev": self.rev,
            "partial": self.partial,
            "error": self.error,
            "loose": [{"t": x["t"], "text": x["text"]} for x in self.loose],
            "paragraphs": [
                {"id": p["id"], "time": p["time"], "raw": p["raw"], "tokens": p["tokens"],
                 "status": p["status"], "note": p["note"]}
                for p in self.paragraphs
            ],
        }

    def snapshot(self) -> dict:
        with self.lock:
            return self._build()

    def snapshot_json(self, extra: dict = None) -> str:
        with self.lock:
            data = self._build()
        if extra:
            data.update(extra)
        return json.dumps(data)

    def full_text(self, raw: bool = False) -> str:
        with self.lock:
            out = [p["raw"] if raw or not p["clean"] else p["clean"] for p in self.paragraphs]
            if self.loose:
                out.append(" ".join(x["text"] for x in self.loose))
        return "\n\n".join(out)
