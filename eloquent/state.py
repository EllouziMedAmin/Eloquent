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
            self.paragraphs = []   # {"id","time","raw","clean","tokens","status","note","speaker","seg_ids"}
            self.loose = []        # {"id","t","start_sec","ts","text","speaker","spk_ready"}
            self.partial = ""
            self.error = ""
            self.report = None     # {"format_id","label","status","text","file","error"}
            self.next_id = 1
            self.next_turn_id = 1
            self.rev += 1
        if notify:
            self.emit("reset")

    def alloc_turn_id(self) -> int:
        with self.lock:
            tid = self.next_turn_id
            self.next_turn_id += 1
            return tid

    # -- recognizer side --------------------------------------------------
    def set_partial(self, text: str):
        with self.lock:
            if text == self.partial:
                return
            self.partial = text
            self.rev += 1
        self.emit("partial", text)

    def add_segment(self, text: str, stamp: str, info: dict,
                    start_sec: float = 0.0, spk_ready: bool = True) -> int:
        import time
        with self.lock:
            tid = self.next_turn_id
            self.next_turn_id += 1
            self.loose.append({
                "id": tid, "t": stamp, "start_sec": start_sec,
                "ts": time.time(), "text": text,
                "speaker": "", "spk_ready": spk_ready,
            })
            self.partial = ""
            self.rev += 1
        self.emit("segment", dict(info, id=tid, text=text, t=stamp))
        return tid

    # -- diarizer side ----------------------------------------------------
    def apply_diarization(self, seg_id: int, parts: list[tuple[int, str, float, str, str]],
                          mapping: dict[int, str]):
        """Attach speaker labels (and any mid-chunk splits) and propagate converged cluster IDs."""
        import time
        now = time.time()
        with self.lock:
            idx = next((i for i, x in enumerate(self.loose) if x["id"] == seg_id), None)
            if idx is not None and parts:
                orig_ts = self.loose[idx]["ts"]
                repl = [
                    {
                        "id": tid, "t": st, "start_sec": t0,
                        "ts": orig_ts if j == 0 else now, "text": txt,
                        "speaker": mapping.get(tid, spk), "spk_ready": True,
                    }
                    for j, (tid, st, t0, txt, spk) in enumerate(parts)
                ]
                self.loose[idx:idx + 1] = repl
            elif parts:
                # Cleaner already pulled seg_id into a paragraph before diarization finished
                first_tid = parts[0][0]
                first_spk = mapping.get(first_tid, parts[0][4])
                for p in self.paragraphs:
                    if seg_id in p.get("seg_ids", ()):
                        p["speaker"] = first_spk
                        break

            if mapping:
                for x in self.loose:
                    if x["id"] in mapping:
                        x["speaker"] = mapping[x["id"]]
                for p in self.paragraphs:
                    for tid in p.get("seg_ids", ()):
                        if tid in mapping:
                            p["speaker"] = mapping[tid]
                            break
            self.rev += 1
        self.emit("speakers", seg_id)

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

    def set_report(self, report: dict):
        with self.lock:
            self.report = dict(report) if report else None
            self.rev += 1
        self.emit("report", self.report)

    # -- readers ----------------------------------------------------------
    def _multi_speaker_locked(self) -> bool:
        seen = set()
        for p in self.paragraphs:
            if p.get("speaker"):
                seen.add(p["speaker"])
                if len(seen) >= 2:
                    return True
        for x in self.loose:
            if x.get("speaker"):
                seen.add(x["speaker"])
                if len(seen) >= 2:
                    return True
        return False

    def _build(self) -> dict:
        return {
            "rev": self.rev,
            "partial": self.partial,
            "error": self.error,
            "multi_speaker": self._multi_speaker_locked(),
            "report": dict(self.report) if self.report else None,
            "loose": [
                {"t": x["t"], "text": x["text"], "speaker": x.get("speaker", "")}
                for x in self.loose
            ],
            "paragraphs": [
                {"id": p["id"], "time": p["time"], "raw": p["raw"], "tokens": p["tokens"],
                 "status": p["status"], "note": p["note"], "speaker": p.get("speaker", "")}
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
            multi = self._multi_speaker_locked()
            out = []
            for p in self.paragraphs:
                txt = p["raw"] if raw or not p["clean"] else p["clean"]
                spk = p.get("speaker", "")
                out.append(f"{spk}: {txt}" if (multi and spk) else txt)
            if self.loose:
                # Group consecutive loose sentences of same speaker
                cur_spk, cur_txt = None, []
                for x in self.loose:
                    spk = x.get("speaker", "") if multi else ""
                    if cur_txt and spk != cur_spk:
                        line = " ".join(cur_txt)
                        out.append(f"{cur_spk}: {line}" if cur_spk else line)
                        cur_txt = [x["text"]]
                        cur_spk = spk
                    else:
                        cur_spk = spk if cur_spk is None else cur_spk
                        cur_txt.append(x["text"])
                if cur_txt:
                    line = " ".join(cur_txt)
                    out.append(f"{cur_spk}: {line}" if cur_spk else line)
        return "\n\n".join(out)

    def export_records(self) -> tuple[bool, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
        """Return `(multi_speaker, [(stamp, speaker, raw)], [(stamp, speaker, clean_or_raw)])`."""
        with self.lock:
            multi = self._multi_speaker_locked()
            raw_items = [(p["time"], p.get("speaker", ""), p["raw"]) for p in self.paragraphs]
            raw_items += [(x["t"], x.get("speaker", ""), x["text"]) for x in self.loose]
            clean_items = [
                (p["time"], p.get("speaker", ""), p["clean"] or p["raw"]) for p in self.paragraphs
            ]
            if self.loose:
                clean_items.append((self.loose[0]["t"], self.loose[0].get("speaker", ""),
                                    " ".join(x["text"] for x in self.loose)))
            return multi, raw_items, clean_items
