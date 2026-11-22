"""Transcript state shared by the recognizer, the cleaner and every UI.

Every change bumps `rev` and calls `emit(kind)` *outside* the lock, so a UI
listener can safely call `snapshot()` from its callback.

Event kinds emitted from here: "partial", "segment", "paragraph", "error".
"""

import json
import threading
from typing import Optional

from .textutil import mark_changes


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
            self.speaker_names = {}  # canonical ID ("Speaker 1") -> user display name
            self.rev += 1
        if notify:
            self.emit("reset")

    def load_saved_session(self, paragraphs: list[dict], report: Optional[dict] = None, notify: bool = True):
        """Populate state from a saved session on disk (safe when idle)."""
        with self.lock:
            self.paragraphs = []
            self.loose = []
            self.partial = ""
            self.error = ""
            self.report = dict(report) if report else None
            self.speaker_names = {}
            self.next_id = 1
            self.next_turn_id = 1
            for item in paragraphs:
                pid = self.next_id
                tid = self.next_turn_id
                self.next_id += 1
                self.next_turn_id += 1
                self.paragraphs.append({
                    "id": pid,
                    "time": item.get("time", ""),
                    "raw": item.get("raw", ""),
                    "clean": item.get("clean", ""),
                    "tokens": item.get("tokens"),
                    "status": "done",
                    "note": "",
                    "speaker": item.get("speaker", ""),
                    "overlap": bool(item.get("overlap", False)),
                    "seg_ids": [tid],
                })
            self.rev += 1
        if notify:
            self.emit("reset")
            self.emit("paragraph", None)

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
                    start_sec: float = 0.0, spk_ready: bool = True,
                    speaker: str = "", gap_before: float = 0.0) -> int:
        import time
        with self.lock:
            tid = self.next_turn_id
            self.next_turn_id += 1
            self.loose.append({
                "id": tid, "t": stamp, "start_sec": start_sec,
                "ts": time.time(), "text": text,
                "speaker": speaker, "spk_ready": spk_ready,
                "gap_before": gap_before, "overlap": False,
            })
            self.partial = ""
            self.rev += 1
        self.emit("segment", dict(info, id=tid, text=text, t=stamp))
        return tid

    # -- diarizer side ----------------------------------------------------
    def apply_diarization(self, seg_id: int, parts: list[tuple[int, str, float, str, str]],
                          mapping: dict[int, str],
                          old_by_spk: Optional[dict[str, list[int]]] = None,
                          id_remap: Optional[dict[str, str]] = None,
                          overlap_ids: Optional[set[int]] = None):
        """Attach speaker labels (and any mid-chunk splits) and propagate converged cluster IDs."""
        import time
        now = time.time()
        ov_set = overlap_ids or set()
        with self.lock:
            segs_dirty = False
            idx = next((i for i, x in enumerate(self.loose) if x["id"] == seg_id), None)
            if idx is not None and parts:
                orig_ts = self.loose[idx]["ts"]
                repl = [
                    {
                        "id": tid, "t": st, "start_sec": t0,
                        "ts": orig_ts if j == 0 else now, "text": txt,
                        "speaker": mapping.get(tid, spk), "spk_ready": True,
                        "overlap": (tid in ov_set),
                    }
                    for j, (tid, st, t0, txt, spk) in enumerate(parts)
                ]
                self.loose[idx:idx + 1] = repl
            elif parts:
                # Cleaner already pulled seg_id into a paragraph before diarization finished
                for pi, p in enumerate(self.paragraphs):
                    sids = p.get("seg_ids", ())
                    if seg_id not in sids:
                        continue
                    segs_meta = p.get("segs")
                    if segs_meta is not None:
                        idx_s = next((si for si, item in enumerate(segs_meta) if item["id"] == seg_id), None)
                        if idx_s is not None:
                            segs_meta[idx_s:idx_s + 1] = [
                                {"id": tid, "t": st, "text": txt, "speaker": mapping.get(tid, spk),
                                 "overlap": (tid in ov_set)}
                                for tid, st, t0, txt, spk in parts
                            ]
                            p["seg_ids"] = [item["id"] for item in segs_meta]
                            p["overlap"] = any(item.get("overlap") for item in segs_meta)
                            segs_dirty = True
                    elif len(sids) == 1 and len(parts) > 1:
                        p["segs"] = [
                            {"id": tid, "t": st, "text": txt, "speaker": mapping.get(tid, spk),
                             "overlap": (tid in ov_set)}
                            for tid, st, t0, txt, spk in parts
                        ]
                        p["seg_ids"] = [item["id"] for item in p["segs"]]
                        p["overlap"] = any(item.get("overlap") for item in p["segs"])
                        segs_dirty = True
                    else:
                        first_tid = parts[0][0]
                        p["speaker"] = mapping.get(first_tid, parts[0][4])
                        if any(tid in ov_set for tid, *_ in parts):
                            p["overlap"] = True
                        if len(parts) > 1:
                            extra = [tid for tid, _, _, _, _ in parts[1:] if tid not in sids]
                            p["seg_ids"] = list(sids) + extra
                    break

            if mapping or segs_dirty or id_remap:
                self._apply_mapping_locked(mapping or {}, id_remap=id_remap)
            if self.speaker_names and old_by_spk and mapping:
                self._remap_speaker_names_locked(old_by_spk, mapping)
            self.rev += 1
        self.emit("speakers", seg_id)

    @staticmethod
    def _spk_sort_key(cid: str) -> tuple[int, int, str]:
        if cid.startswith("Speaker "):
            tail = cid[8:].strip()
            if tail.isdigit():
                return (0, int(tail), cid)
        return (1, 0, cid)

    def _remap_speaker_names_locked(self, old_by_spk: dict[str, list[int]], mapping: dict[int, str]):
        """Keep custom display names aligned with their real clusters across live renumbering."""
        if not self.speaker_names or not old_by_spk or not mapping:
            return
        new_names: dict[str, str] = {}
        tracked = []
        untracked = []
        for old_cid, custom_name in sorted(self.speaker_names.items(), key=lambda kv: self._spk_sort_key(kv[0])):
            if not custom_name:
                continue
            tids = old_by_spk.get(old_cid, ())
            if tids:
                tracked.append((old_cid, custom_name, tids))
            else:
                untracked.append((old_cid, custom_name))

        for old_cid, custom_name, tids in tracked:
            counts: dict[str, int] = {}
            for tid in tids:
                ncid = mapping.get(tid)
                if ncid:
                    counts[ncid] = counts.get(ncid, 0) + 1
            if counts:
                # Persistent IDs: keep custom name anchored to old_cid whenever old_cid still exists
                if counts.get(old_cid, 0) > 0:
                    best_ncid = old_cid
                else:
                    best_ncid = max(counts.items(), key=lambda kv: (kv[1], kv[0] == old_cid))[0]
                if best_ncid not in new_names and custom_name != best_ncid:
                    new_names[best_ncid] = custom_name

        for old_cid, custom_name in untracked:
            if old_cid not in new_names and custom_name != old_cid:
                new_names[old_cid] = custom_name

        self.speaker_names = new_names

    def _apply_mapping_locked(self, mapping: dict[int, str],
                              id_remap: Optional[dict[str, str]] = None):
        remap = id_remap or {}
        for x in self.loose:
            if x["id"] in mapping:
                x["speaker"] = mapping[x["id"]]
            elif remap and x.get("speaker") in remap:
                x["speaker"] = remap[x["speaker"]]
        new_paragraphs = []
        for p in self.paragraphs:
            segs_meta = p.get("segs")
            if segs_meta:
                for item in segs_meta:
                    if item["id"] in mapping:
                        item["speaker"] = mapping[item["id"]]
                    elif remap and item.get("speaker") in remap:
                        item["speaker"] = remap[item["speaker"]]
                # Split into consecutive speaker runs if multiple speakers exist inside this paragraph
                spk_set = {item["speaker"] for item in segs_meta if item.get("speaker")}
                if len(spk_set) > 1:
                    runs = []
                    for item in segs_meta:
                        sp = item.get("speaker", "")
                        if runs and runs[-1]["speaker"] == sp:
                            runs[-1]["items"].append(item)
                        else:
                            runs.append({"speaker": sp, "items": [item]})
                    for ri, run in enumerate(runs):
                        items = run["items"]
                        raw_run = " ".join(it["text"] for it in items)
                        if ri == 0:
                            base = p
                            if base.get("raw") != raw_run:
                                base["clean"] = None
                                base["tokens"] = None
                                base["status"] = "raw"
                        else:
                            base = {
                                "id": self.next_id, "clean": None, "tokens": None, "status": "raw", "note": "",
                            }
                            self.next_id += 1
                        base["time"] = items[0]["t"]
                        base["raw"] = raw_run
                        base["speaker"] = run["speaker"]
                        base["seg_ids"] = [it["id"] for it in items]
                        base["segs"] = items
                        base["overlap"] = any(it.get("overlap") for it in items)
                        new_paragraphs.append(base)
                    continue
                if spk_set:
                    p["speaker"] = next(iter(spk_set))
                elif remap and p.get("speaker") in remap:
                    p["speaker"] = remap[p["speaker"]]
                p["overlap"] = any(it.get("overlap") for it in segs_meta)
            else:
                counts: dict[str, int] = {}
                for tid in p.get("seg_ids", ()):
                    if tid in mapping:
                        sp = mapping[tid]
                        counts[sp] = counts.get(sp, 0) + 1
                if counts:
                    p["speaker"] = max(counts.items(), key=lambda kv: kv[1])[0]
                elif remap and p.get("speaker") in remap:
                    p["speaker"] = remap[p["speaker"]]
            new_paragraphs.append(p)
        self.paragraphs = new_paragraphs

    def speaker_names_copy(self) -> dict[str, str]:
        with self.lock:
            return dict(self.speaker_names)

    def apply_cluster_remap(self, mapping: dict[int, str],
                            new_speaker_names: Optional[dict[str, str]] = None,
                            id_remap: Optional[dict[str, str]] = None) -> bool:
        """Apply a full cluster remap (from merge/dissolve/max-speakers) and updated names."""
        if not mapping and not id_remap and new_speaker_names is None:
            return False
        with self.lock:
            if mapping or id_remap:
                self._apply_mapping_locked(mapping or {}, id_remap=id_remap)
            if new_speaker_names is not None:
                self.speaker_names = {
                    k: v.strip() for k, v in new_speaker_names.items()
                    if k and v and v.strip() and v.strip() != k
                }
            self.rev += 1
        self.emit("speakers", None)
        return True

    def merge_speakers_fallback(self, src_cid: str, dst_cid: str) -> bool:
        """Merge *src_cid* into *dst_cid* directly in State when voiceprints are unavailable."""
        if not src_cid or not dst_cid or src_cid == dst_cid:
            return False
        changed = False
        with self.lock:
            for x in self.loose:
                if x.get("speaker") == src_cid:
                    x["speaker"] = dst_cid
                    changed = True
            for p in self.paragraphs:
                if p.get("speaker") == src_cid:
                    p["speaker"] = dst_cid
                    changed = True
                for item in p.get("segs") or ():
                    if item.get("speaker") == src_cid:
                        item["speaker"] = dst_cid
                        changed = True
            if changed:
                src_name = self.speaker_names.pop(src_cid, "")
                if src_name and dst_cid not in self.speaker_names:
                    self.speaker_names[dst_cid] = src_name
                self.rev += 1
        if changed:
            self.emit("speakers", None)
        return changed

    def item_turn_ids(self, item_kind: str, item_id: int) -> list[int]:
        """Return the diarizer turn IDs belonging to a clicked paragraph (`'p'`) or loose segment (`'l'`)."""
        with self.lock:
            if item_kind == "l":
                return [item_id]
            if item_kind == "p":
                for p in self.paragraphs:
                    if p["id"] == item_id:
                        segs = p.get("segs")
                        if segs:
                            return [it["id"] for it in segs]
                        return list(p.get("seg_ids") or ())
        return []

    def reassign_item_speaker(self, item_kind: str, item_id: int, dst_cid: str) -> bool:
        """Reassign only a single paragraph (`'p'`) or loose segment (`'l'`) to *dst_cid*."""
        if not dst_cid:
            return False
        changed = False
        with self.lock:
            if item_kind == "l":
                for x in self.loose:
                    if x["id"] == item_id:
                        if x.get("speaker") != dst_cid:
                            x["speaker"] = dst_cid
                            changed = True
                        break
                else:
                    # Loose segment may have just been grouped into a paragraph by Cleaner
                    for p in self.paragraphs:
                        if item_id in (p.get("seg_ids") or ()):
                            self._apply_mapping_locked({item_id: dst_cid})
                            changed = True
                            break
            elif item_kind == "p":
                for p in self.paragraphs:
                    if p["id"] == item_id:
                        if p.get("speaker") != dst_cid:
                            p["speaker"] = dst_cid
                            changed = True
                        for it in p.get("segs") or ():
                            if it.get("speaker") != dst_cid:
                                it["speaker"] = dst_cid
                                changed = True
                        break
            if changed:
                self.rev += 1
        if changed:
            self.emit("speakers", None)
        return changed

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
        for p in self.paragraphs:
            if p.get("speaker"):
                return True
        for x in self.loose:
            if x.get("speaker"):
                return True
        return False

    def _disp_spk(self, spk_id: str) -> str:
        return self.speaker_names.get(spk_id, spk_id) if spk_id else ""

    def display_speaker(self, spk_id: str) -> str:
        with self.lock:
            return self._disp_spk(spk_id)

    def _speakers_locked(self) -> list[tuple[str, str]]:
        seen: dict[str, str] = {}
        for item in self.paragraphs:
            cid = item.get("speaker", "")
            if cid and cid not in seen:
                seen[cid] = self._disp_spk(cid)
        for item in self.loose:
            cid = item.get("speaker", "")
            if cid and cid not in seen:
                seen[cid] = self._disp_spk(cid)
        return list(seen.items())

    def speakers(self) -> list[tuple[str, str]]:
        """Return `[(canonical_id, display_name), ...]` in order of first appearance."""
        with self.lock:
            return self._speakers_locked()

    def rename_speakers(self, mapping: dict[str, str]):
        """Update display names `{canonical_id_or_current_name: new_name}` without losing clusters."""
        if not mapping:
            return
        with self.lock:
            cur = self._speakers_locked()
            known = {cid for cid, _ in cur}
            reverse = {disp: cid for cid, disp in cur}
            changed = False
            for key, raw_name in mapping.items():
                cid = key if key in known else reverse.get(key, key)
                if not cid:
                    continue
                new_name = (raw_name or "").strip()
                if not new_name or new_name == cid:
                    if cid in self.speaker_names:
                        del self.speaker_names[cid]
                        changed = True
                elif self.speaker_names.get(cid) != new_name:
                    self.speaker_names[cid] = new_name
                    changed = True
            if not changed:
                return
            self.rev += 1
        self.emit("speakers", None)

    def replace_text(self, old_text: str, new_text: str) -> int:
        """Replace *old_text* with *new_text* across active paragraphs and loose segments."""
        if not old_text or not new_text or old_text == new_text:
            return 0
        changed = 0
        with self.lock:
            for p in self.paragraphs:
                cur = p["clean"] if p.get("clean") else p["raw"]
                if old_text in cur:
                    updated = cur.replace(old_text, new_text)
                    p["clean"] = updated
                    p["tokens"] = mark_changes(p["raw"], updated)
                    p["status"] = "done"
                    changed += 1
            for x in self.loose:
                if old_text in x["text"]:
                    x["text"] = x["text"].replace(old_text, new_text)
                    changed += 1
            if changed:
                self.rev += 1
        if changed:
            self.emit("paragraph", None)
        return changed

    def _build(self) -> dict:
        return {
            "rev": self.rev,
            "partial": self.partial,
            "error": self.error,
            "multi_speaker": self._multi_speaker_locked(),
            "speakers": [{"id": cid, "name": name} for cid, name in self._speakers_locked()],
            "report": dict(self.report) if self.report else None,
            "loose": [
                {"id": x["id"], "t": x["t"], "text": x["text"],
                 "speaker": self._disp_spk(x.get("speaker", "")),
                 "speaker_id": x.get("speaker", ""),
                 "overlap": bool(x.get("overlap", False))}
                for x in self.loose
            ],
            "paragraphs": [
                {"id": p["id"], "time": p["time"], "raw": p["raw"], "tokens": p["tokens"],
                 "status": p["status"], "note": p["note"],
                 "speaker": self._disp_spk(p.get("speaker", "")),
                 "speaker_id": p.get("speaker", ""),
                 "overlap": bool(p.get("overlap", False))}
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
                spk = self._disp_spk(p.get("speaker", ""))
                out.append(f"{spk}: {txt}" if (multi and spk) else txt)
            if self.loose:
                # Group consecutive loose sentences of same speaker
                cur_spk, cur_txt = None, []
                for x in self.loose:
                    spk = self._disp_spk(x.get("speaker", "")) if multi else ""
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

    def export_records(self, include_overlap: bool = False):
        """Return `(multi_speaker, raw_items, clean_items)` with optional trailing overlap flag."""
        with self.lock:
            multi = self._multi_speaker_locked()
            if include_overlap:
                raw_items = [
                    (p["time"], self._disp_spk(p.get("speaker", "")), p["raw"], bool(p.get("overlap")))
                    for p in self.paragraphs
                ]
                raw_items += [
                    (x["t"], self._disp_spk(x.get("speaker", "")), x["text"], bool(x.get("overlap")))
                    for x in self.loose
                ]
                clean_items = [
                    (p["time"], self._disp_spk(p.get("speaker", "")), p["clean"] or p["raw"], bool(p.get("overlap")))
                    for p in self.paragraphs
                ]
                if self.loose:
                    clean_items.append((
                        self.loose[0]["t"],
                        self._disp_spk(self.loose[0].get("speaker", "")),
                        " ".join(x["text"] for x in self.loose),
                        any(x.get("overlap") for x in self.loose),
                    ))
                return multi, raw_items, clean_items
            raw_items = [(p["time"], self._disp_spk(p.get("speaker", "")), p["raw"]) for p in self.paragraphs]
            raw_items += [(x["t"], self._disp_spk(x.get("speaker", "")), x["text"]) for x in self.loose]
            clean_items = [
                (p["time"], self._disp_spk(p.get("speaker", "")), p["clean"] or p["raw"]) for p in self.paragraphs
            ]
            if self.loose:
                clean_items.append((self.loose[0]["t"], self._disp_spk(self.loose[0].get("speaker", "")),
                                    " ".join(x["text"] for x in self.loose)))
            return multi, raw_items, clean_items
