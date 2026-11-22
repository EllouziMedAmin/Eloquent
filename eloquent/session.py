"""Session files: sessions/<date_time>/ raw.txt, clean.txt, optional audio.wav, and saved reports."""

import re
import shutil
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .llm.prompts import REPORT_FORMATS
from .speech import SAMPLE_RATE
from .textutil import mark_changes

_FOLDER_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})$")
_RE_TS_SPK_OV = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s+(.+?)\s*\[overlap\]:\s*(.*)$")
_RE_TS_SPK = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s+([^:\[\]]{1,40}?):\s*(.*)$")
_RE_TS_OV = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s*\[overlap\]\s*(.*)$")
_RE_TS_ONLY = re.compile(r"^\[(\d+:\d{2}(?::\d{2})?)\]\s+(.*)$")

_REPORT_META_BY_FILE = {f["filename"]: (f["id"], f["label"]) for f in REPORT_FORMATS}
_REPORT_META_BY_FILE["custom_output.md"] = ("custom", "Custom Output")


def _has_arabic_text(text: str) -> bool:
    ar = 0
    latin = 0
    for ch in text[:1200]:
        if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f" or "\u08a0" <= ch <= "\u08ff":
            ar += 1
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            latin += 1
    return ar >= 4 and ar >= latin * 0.35


def format_session_date(folder_name: str, now: Optional[datetime] = None) -> str:
    """Turn ``YYYY-MM-DD_HH-MM-SS`` into a clean short human date label."""
    m = _FOLDER_RE.match(folder_name)
    if not m:
        return folder_name
    try:
        dt = datetime(*(int(g) for g in m.groups()))
    except ValueError:
        return folder_name
    ref = now or datetime.now()
    hhmm = dt.strftime("%H:%M")
    delta_days = (ref.date() - dt.date()).days
    if delta_days == 0:
        return f"Today · {hhmm}"
    if delta_days == 1:
        return f"Yesterday · {hhmm}"
    if dt.year == ref.year:
        return f"{dt.strftime('%b %d').replace(' 0', ' ')} · {hhmm}"
    return f"{dt.strftime('%b %d, %Y').replace(' 0', ' ')} · {hhmm}"


def _parse_block(block: str) -> tuple[str, str, bool, str]:
    """Parse one saved paragraph/line into ``(time, speaker, overlap, text)``."""
    s = block.strip()
    if not s:
        return "", "", False, ""
    m = _RE_TS_SPK_OV.match(s)
    if m:
        return m.group(1), m.group(2).strip(), True, m.group(3).strip()
    m = _RE_TS_SPK.match(s)
    if m:
        return m.group(1), m.group(2).strip(), False, m.group(3).strip()
    m = _RE_TS_OV.match(s)
    if m:
        return m.group(1), "", True, m.group(2).strip()
    m = _RE_TS_ONLY.match(s)
    if m:
        return m.group(1), "", False, m.group(2).strip()
    return "", "", False, s


def _parse_saved_text(text: str, line_based: bool = False) -> list[dict]:
    """Parse ``clean.txt`` (paragraph blocks) or ``raw.txt`` (line-per-turn) into structured entries."""
    chunks = text.splitlines() if line_based else re.split(r"\r?\n\s*\r?\n", text.strip())
    entries = []
    for ch in chunks:
        st, spk, ov, body = _parse_block(ch)
        if not body:
            continue
        entries.append({"time": st, "speaker": spk, "overlap": ov, "text": body})
    return entries


def _list_saved_reports(session_dir: Path) -> list[dict]:
    """Discover saved ``.md`` outputs inside a session folder, newest first."""
    found = []
    try:
        mds = sorted(session_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for p in mds:
        try:
            if p.stat().st_size == 0:
                continue
        except OSError:
            continue
        fid, lbl = _REPORT_META_BY_FILE.get(p.name, ("custom", p.stem.replace("_", " ").title()))
        found.append({"format_id": fid, "label": lbl, "filename": p.name, "path": p})
    return found


def _short_report_tag(label: str) -> str:
    low = label.lower()
    if "summary" in low:
        return "Summary"
    if "bullet" in low:
        return "Bullets"
    if "meeting" in low:
        return "Minutes"
    if "lecture" in low:
        return "Notes"
    if "report" in low:
        return "Report"
    if "memo" in low or "article" in low:
        return "Memo"
    return "Doc"


def list_sessions(base: Path, query: str = "", limit: int = 120) -> list[dict]:
    """Scan *base* for non-empty saved sessions, newest first, with optional search filter."""
    base = Path(base)
    if not base.exists() or not base.is_dir():
        return []
    q = (query or "").strip().lower()
    now = datetime.now()
    try:
        dirs = sorted((d for d in base.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    except OSError:
        return []

    out = []
    for d in dirs:
        clean_p = d / "clean.txt"
        raw_p = d / "raw.txt"
        reports = _list_saved_reports(d)
        clean_size = clean_p.stat().st_size if clean_p.exists() else 0
        raw_size = raw_p.stat().st_size if raw_p.exists() else 0
        if clean_size == 0 and raw_size == 0 and not reports:
            continue

        primary_p = clean_p if clean_size > 0 else raw_p
        head_txt = ""
        if primary_p.exists() and primary_p.stat().st_size > 0:
            try:
                with open(primary_p, "r", encoding="utf-8", errors="ignore") as f:
                    head_txt = f.read(4096)
            except OSError:
                head_txt = ""

        entries = _parse_saved_text(head_txt, line_based=(primary_p == raw_p))
        if not entries and not reports:
            continue

        speakers = []
        seen_spk = set()
        for e in entries:
            sp = e["speaker"]
            if sp and sp not in seen_spk:
                seen_spk.add(sp)
                speakers.append(sp)

        preview_src = " ".join(e["text"] for e in entries[:3]).strip()
        preview = re.sub(r"\s+", " ", preview_src)
        if not preview and reports:
            preview = f"Saved {reports[0]['label']}"

        date_label = format_session_date(d.name, now=now)
        if q:
            hay = f"{d.name} {date_label} {' '.join(speakers)} {preview}".lower()
            if q not in hay:
                # Fall back to searching full file text if not matched in header snippet
                matched = False
                for fp in (clean_p, raw_p):
                    if fp.exists() and fp.stat().st_size > 0:
                        try:
                            if q in fp.read_text(encoding="utf-8", errors="ignore").lower():
                                matched = True
                                break
                        except OSError:
                            pass
                if not matched:
                    continue

        wav_p = d / "audio.wav"
        has_audio = wav_p.exists() and wav_p.stat().st_size > 44
        rep_tag = ""
        if len(reports) == 1:
            rep_tag = _short_report_tag(reports[0]["label"])
        elif len(reports) > 1:
            rep_tag = f"{len(reports)} docs"

        out.append({
            "dir": d,
            "name": d.name,
            "date_label": date_label,
            "preview": preview[:140],
            "speakers": speakers,
            "speaker_count": len(speakers),
            "rtl": _has_arabic_text(preview_src or head_txt),
            "has_audio": has_audio,
            "reports": reports,
            "report_tag": rep_tag,
        })
        if len(out) >= limit:
            break
    return out


def load_session_data(session_dir: Path) -> dict:
    """Load a saved session directory into State-ready paragraphs, RTL flag, and saved reports."""
    d = Path(session_dir)
    clean_p = d / "clean.txt"
    raw_p = d / "raw.txt"
    clean_txt = clean_p.read_text(encoding="utf-8", errors="ignore") if clean_p.exists() else ""
    raw_txt = raw_p.read_text(encoding="utf-8", errors="ignore") if raw_p.exists() else ""

    clean_entries = _parse_saved_text(clean_txt, line_based=False) if clean_txt.strip() else []
    raw_entries = _parse_saved_text(raw_txt, line_based=True) if raw_txt.strip() else []

    paragraphs = []
    if clean_entries:
        same_len = (len(raw_entries) == len(clean_entries))
        # Build timestamp lookup when raw has more granular lines than clean
        raw_by_ts: dict[str, list[str]] = {}
        if not same_len and raw_entries:
            for re_it in raw_entries:
                if re_it["time"]:
                    raw_by_ts.setdefault(re_it["time"], []).append(re_it["text"])
        for i, ce in enumerate(clean_entries):
            c_body = ce["text"]
            if same_len:
                r_body = raw_entries[i]["text"]
            elif ce["time"] and ce["time"] in raw_by_ts:
                r_body = " ".join(raw_by_ts[ce["time"]])
            else:
                r_body = c_body
            tokens = mark_changes(r_body, c_body) if r_body != c_body else [(w, False) for w in c_body.split()]
            paragraphs.append({
                "time": ce["time"],
                "speaker": ce["speaker"],
                "overlap": ce["overlap"],
                "raw": r_body,
                "clean": c_body,
                "tokens": tokens,
            })
    elif raw_entries:
        for re_it in raw_entries:
            body = re_it["text"]
            paragraphs.append({
                "time": re_it["time"],
                "speaker": re_it["speaker"],
                "overlap": re_it["overlap"],
                "raw": body,
                "clean": body,
                "tokens": None,
            })

    reports = _list_saved_reports(d)
    loaded_report = None
    if reports:
        first = reports[0]
        try:
            rep_text = first["path"].read_text(encoding="utf-8", errors="ignore").strip()
            if rep_text:
                loaded_report = {
                    "format_id": first["format_id"],
                    "label": first["label"],
                    "status": "done",
                    "text": rep_text,
                    "file": str(first["path"]),
                    "error": "",
                }
        except OSError:
            loaded_report = None

    sample_text = " ".join(p["clean"] or p["raw"] for p in paragraphs[:6])
    return {
        "dir": d,
        "name": d.name,
        "date_label": format_session_date(d.name),
        "paragraphs": paragraphs,
        "rtl": _has_arabic_text(sample_text),
        "reports": reports,
        "report": loaded_report,
    }


def delete_session(session_dir: Path, sessions_root: Path) -> bool:
    """Safely delete a session folder strictly inside *sessions_root*."""
    try:
        target = Path(session_dir).resolve()
        root = Path(sessions_root).resolve()
        if target == root or target.parent != root or not target.is_dir():
            return False
        shutil.rmtree(target)
        return True
    except OSError:
        return False


def _format_session_texts(state) -> tuple[str, str]:
    multi, raw_items, clean_items = state.export_records(include_overlap=True)
    raw_lines = []
    for stamp, spk, txt, ov in raw_items:
        ov_tag = " [overlap]" if ov else ""
        prefix = f"[{stamp}] {spk}{ov_tag}:" if (multi and spk) else f"[{stamp}]{ov_tag}"
        raw_lines.append(f"{prefix} {txt}\n")
    clean_lines = []
    for stamp, spk, txt, ov in clean_items:
        ov_tag = " [overlap]" if ov else ""
        if multi and spk:
            clean_lines.append(f"[{stamp}] {spk}{ov_tag}: {txt.strip()}\n\n")
        elif ov:
            clean_lines.append(f"[{stamp}] [overlap] {txt.strip()}\n\n")
        else:
            clean_lines.append(txt.strip() + "\n\n")
    return "".join(raw_lines), "".join(clean_lines)


def write_session_files(session_dir: Path, state) -> None:
    """Rewrite closed session raw.txt and clean.txt on disk (e.g. after speaker rename)."""
    if not session_dir.exists():
        return
    raw_body, clean_body = _format_session_texts(state)
    (session_dir / "raw.txt").write_text(raw_body, encoding="utf-8")
    (session_dir / "clean.txt").write_text(clean_body, encoding="utf-8")


class Session:
    def __init__(self, base: Path, save_audio: bool):
        self.dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.raw_f = open(self.dir / "raw.txt", "w", encoding="utf-8")
        self.clean_f = open(self.dir / "clean.txt", "w", encoding="utf-8")
        self.wav = None
        if save_audio:
            self.wav = wave.open(str(self.dir / "audio.wav"), "wb")
            self.wav.setnchannels(1)
            self.wav.setsampwidth(2)
            self.wav.setframerate(SAMPLE_RATE)

    def add_raw(self, stamp: str, text: str):
        with self._lock:
            if not self.raw_f.closed:
                self.raw_f.write(f"[{stamp}] {text}\n")
                self.raw_f.flush()

    def add_clean(self, text: str):
        with self._lock:
            if not self.clean_f.closed:
                self.clean_f.write(text.strip() + "\n\n")
                self.clean_f.flush()

    def sync_from_state(self, state):
        """Rewrite raw.txt and clean.txt with finalized multi-speaker splits and labels."""
        raw_body, clean_body = _format_session_texts(state)
        with self._lock:
            if not self.raw_f.closed:
                self.raw_f.seek(0)
                self.raw_f.truncate()
                self.raw_f.write(raw_body)
                self.raw_f.flush()
            if not self.clean_f.closed:
                self.clean_f.seek(0)
                self.clean_f.truncate()
                self.clean_f.write(clean_body)
                self.clean_f.flush()

    def write_audio(self, frame: np.ndarray):
        if self.wav is not None:
            self.wav.writeframes((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())

    def close(self):
        with self._lock:
            for f in (self.raw_f, self.clean_f):
                f.close()
            if self.wav is not None:
                self.wav.close()
                self.wav = None
