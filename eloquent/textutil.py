"""Text utilities: LLM safety check and change highlighting."""

import difflib
import re

WORD_RE = re.compile(r"[\w']+", re.UNICODE)


def norm_words(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]


def accept_cleaned(raw: str, clean: str, min_sim: float):
    """Reject the model's output if it rewrote, shortened or padded the text."""
    a, b = norm_words(raw), norm_words(clean)
    if not b:
        return False, 0.0
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    ratio = len(b) / max(1, len(a))
    threshold = min_sim if len(a) >= 12 else min(min_sim, 0.5)
    return (sim >= threshold and 0.5 <= ratio <= 1.3), sim


def mark_changes(raw: str, clean: str):
    """[[word, changed?], ...] for the cleaned text (punctuation-only edits are not flagged)."""
    def key(w):
        return re.sub(r"[^\w']", "", w.lower())

    rt, ct = raw.split(), clean.split()
    rk, ck = [key(w) for w in rt], [key(w) for w in ct]
    flags = [0] * len(ct)
    for op, _, _, j1, j2 in difflib.SequenceMatcher(None, rk, ck, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            for j in range(j1, j2):
                if ck[j]:
                    flags[j] = 1
    return [[w, f] for w, f in zip(ct, flags)]


def strip_wrappers(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'“”":
        t = t[1:-1].strip()
    return t


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.?!…]['\"”)]*$", text.strip()))
