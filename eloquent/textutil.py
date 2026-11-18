"""Text utilities: LLM safety check and change highlighting."""

import difflib
import re
import unicodedata

WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_HARAKAT = frozenset(chr(c) for c in range(0x064B, 0x0660)) | {"\u0670", "\u0640"}
_PHONETIC_CLUSTERS = (
    ("qu", "k"), ("ch", "s"), ("sh", "s"), ("ph", "f"), ("th", "t"),
    ("gh", "g"), ("kh", "h"), ("dj", "j"), ("ou", "w"), ("eau", "o"),
    ("tion", "sn"), ("x", "ks"),
)
_PHONETIC_MAP = {
    "ب": "b", "پ": "b", "ت": "t", "ط": "t", "ة": "", "د": "d", "ض": "d",
    "ك": "k", "ق": "k", "گ": "k", "ف": "f", "ڤ": "f", "س": "s", "ص": "s",
    "ث": "t", "ز": "z", "ذ": "z", "ظ": "z", "ش": "s", "ج": "j", "ل": "l",
    "م": "m", "ن": "n", "ر": "r", "ه": "h", "ح": "h", "خ": "h", "غ": "g",
    "و": "", "ؤ": "", "ي": "", "ى": "", "ئ": "", "ا": "", "أ": "", "إ": "", "آ": "", "ع": "", "ء": "",
    "b": "b", "p": "b", "t": "t", "d": "d", "k": "k", "c": "k", "q": "k",
    "f": "f", "v": "f", "s": "s", "ç": "s", "z": "z", "j": "j", "g": "k",
    "l": "l", "m": "m", "n": "n", "r": "r", "h": "h", "w": "",
    "a": "", "e": "", "i": "", "o": "", "u": "", "y": "",
}


def strip_diacritics_and_norm(s: str) -> str:
    """Lowercase, fold French ligatures/cedilla, and strip combining diacritics/tatweel."""
    s = s.lower().replace("ç", "s").replace("œ", "oe").replace("æ", "ae")
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn" and c != "\u0640")


def norm_ar_for_sim(text: str) -> str:
    """Normalize Arabic diacritics and common hamza/alef-maksura variants before comparison."""
    out = "".join(c for c in text if c not in _HARAKAT)
    return out.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))


def phonetic_skeleton(text: str) -> str:
    """Map Arabic and Latin words to a shared consonant skeleton so bilingual restorations align."""
    s = strip_diacritics_and_norm(text)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = re.sub(r"g(?=[eiy])", "j", s)
    for src, dst in _PHONETIC_CLUSTERS:
        s = s.replace(src, dst)
    out: list[str] = []
    for ch in s:
        if ch in _PHONETIC_MAP:
            mapped = _PHONETIC_MAP[ch]
            if mapped and (not out or out[-1] != mapped):
                out.append(mapped)
        elif ch.isdigit():
            out.append(ch)
    return "".join(out)


def accept_bilingual_fix(raw: str, clean: str, min_sim: float, context: str = "",
                         dictation: bool = False) -> tuple[bool, float]:
    """Allow meaning-based word fixes and French/English term restorations while blocking rewrites and hallucinations."""
    rw = WORD_RE.findall(norm_ar_for_sim(strip_diacritics_and_norm(raw)))
    cw = WORD_RE.findall(norm_ar_for_sim(strip_diacritics_and_norm(clean)))
    if not cw:
        return False, 0.0
    nr, nc = len(rw), len(cw)
    if dictation:
        if nc < max(1, int(nr * 0.20)) or (nr > 4 and nc > int(nr * 1.55) + 2):
            return False, 0.0
        if nr <= 4 and (nc > max(7, nr * 2 + 2) or len(clean) > max(75, len(raw) * 3 + 15)):
            return False, 0.0
    elif nr <= 4:
        if nc > max(6, nr * 2 + 1) or len(clean) > max(55, len(raw) * 2 + 20):
            return False, 0.0
    elif nc < max(1, int(nr * 0.60)) or nc > int(nr * 1.45) + 2:
        return False, 0.0

    if context and nc > nr:
        ctx_w = WORD_RE.findall(norm_ar_for_sim(strip_diacritics_and_norm(context)))
        span = min(len(ctx_w), max(2, nc - nr))
        head = cw[:span]
        if len(head) >= 2 and any(ctx_w[i:i + len(head)] == head for i in range(max(0, len(ctx_w) - len(head) + 1))):
            return False, 0.0

    lex_sim = difflib.SequenceMatcher(None, rw, cw, autojunk=False).ratio()
    sk_raw, sk_clean = phonetic_skeleton(raw), phonetic_skeleton(clean)
    ph_sim = difflib.SequenceMatcher(None, sk_raw, sk_clean, autojunk=False).ratio() if (sk_raw or sk_clean) else lex_sim
    sim = max(lex_sim, ph_sim)

    if not dictation and nr >= 4:
        # Ensure the LLM kept most of the sentence's words (allowing 1-2 meaning fixes in the middle)
        raw_sks = [phonetic_skeleton(w) for w in rw]
        anchored = 0
        for w, c_sk in zip(cw, (phonetic_skeleton(x) for x in cw)):
            if w in rw or (c_sk and any(
                c_sk == r_sk or difflib.SequenceMatcher(None, c_sk, r_sk, autojunk=False).ratio() >= 0.6
                for r_sk in raw_sks if r_sk
            )):
                anchored += 1
        if anchored / max(1, len(cw)) < 0.55:
            return False, sim

    if dictation:
        eff_thresh = min(min_sim, 0.30)
    else:
        eff_thresh = min(min_sim, 0.45) if nr <= 3 else max(min_sim, 0.52)
    return sim >= eff_thresh, sim


def norm_words(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]


def accept_cleaned(raw: str, clean: str, min_sim: float, dictation: bool = False):
    """Reject the model's output if it rewrote, shortened or padded the text."""
    a, b = norm_words(raw), norm_words(clean)
    if not b:
        return False, 0.0
    sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    ratio = len(b) / max(1, len(a))
    if dictation:
        threshold = min(min_sim, 0.25)
        return (sim >= threshold and 0.20 <= ratio <= 1.35), sim
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
    return bool(re.search(r"[.?!؟…]['\"”)]*$", text.strip()))
