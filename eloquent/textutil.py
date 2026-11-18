"""Text utilities: LLM safety check and change highlighting."""

import difflib
import re
import unicodedata

WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_ARROW_RE = re.compile(r"\s*(?:->|=>|→)\s*")
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


def parse_custom_vocabulary(vocab) -> tuple[list[str], list[tuple[str, str]]]:
    """Parse comma/newline/semicolon-separated vocabulary into standalone terms and ``(wrong, right)`` pairs."""
    if not vocab:
        return [], []
    raw_items = re.split(r"[,;\n،]+", vocab) if isinstance(vocab, str) else list(vocab)
    terms: list[str] = []
    seen_terms: set[str] = set()
    pairs: list[tuple[str, str]] = []
    pair_idx: dict[str, int] = {}

    for item in raw_items:
        s = (item or "").strip()
        if not s:
            continue
        if _ARROW_RE.search(s):
            parts = _ARROW_RE.split(s, maxsplit=1)
            wrong, right = parts[0].strip(), parts[1].strip()
            if wrong and right and wrong.lower() != right.lower():
                key = wrong.lower()
                if key in pair_idx:
                    pairs[pair_idx[key]] = (wrong, right)
                else:
                    pair_idx[key] = len(pairs)
                    pairs.append((wrong, right))
        else:
            key = s.lower()
            if key not in seen_terms:
                seen_terms.add(key)
                terms.append(s)
    return terms, pairs


def format_custom_vocabulary(terms, pairs) -> str:
    """Format standalone *terms* and ``(wrong, right)`` *pairs* back into a normalized settings string."""
    combined = [str(t).strip() for t in (terms or ()) if str(t).strip()]
    combined += [f"{str(w).strip()} -> {str(r).strip()}" for w, r in (pairs or ()) if str(w).strip() and str(r).strip()]
    clean_terms, clean_pairs = parse_custom_vocabulary(combined)
    return ", ".join(clean_terms + [f"{w} -> {r}" for w, r in clean_pairs])


def apply_learned_replacements(text: str, vocab) -> str:
    """Apply exact learned ``wrong -> right`` rules to *text* (useful when LLM cleanup is off)."""
    if not text or not vocab:
        return text
    _, pairs = parse_custom_vocabulary(vocab)
    out = text
    for wrong, right in pairs:
        pattern = re.compile(rf"(?<!\w){re.escape(wrong)}(?!\w)", re.IGNORECASE)
        out = pattern.sub(lambda _m, r=right: r, out)
    return out


def match_custom_vocabulary(raw: str, vocab, max_matches: int = 8) -> tuple[list[str], list[tuple[str, str]]]:
    """Return only custom vocabulary terms and learned pairs phonetically/lexically relevant to *raw*."""
    terms, pairs = parse_custom_vocabulary(vocab)
    if (not terms and not pairs) or not raw or not raw.strip():
        return [], []
    raw_norm = norm_ar_for_sim(strip_diacritics_and_norm(raw))
    raw_words = WORD_RE.findall(raw_norm)
    if not raw_words:
        return [], []

    windows: list[tuple[int, str, str]] = []
    n_raw = len(raw_words)
    for wlen in (1, 2, 3):
        for i in range(max(0, n_raw - wlen + 1)):
            chunk = raw_words[i:i + wlen]
            windows.append((wlen, "".join(chunk), phonetic_skeleton(" ".join(chunk))))

    def _matches(phrase: str) -> bool:
        p_norm = norm_ar_for_sim(strip_diacritics_and_norm(phrase))
        if p_norm and p_norm in raw_norm:
            return True
        p_words = WORD_RE.findall(p_norm)
        if not p_words:
            return False
        t_len = len(p_words)
        t_compact = "".join(p_words)
        t_sk = phonetic_skeleton(phrase)
        for wlen, w_compact, w_sk in windows:
            if wlen > t_len + 1:
                continue
            if difflib.SequenceMatcher(None, t_compact, w_compact, autojunk=False).ratio() >= 0.74:
                return True
            if not t_sk or not w_sk:
                continue
            sk_sim = difflib.SequenceMatcher(None, t_sk, w_sk, autojunk=False).ratio()
            if len(t_sk) >= 4 and (t_sk == w_sk or sk_sim >= 0.75):
                return True
            if len(t_sk) == 3 and (t_sk == w_sk or (sk_sim >= 0.80 and t_sk[0] == w_sk[0])):
                return True
            if len(t_sk) == 2 and t_sk == w_sk and abs(len(w_compact) - len(t_compact)) <= 2 and len(w_compact) >= 4:
                return True
        return False

    matched_pairs = [(w, r) for w, r in pairs if _matches(w) or _matches(r)][:max_matches]
    matched_terms = [t for t in terms if _matches(t)][:max_matches]
    return matched_terms, matched_pairs


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
    word_sim = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
    compact_sim = difflib.SequenceMatcher(None, "".join(a), "".join(b), autojunk=False).ratio()
    sim = max(word_sim, compact_sim * 0.9)
    ratio = len(b) / max(1, len(a))
    if dictation:
        threshold = min(min_sim, 0.25)
        return (sim >= threshold and 0.20 <= ratio <= 1.35), sim
    threshold = min_sim if len(a) >= 12 else min(min_sim, 0.5)
    return (sim >= threshold and 0.45 <= ratio <= 1.3), sim


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
