"""Offline Markdown + LaTeX math renderer for Eloquent AI reports.

Extracts LaTeX math spans before CommonMark parsing so underscores, asterisks,
and backslashes inside formulas are never mangled, parses formulas into a small
AST, and renders both:
  - Qt QTextDocument HTML + crisp 2x HiDPI QPainter images for `widget.py`
  - Semantic HTML5 + native MathML Core (<math>) for `web.py`
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import html
import math as _pymath
import re
from typing import Any


# ---------------------------------------------------------------------------
# Symbol & command tables
# ---------------------------------------------------------------------------

GREEK_LOWER: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ϵ",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν",
    "xi": "ξ", "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ",
    "sigma": "σ", "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "ϕ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
}

GREEK_UPPER: dict[str, str] = {
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
}

BIN_OPS: dict[str, str] = {
    "pm": "±", "mp": "∓", "times": "×", "div": "÷", "cdot": "·", "cdotp": "·",
    "ast": "∗", "star": "⋆", "circ": "∘", "bullet": "•",
    "oplus": "⊕", "ominus": "⊖", "otimes": "⊗", "oslash": "⊘", "odot": "⊙",
    "cup": "∪", "cap": "∩", "setminus": "∖", "smallsetminus": "∖",
    "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨",
    "uplus": "⊎", "sqcup": "⊔", "sqcap": "⊓", "amalg": "⨿",
}

REL_OPS: dict[str, str] = {
    "le": "≤", "leq": "≤", "leqslant": "⩽", "ge": "≥", "geq": "≥",
    "geqslant": "⩾", "ne": "≠", "neq": "≠", "equiv": "≡", "approx": "≈",
    "sim": "∼", "simeq": "≃", "cong": "≅", "propto": "∝",
    "ll": "≪", "gg": "≫", "subset": "⊂", "supset": "⊃",
    "subseteq": "⊆", "supseteq": "⊇", "subsetneq": "⊊", "supsetneq": "⊋",
    "in": "∈", "notin": "∉", "ni": "∋", "owns": "∋",
    "perp": "⊥", "parallel": "∥", "mid": "∣", "nmid": "∤",
    "vdash": "⊢", "dashv": "⊣", "models": "⊨", "triangleq": "≜",
    "asymp": "≍", "doteq": "≐", "prec": "≺", "succ": "≻",
    "preceq": "⪯", "succeq": "⪰",
    "to": "→", "rightarrow": "→", "gets": "←", "leftarrow": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "Leftrightarrow": "⇔", "iff": "⟺", "implies": "⟹", "impliedby": "⟸",
    "mapsto": "↦", "longmapsto": "⟼", "longrightarrow": "⟶",
    "longleftarrow": "⟵", "longleftrightarrow": "⟷",
    "hookrightarrow": "↪", "hookleftarrow": "↩",
    "uparrow": "↑", "downarrow": "↓", "updownarrow": "↕",
    "Uparrow": "⇑", "Downarrow": "⇓", "nearrow": "↗", "searrow": "↘",
}

BIG_OPS: dict[str, tuple[str, bool]] = {
    "sum": ("∑", True), "prod": ("∏", True), "coprod": ("∐", True),
    "int": ("∫", False), "iint": ("∬", False), "iiint": ("∭", False),
    "oint": ("∮", False), "oiint": ("∯", False),
    "bigcup": ("⋃", True), "bigcap": ("⋂", True),
    "bigoplus": ("⨁", True), "bigotimes": ("⨂", True),
    "bigvee": ("⋁", True), "bigwedge": ("⋀", True),
    "biguplus": ("⨄", True), "bigsqcup": ("⨆", True),
}

LIMIT_FUNCS: set[str] = {
    "lim", "limsup", "liminf", "sup", "inf", "max", "min",
    "argmax", "argmin", "det", "gcd", "Pr",
}

PLAIN_FUNCS: set[str] = {
    "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth",
    "ln", "log", "lg", "exp", "dim", "ker", "deg", "hom", "rank",
    "tr", "diag", "sgn", "arg", "mod", "bmod", "projlim", "injlim",
}

MISC_SYMS: dict[str, str] = {
    "infty": "∞", "partial": "∂", "nabla": "∇",
    "forall": "∀", "exists": "∃", "nexists": "∄",
    "neg": "¬", "lnot": "¬", "emptyset": "∅", "varnothing": "∅",
    "aleph": "ℵ", "hbar": "ℏ", "ell": "ℓ", "imath": "ı", "jmath": "ȷ",
    "Re": "ℜ", "Im": "ℑ", "wp": "℘", "top": "⊤", "bot": "⊥",
    "angle": "∠", "measuredangle": "∡", "triangle": "△",
    "triangledown": "▽", "square": "□", "Box": "□", "diamond": "◇",
    "cdots": "⋯", "ldots": "…", "dots": "…", "vdots": "⋮", "ddots": "⋱",
    "because": "∵", "therefore": "∴", "prime": "′", "degree": "°",
    "surd": "√", "checkmark": "✓", "colon": ":",
    "%": "%", "$": "$", "#": "#", "_": "_", "&": "&",
    "backslash": "\\",
}

DELIM_NAMES: dict[str, str] = {
    "(": "(", ")": ")", "[": "[", "]": "]",
    "{": "{", "}": "}", "lbrace": "{", "rbrace": "}",
    "|": "|", "vert": "|", "lvert": "|", "rvert": "|",
    "Vert": "‖", "lVert": "‖", "rVert": "‖", "parallel": "‖",
    "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "/": "/", "backslash": "\\", ".": "",
}

ACCENTS: dict[str, str] = {
    "vec": "→", "overrightarrow": "→", "overleftarrow": "←",
    "hat": "^", "widehat": "^", "tilde": "~", "widetilde": "~",
    "bar": "‾", "overline": "‾", "underline": "_",
    "dot": "˙", "ddot": "¨", "check": "ˇ", "breve": "˘",
}

NEGATIONS: dict[str, str] = {
    "=": "≠", "<": "≮", ">": "≯",
    "\\in": "∉", "\\subset": "⊄", "\\supset": "⊅",
    "\\subseteq": "⊈", "\\supseteq": "⊉",
    "\\equiv": "≢", "\\approx": "≉", "\\sim": "≁", "\\cong": "≇",
    "\\le": "≰", "\\leq": "≰", "\\ge": "≱", "\\geq": "≱",
    "\\parallel": "∦", "\\mid": "∤",
}

BLACKBOARD_CHARS: dict[str, str] = {
    "A": "𝔸", "B": "𝔹", "C": "ℂ", "D": "𝔻", "E": "𝔼", "F": "𝔽", "G": "𝔾",
    "H": "ℍ", "I": "𝕀", "J": "𝕁", "K": "𝕂", "L": "𝕃", "M": "𝕄", "N": "ℕ",
    "O": "𝕆", "P": "ℙ", "Q": "ℚ", "R": "ℝ", "S": "𝕊", "T": "𝕋", "U": "𝕌",
    "V": "𝕍", "W": "𝕎", "X": "𝕏", "Y": "𝕐", "Z": "ℤ", "1": "𝟙", "0": "𝟘",
}

SCRIPT_CHARS: dict[str, str] = {
    "A": "𝒜", "B": "ℬ", "C": "𝒞", "D": "𝒟", "E": "ℰ", "F": "ℱ", "G": "𝒢",
    "H": "ℋ", "I": "ℐ", "J": "𝒥", "K": "𝒦", "L": "ℒ", "M": "ℳ", "N": "𝒩",
    "O": "𝒪", "P": "𝒫", "Q": "𝒬", "R": "ℛ", "S": "𝒮", "T": "𝒯", "U": "𝒰",
    "V": "𝒱", "W": "𝒲", "X": "𝒳", "Y": "𝒴", "Z": "𝒵",
}

SPACES_EM: dict[str, float] = {
    "!": -0.16, ",": 0.17, ":": 0.22, ";": 0.28, " ": 0.30, "~": 0.30,
    "thinspace": 0.17, "medspace": 0.22, "thickspace": 0.28,
    "quad": 1.0, "qquad": 2.0, "enspace": 0.5, "kern": 0.25,
}

STYLE_CMDS: set[str] = {
    "displaystyle", "textstyle", "scriptstyle", "scriptscriptstyle",
    "limits", "nolimits", "nonumber", "notag",
}

SIZE_CMDS: set[str] = {
    "big", "Big", "bigg", "Bigg",
    "bigl", "Bigl", "biggl", "Biggl",
    "bigr", "Bigr", "biggr", "Biggr",
    "bigm", "Bigm", "biggm", "Biggm",
}

TEXT_STYLE_CMDS: dict[str, str] = {
    "text": "text", "textrm": "text", "textup": "text", "textnormal": "text",
    "hbox": "text", "mbox": "text", "mathrm": "rm", "operatorname": "rm",
    "mathbf": "bf", "textbf": "bf", "boldsymbol": "bf", "bm": "bf",
    "mathit": "math", "textit": "it", "mathbb": "bb", "mathcal": "cal",
    "mathsf": "rm", "mathtt": "rm", "mathfrak": "rm",
}

STANDALONE_ENVS: set[str] = {
    "equation", "equation*", "align", "align*", "gather", "gather*",
    "multline", "multline*", "cases", "pmatrix", "bmatrix", "Bmatrix",
    "vmatrix", "Vmatrix", "matrix",
}

MATRIX_DELIMS: dict[str, tuple[str, str]] = {
    "matrix": ("", ""), "smallmatrix": ("", ""),
    "pmatrix": ("(", ")"), "bmatrix": ("[", "]"),
    "Bmatrix": ("{", "}"), "vmatrix": ("|", "|"), "Vmatrix": ("‖", "‖"),
    "cases": ("{", ""),
}


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    items: tuple[Any, ...]


@dataclass(frozen=True)
class Atom:
    val: str
    style: str = "math"  # math | num | rm | bf | bb | cal | text | it


@dataclass(frozen=True)
class Sym:
    unicode: str
    kind: str = "ord"    # ord | op | bin | rel | open | close | punct | bigop | lim


@dataclass(frozen=True)
class Frac:
    num: Any
    den: Any
    line: bool = True


@dataclass(frozen=True)
class Sqrt:
    rad: Any
    index: Any = None


@dataclass(frozen=True)
class Scripts:
    base: Any
    sub: Any = None
    sup: Any = None
    limits: bool = False


@dataclass(frozen=True)
class Accent:
    base: Any
    acc: str


@dataclass(frozen=True)
class Delim:
    left: str
    body: Any
    right: str


@dataclass(frozen=True)
class Table:
    rows: tuple[tuple[Any, ...], ...]
    kind: str = "matrix"


@dataclass(frozen=True)
class Space:
    em: float


def _as_row(items: list[Any]) -> Any:
    if not items:
        return Row(())
    if len(items) == 1:
        return items[0]
    return Row(tuple(items))


# ---------------------------------------------------------------------------
# Math extraction from Markdown (before markdown-it runs)
# ---------------------------------------------------------------------------

def _is_line_start(src: str, pos: int) -> bool:
    return pos == 0 or src[pos - 1] == "\n"


def _find_closing_fence(src: str, pos: int, ch: str, length: int) -> int:
    while pos < len(src):
        nl = src.find("\n", pos)
        line_end = len(src) if nl == -1 else nl + 1
        stripped = src[pos:line_end].rstrip("\r\n").lstrip(" ")
        if stripped.startswith(ch * length) and not stripped.strip(ch):
            return line_end
        pos = line_end
    return len(src)


def _find_closing_math(src: str, start: int, close_tok: str, allow_blank_lines: bool) -> int:
    """Find matching unescaped close delimiter; return start index of close_tok or -1."""
    n = len(src)
    i = start
    while i < n:
        if src.startswith(close_tok, i):
            return i
        if src[i] == "\\":
            i += 2
            continue
        if not allow_blank_lines and src[i] == "\n" and i + 1 < n and src[i + 1] == "\n":
            return -1
        i += 1
    return -1


_MATH_SIGNAL_RE = re.compile(r"[\\^_=<>+*/|{}]")


def _find_inline_dollar_end(src: str, start: int) -> int:
    """Find closing `$` for guarded inline math; return index of `$` or -1."""
    n = len(src)
    if start >= n or src[start] in " \t\r\n$`":
        return -1
    i = start
    while i < n:
        ch = src[i]
        if ch == "`":
            return -1
        if ch == "\\":
            i += 2
            continue
        if ch == "\n":
            j = i + 1
            while j < n and src[j] in " \t":
                j += 1
            if j < n and src[j] == "\n":
                return -1
            i += 1
            continue
        if ch == "$":
            if i + 1 < n and src[i + 1] == "$":
                return -1
            if src[i - 1] in " \t\r\n":
                return -1
            if i + 1 < n and src[i + 1].isdigit():
                return -1
            cand = src[start:i]
            # Reject currency-like prose such as "$100 (or $200)" or long prose spans
            if cand[0].isdigit() and " " in cand and not _MATH_SIGNAL_RE.search(cand):
                return -1
            if len(cand) > 160 and "\\" not in cand and "=" not in cand:
                return -1
            return i
        i += 1
    return -1


def _normalize_standalone_env(env: str, raw: str) -> str:
    if env in ("equation", "equation*", "multline", "multline*"):
        inner = re.sub(rf"^\\begin\{{{re.escape(env)}\}}", "", raw)
        return re.sub(rf"\\end\{{{re.escape(env)}\}}$", "", inner).strip()
    if env in ("align", "align*"):
        return raw.replace(f"{{{env}}}", "{aligned}")
    if env in ("gather", "gather*"):
        return raw.replace(f"{{{env}}}", "{gathered}")
    return raw


_ENV_OPEN_RE = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")


def _extract_math(src: str) -> tuple[str, dict[str, tuple[str, bool]]]:
    """Mask LaTeX math spans with alphanumeric tokens safe from Markdown parsing."""
    tokens: dict[str, tuple[str, bool]] = {}
    out: list[str] = []
    n = len(src)
    i = 0

    def add_tok(latex: str, display: bool) -> str:
        tag = f"XELQMATH{'D' if display else 'I'}{len(tokens)}X"
        tokens[tag] = (latex.strip(), display)
        return tag

    while i < n:
        # 1. Fenced code block (``` or ~~~)
        if _is_line_start(src, i):
            m_fence = re.match(r" {0,3}(`{3,}|~{3,})[^\n]*(?:\n|$)", src[i:])
            if m_fence:
                ch = m_fence.group(1)[0]
                flen = len(m_fence.group(1))
                end = _find_closing_fence(src, i + m_fence.end(), ch, flen)
                out.append(src[i:end])
                i = end
                continue

        ch = src[i]
        # 2. Inline code span (`...`)
        if ch == "`":
            k = 0
            while i + k < n and src[i + k] == "`":
                k += 1
            j = i + k
            found = -1
            while j < n:
                if src[j] == "\n" and j + 1 < n and src[j + 1] == "\n":
                    break
                if src[j] == "`":
                    m = 0
                    while j + m < n and src[j + m] == "`":
                        m += 1
                    if m == k:
                        found = j + m
                        break
                    j += m
                    continue
                j += 1
            if found != -1:
                out.append(src[i:found])
                i = found
            else:
                out.append("`" * k)
                i += k
            continue

        # 3. Display \[ ... \] and inline \( ... \)
        if src.startswith("\\[", i):
            end = _find_closing_math(src, i + 2, "\\]", allow_blank_lines=True)
            if end != -1:
                out.append(add_tok(src[i + 2:end], True))
                i = end + 2
                continue
        if src.startswith("\\(", i):
            end = _find_closing_math(src, i + 2, "\\)", allow_blank_lines=False)
            if end != -1:
                out.append(add_tok(src[i + 2:end], False))
                i = end + 2
                continue

        # 4. Standalone \begin{env}...\end{env}
        if src.startswith("\\begin{", i):
            m_env = _ENV_OPEN_RE.match(src, i)
            if m_env and m_env.group(1) in STANDALONE_ENVS:
                env = m_env.group(1)
                close_tag = f"\\end{{{env}}}"
                end = src.find(close_tag, m_env.end())
                if end != -1:
                    raw = src[i:end + len(close_tag)]
                    out.append(add_tok(_normalize_standalone_env(env, raw), True))
                    i = end + len(close_tag)
                    continue

        # 5. Other backslash escape (\$100, \\, etc.)
        if ch == "\\":
            out.append(src[i:i + 2])
            i += 2
            continue

        # 6. Display $$...$$ and guarded inline $...$
        if src.startswith("$$", i):
            end = _find_closing_math(src, i + 2, "$$", allow_blank_lines=True)
            if end != -1:
                out.append(add_tok(src[i + 2:end], True))
                i = end + 2
                continue
        if ch == "$":
            end = _find_inline_dollar_end(src, i + 1)
            if end != -1:
                out.append(add_tok(src[i + 1:end], False))
                i = end + 1
                continue

        out.append(ch)
        i += 1

    return "".join(out), tokens


# ---------------------------------------------------------------------------
# LaTeX math lexer + recursive-descent parser
# ---------------------------------------------------------------------------

def _tokenize_math(src: str) -> list[tuple[str, str]]:
    """Tokenize LaTeX math into (kind, value) pairs."""
    toks: list[tuple[str, str]] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in " \t\r\n":
            i += 1
            continue
        if c == "%":
            nl = src.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if c == "\\":
            if i + 1 >= n:
                break
            nxt = src[i + 1]
            if nxt == "\\":
                toks.append(("cmd", "\\\\"))
                i += 2
            elif nxt.isalpha():
                j = i + 2
                while j < n and src[j].isalpha():
                    j += 1
                name = src[i + 1:j]
                if j < n and src[j] == "*" and name in ("begin", "end", "operatorname"):
                    name += "*"
                    j += 1
                toks.append(("cmd", name))
                i = j
            else:
                toks.append(("cmd", nxt))
                i += 2
            continue
        if c in "{}[]^_&'~":
            toks.append(("spec", c))
            i += 1
            continue
        if c.isdigit():
            j = i + 1
            while j < n and (src[j].isdigit() or (src[j] == "." and j + 1 < n and src[j + 1].isdigit())):
                j += 1
            toks.append(("num", src[i:j]))
            i = j
            continue
        if c.isalpha():
            toks.append(("id", c))
            i += 1
            continue
        toks.append(("char", c))
        i += 1
    return toks


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.toks = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> tuple[str, str] | None:
        idx = self.pos + offset
        return self.toks[idx] if 0 <= idx < len(self.toks) else None

    def pop(self) -> tuple[str, str]:
        tok = self.toks[self.pos]
        self.pos += 1
        return tok

    def parse_all(self, stop_cmds: set[str] | None = None) -> Any:
        items: list[Any] = []
        stop_cmds = stop_cmds or set()
        while self.pos < len(self.toks):
            k, v = self.toks[self.pos]
            if k == "spec" and v in ("}", "&"):
                break
            if k == "cmd" and v in stop_cmds:
                break
            if k == "cmd" and v in ("\\\\", "hline", "cr"):
                break
            items.append(self._parse_factor())
        return _as_row(items)

    def _consume_braced_raw(self) -> str:
        """Consume `{...}` as plain raw string (used for env names, tags, text)."""
        tok = self.peek()
        if not tok:
            return ""
        if tok != ("spec", "{"):
            self.pop()
            return tok[1]
        self.pop()
        depth = 1
        parts: list[str] = []
        while self.pos < len(self.toks) and depth > 0:
            k, v = self.pop()
            if k == "spec" and v == "{":
                depth += 1
                parts.append("{")
            elif k == "spec" and v == "}":
                depth -= 1
                if depth > 0:
                    parts.append("}")
            elif k == "cmd":
                parts.append(v if v in ("%", "$", "#", "_", "&", "{", "}") else f"\\{v}")
            else:
                parts.append(v)
        return "".join(parts)

    def _parse_arg(self) -> Any:
        tok = self.peek()
        if not tok:
            return Row(())
        if tok == ("spec", "{"):
            self.pop()
            inner = self.parse_all()
            if self.peek() == ("spec", "}"):
                self.pop()
            return inner
        return self._parse_atom()

    def _parse_opt_bracket(self) -> Any | None:
        if self.peek() != ("spec", "["):
            return None
        self.pop()
        items: list[Any] = []
        while self.pos < len(self.toks):
            if self.peek() == ("spec", "]"):
                self.pop()
                break
            if self.peek() == ("spec", "}"):
                break
            items.append(self._parse_factor())
        return _as_row(items)

    def _parse_factor(self) -> Any:
        base = self._parse_atom()
        sub = None
        sup = None
        explicit_limits: bool | None = None
        while self.pos < len(self.toks):
            tok = self.peek()
            if tok == ("cmd", "limits"):
                self.pop()
                explicit_limits = True
                continue
            if tok == ("cmd", "nolimits"):
                self.pop()
                explicit_limits = False
                continue
            if tok == ("spec", "'"):
                primes = ""
                while self.peek() == ("spec", "'"):
                    self.pop()
                    primes += "′"
                p_node = Sym(primes, "ord")
                sup = p_node if sup is None else Row((sup, p_node))
                continue
            if tok == ("spec", "_"):
                self.pop()
                arg = self._parse_arg()
                sub = arg if sub is None else Row((sub, arg))
                continue
            if tok == ("spec", "^"):
                self.pop()
                arg = self._parse_arg()
                sup = arg if sup is None else Row((sup, arg))
                continue
            break
        if sub is None and sup is None:
            return base
        use_lim = isinstance(base, Sym) and base.kind in ("bigop", "lim")
        if explicit_limits is not None:
            use_lim = explicit_limits
        elif isinstance(base, Sym) and base.unicode in ("∫", "∬", "∭", "∮", "∯"):
            use_lim = False
        return Scripts(base, sub=sub, sup=sup, limits=use_lim)

    def _parse_delim_char(self) -> str:
        tok = self.peek()
        if not tok:
            return ""
        self.pop()
        _, v = tok
        return DELIM_NAMES.get(v, v)

    def _parse_atom(self) -> Any:
        tok = self.peek()
        if not tok:
            return Row(())
        k, v = self.pop()

        if k == "num":
            return Atom(v, "num")
        if k == "id":
            return Atom(v, "math")
        if k == "spec":
            if v == "{":
                inner = self.parse_all()
                if self.peek() == ("spec", "}"):
                    self.pop()
                return inner
            if v in ("[", "]"):
                return Sym(v, "open" if v == "[" else "close")
            if v == "~":
                return Space(0.30)
            if v in ("_", "^"):
                # Stray script with missing base
                arg = self._parse_arg()
                return Scripts(Row(()), sub=arg if v == "_" else None, sup=arg if v == "^" else None)
            if v == "'":
                return Sym("′", "ord")
            return Sym(v, "ord")

        if k == "char":
            if v == "+":
                return Sym("+", "bin")
            if v == "-":
                return Sym("−", "bin")
            if v in "*/":
                return Sym(v, "bin" if v == "*" else "ord")
            if v in "=<>":
                return Sym(v, "rel")
            if v == "(":
                return Sym("(", "open")
            if v == ")":
                return Sym(")", "close")
            if v in ",;:":
                return Sym(v, "punct")
            if v == "|":
                return Sym("|", "ord")
            return Sym(v, "ord")

        # k == "cmd"
        if v == "left":
            left = self._parse_delim_char()
            body = self.parse_all(stop_cmds={"right"})
            right = ""
            if self.peek() == ("cmd", "right"):
                self.pop()
                right = self._parse_delim_char()
            return Delim(left, body, right)
        if v == "right":
            return Sym(self._parse_delim_char(), "close")
        if v in SIZE_CMDS:
            nxt = self.peek()
            if nxt and nxt[1] in DELIM_NAMES:
                return Sym(self._parse_delim_char(), "ord")
            return Row(())
        if v in STYLE_CMDS:
            return Row(())
        if v in ("tag", "label"):
            self._consume_braced_raw()
            return Row(())
        if v == "not":
            nxt = self.peek()
            if nxt:
                key = f"\\{nxt[1]}" if nxt[0] == "cmd" else nxt[1]
                if key in NEGATIONS:
                    self.pop()
                    return Sym(NEGATIONS[key], "rel")
            return Sym("̸", "ord")
        if v in ("frac", "dfrac", "tfrac", "cfrac"):
            return Frac(self._parse_arg(), self._parse_arg(), line=True)
        if v in ("binom", "dbinom", "tbinom"):
            return Delim("(", Frac(self._parse_arg(), self._parse_arg(), line=False), ")")
        if v == "sqrt":
            idx = self._parse_opt_bracket()
            return Sqrt(self._parse_arg(), index=idx)
        if v in ("overset", "stackrel"):
            top = self._parse_arg()
            base = self._parse_arg()
            return Scripts(base, sup=top, limits=True)
        if v == "underset":
            bot = self._parse_arg()
            base = self._parse_arg()
            return Scripts(base, sub=bot, limits=True)
        if v == "boxed":
            return Delim("[", self._parse_arg(), "]")
        if v in ACCENTS:
            return Accent(self._parse_arg(), v)
        if v in TEXT_STYLE_CMDS:
            return self._parse_styled_text(v, TEXT_STYLE_CMDS[v])
        if v == "pmod":
            return Row((Space(0.4), Sym("(", "open"), Atom("mod", "rm"), Space(0.25),
                        self._parse_arg(), Sym(")", "close")))
        if v == "begin":
            return self._parse_env()
        if v in SPACES_EM:
            if v == "kern":
                return Space(0.25)
            return Space(SPACES_EM[v])
        if v in GREEK_LOWER:
            return Atom(GREEK_LOWER[v], "math")
        if v in GREEK_UPPER:
            return Atom(GREEK_UPPER[v], "rm")
        if v in BIN_OPS:
            return Sym(BIN_OPS[v], "bin")
        if v in REL_OPS:
            return Sym(REL_OPS[v], "rel")
        if v in BIG_OPS:
            glyph, is_lim = BIG_OPS[v]
            return Sym(glyph, "bigop" if is_lim else "bigop")
        if v in LIMIT_FUNCS:
            return Sym(v, "lim")
        if v in PLAIN_FUNCS:
            return Sym(v, "op")
        if v in MISC_SYMS:
            return Sym(MISC_SYMS[v], "ord")
        if v in DELIM_NAMES:
            return Sym(DELIM_NAMES[v], "ord")
        return Atom(v, "rm")

    def _parse_styled_text(self, cmd: str, style: str) -> Any:
        if cmd in ("text", "textrm", "textup", "textnormal", "hbox", "mbox", "textbf", "textit"):
            return self._parse_literal_text(style)
        arg = self._parse_arg()
        if style in ("bb", "cal"):
            return _map_styled_chars(arg, BLACKBOARD_CHARS if style == "bb" else SCRIPT_CHARS, style)
        return _restyle_node(arg, style)

    def _parse_literal_text(self, style: str) -> Any:
        tok = self.peek()
        if tok != ("spec", "{"):
            arg = self._parse_arg()
            return _restyle_node(arg, style)
        # Re-read literal characters inside braces from raw token list isn't exact for spaces,
        # so convert tokens inside balanced {...} with clean spacing.
        self.pop()
        depth = 1
        words: list[str] = []
        prev_alnum = False
        while self.pos < len(self.toks) and depth > 0:
            k, val = self.pop()
            if k == "spec" and val == "{":
                depth += 1
            elif k == "spec" and val == "}":
                depth -= 1
            elif k == "spec" and val == "~":
                words.append(" ")
                prev_alnum = False
            elif k == "cmd":
                sym = GREEK_LOWER.get(val) or GREEK_UPPER.get(val) or MISC_SYMS.get(val, f" {val} ")
                words.append(sym)
                prev_alnum = False
            else:
                if prev_alnum and val.isalnum() and (not words or words[-1][-1].isalnum()):
                    # Single-char math ids come sequentially in same word ("if" -> ["i","f"])
                    pass
                words.append(val)
                prev_alnum = val.isalnum()
        return Atom("".join(words), style)

    def _parse_env(self) -> Any:
        env = self._consume_braced_raw() or "matrix"
        if env == "array":
            self._consume_braced_raw()  # discard {lcc} alignment spec
        rows: list[tuple[Any, ...]] = []
        while self.pos < len(self.toks):
            if self.peek() == ("cmd", "end"):
                self.pop()
                self._consume_braced_raw()
                break
            row_cells: list[Any] = []
            while self.pos < len(self.toks):
                row_cells.append(self.parse_all(stop_cmds={"end"}))
                tok = self.peek()
                if tok == ("spec", "&"):
                    self.pop()
                    continue
                break
            rows.append(tuple(row_cells))
            tok = self.peek()
            if tok == ("cmd", "\\\\") or tok == ("cmd", "cr"):
                self.pop()
                if self.peek() == ("spec", "["):
                    self._parse_opt_bracket()
                while self.peek() == ("cmd", "hline"):
                    self.pop()
                continue
            if tok == ("cmd", "hline"):
                self.pop()
                continue
            if tok == ("cmd", "end"):
                self.pop()
                self._consume_braced_raw()
                break
            break
        return Table(tuple(rows), env)


def _map_styled_chars(node: Any, table: dict[str, str], fallback_style: str) -> Any:
    if isinstance(node, Atom):
        mapped = "".join(table.get(ch, ch) for ch in node.val)
        return Atom(mapped, fallback_style)
    if isinstance(node, Row):
        return Row(tuple(_map_styled_chars(c, table, fallback_style) for c in node.items))
    return _restyle_node(node, fallback_style)


def _restyle_node(node: Any, style: str) -> Any:
    if isinstance(node, Atom):
        return Atom(node.val, style)
    if isinstance(node, Row):
        return Row(tuple(_restyle_node(c, style) for c in node.items))
    return node


def parse_latex(src: str) -> Any:
    """Parse a LaTeX math expression into an AST (never raises on malformed input)."""
    # Preserve spaces cleanly inside \text{...} BEFORE collapsing whitespace in lexer
    # by converting spaces inside \text{...} to `~` when simple.
    cleaned = _preserve_text_spaces(src)
    return _Parser(_tokenize_math(cleaned)).parse_all()


_TEXT_CMD_RE = re.compile(r"\\(text|textrm|textup|textnormal|hbox|mbox|textbf|textit)\{([^{}]*)\}")


def _preserve_text_spaces(src: str) -> str:
    return _TEXT_CMD_RE.sub(
        lambda m: f"\\{m.group(1)}{{{m.group(2).replace(' ', '~')}}}",
        src,
    )


# ---------------------------------------------------------------------------
# Unary +/- & spacing helpers shared by MathML + Qt renderers
# ---------------------------------------------------------------------------

def _is_unary_context(prev: Any | None) -> bool:
    if prev is None:
        return True
    if isinstance(prev, Space):
        return False
    if isinstance(prev, Sym):
        return prev.kind in ("bin", "rel", "open", "punct")
    return False


# ---------------------------------------------------------------------------
# Native HTML5 MathML Core renderer (Web UI)
# ---------------------------------------------------------------------------

def _accent_op(cmd: str) -> str:
    return {
        "vec": "→", "overrightarrow": "→", "overleftarrow": "←",
        "hat": "^", "widehat": "^", "tilde": "~", "widetilde": "~",
        "bar": "‾", "overline": "‾", "underline": "_",
        "dot": "˙", "ddot": "¨", "check": "ˇ", "breve": "˘",
    }.get(cmd, "‾")


def _to_mathml(node: Any) -> str:
    esc = html.escape
    if isinstance(node, Row):
        if not node.items:
            return "<mrow></mrow>"
        parts: list[str] = []
        prev = None
        for item in node.items:
            if isinstance(item, Sym) and item.unicode in "+−" and _is_unary_context(prev):
                parts.append(f'<mo form="prefix" lspace="0em" rspace="0em">{esc(item.unicode)}</mo>')
            else:
                parts.append(_to_mathml(item))
            prev = item
        return parts[0] if len(parts) == 1 else f"<mrow>{''.join(parts)}</mrow>"
    if isinstance(node, Atom):
        v = esc(node.val)
        if node.style == "num":
            return f"<mn>{v}</mn>"
        if node.style in ("text", "rm"):
            return f"<mtext>{v}</mtext>" if " " in node.val else f'<mi mathvariant="normal">{v}</mi>'
        if node.style == "bf":
            return f'<mi mathvariant="bold">{v}</mi>'
        if node.style in ("bb", "cal", "it", "math"):
            return f"<mi>{v}</mi>"
        return f"<mi>{v}</mi>"
    if isinstance(node, Sym):
        u = esc(node.unicode)
        if node.kind == "lim":
            return f"<mo>{u}</mo>"
        if node.kind == "op":
            return f"<mi>{u}</mi>"
        return f"<mo>{u}</mo>"
    if isinstance(node, Space):
        return f'<mspace width="{node.em:.2f}em"/>'
    if isinstance(node, Frac):
        num = _to_mathml(node.num)
        den = _to_mathml(node.den)
        attr = "" if node.line else ' linethickness="0"'
        return f"<mfrac{attr}><mrow>{num}</mrow><mrow>{den}</mrow></mfrac>"
    if isinstance(node, Sqrt):
        rad = f"<mrow>{_to_mathml(node.rad)}</mrow>"
        if node.index is None:
            return f"<msqrt>{rad}</msqrt>"
        return f"<mroot>{rad}<mrow>{_to_mathml(node.index)}</mrow></mroot>"
    if isinstance(node, Scripts):
        base = _to_mathml(node.base)
        sub = f"<mrow>{_to_mathml(node.sub)}</mrow>" if node.sub is not None else None
        sup = f"<mrow>{_to_mathml(node.sup)}</mrow>" if node.sup is not None else None
        if node.limits:
            if sub and sup:
                return f"<munderover>{base}{sub}{sup}</munderover>"
            if sub:
                return f"<munder>{base}{sub}</munder>"
            if sup:
                return f"<mover>{base}{sup}</mover>"
        if sub and sup:
            return f"<msubsup>{base}{sub}{sup}</msubsup>"
        if sub:
            return f"<msub>{base}{sub}</msub>"
        if sup:
            return f"<msup>{base}{sup}</msup>"
        return base
    if isinstance(node, Accent):
        op = esc(_accent_op(node.acc))
        tag = "munder" if node.acc == "underline" else "mover"
        return f'<{tag} accent="true"><mrow>{_to_mathml(node.base)}</mrow><mo stretchy="true">{op}</mo></{tag}>'
    if isinstance(node, Delim):
        left = f'<mo stretchy="true">{esc(node.left)}</mo>' if node.left else ""
        right = f'<mo stretchy="true">{esc(node.right)}</mo>' if node.right else ""
        return f"<mrow>{left}{_to_mathml(node.body)}{right}</mrow>"
    if isinstance(node, Table):
        return _table_to_mathml(node)
    return "<mrow></mrow>"


def _table_to_mathml(tab: Table) -> str:
    esc = html.escape
    kind = tab.kind
    aligned = kind.startswith("align")
    cases = kind == "cases"
    trs: list[str] = []
    for row in tab.rows:
        tds: list[str] = []
        for idx, cell in enumerate(row):
            if aligned:
                al = "right" if idx % 2 == 0 else "left"
                pad = "0.18em" if idx % 2 == 0 else "0"
                tds.append(f'<mtd style="text-align:{al};padding-right:{pad}">{_to_mathml(cell)}</mtd>')
            elif cases:
                tds.append(f'<mtd style="text-align:left;padding-right:0.6em">{_to_mathml(cell)}</mtd>')
            else:
                tds.append(f"<mtd>{_to_mathml(cell)}</mtd>")
        trs.append(f"<mtr>{''.join(tds)}</mtr>")
    tbl = f"<mtable>{''.join(trs)}</mtable>"
    l_del, r_del = MATRIX_DELIMS.get(kind, ("", ""))
    left = f'<mo stretchy="true">{esc(l_del)}</mo>' if l_del else ""
    right = f'<mo stretchy="true">{esc(r_del)}</mo>' if r_del else ""
    return f"<mrow>{left}{tbl}{right}</mrow>" if (left or right) else tbl


def ast_to_mathml(latex: str, display: bool) -> str:
    """Convert a LaTeX math string into an HTML5 MathML Core `<math>` element."""
    ast = parse_latex(latex)
    attr = ' display="block"' if display else ""
    return f'<math xmlns="http://www.w3.org/1998/Math/MathML"{attr}>{_to_mathml(ast)}</math>'


# ---------------------------------------------------------------------------
# Qt QTextDocument inline HTML + 2x HiDPI QPainter math box renderer
# ---------------------------------------------------------------------------

_QT_IMG_CACHE: OrderedDict[tuple[str, bool, int, str], tuple[Any, float, float]] = OrderedDict()
_QT_CACHE_MAX = 256


def _is_simple_inline_ast(node: Any) -> bool:
    """True when an inline formula renders cleanly as selectable HTML without QPainter."""
    if isinstance(node, (Frac, Sqrt, Accent, Table)):
        return False
    if isinstance(node, Delim):
        return _is_simple_inline_ast(node.body)
    if isinstance(node, Sym):
        return node.kind != "bigop" and node.kind != "lim"
    if isinstance(node, Scripts):
        if node.limits or (node.sub is not None and node.sup is not None):
            return False
        if not isinstance(node.base, (Atom, Sym)):
            return False
        if node.sub is not None and _has_scripts_or_complex(node.sub):
            return False
        if node.sup is not None and _has_scripts_or_complex(node.sup):
            return False
        return True
    if isinstance(node, Row):
        return all(_is_simple_inline_ast(c) for c in node.items)
    return isinstance(node, (Atom, Space))


def _has_scripts_or_complex(node: Any) -> bool:
    if isinstance(node, (Frac, Sqrt, Accent, Table, Scripts, Delim)):
        return True
    if isinstance(node, Row):
        return any(_has_scripts_or_complex(c) for c in node.items)
    return False


def _ast_to_simple_html(node: Any) -> str:
    esc = html.escape
    if isinstance(node, Row):
        out: list[str] = []
        prev = None
        for item in node.items:
            if isinstance(item, Sym) and item.unicode in "+−" and _is_unary_context(prev):
                out.append(esc(item.unicode))
            else:
                out.append(_ast_to_simple_html(item))
            prev = item
        return "".join(out)
    if isinstance(node, Atom):
        txt = esc(node.val)
        if node.style == "math":
            return f"<i>{txt}</i>"
        if node.style == "bf":
            return f"<b>{txt}</b>"
        if node.style == "it":
            return f"<i>{txt}</i>"
        return txt
    if isinstance(node, Sym):
        u = esc(node.unicode)
        if node.kind == "bin":
            return f" {u} "
        if node.kind == "rel":
            return f" {u} "
        if node.kind == "punct":
            return f"{u} "
        if node.kind == "op":
            return u
        return u
    if isinstance(node, Space):
        return "&nbsp;" if node.em >= 0.45 else "&thinsp;"
    if isinstance(node, Delim):
        return f"{esc(node.left)}{_ast_to_simple_html(node.body)}{esc(node.right)}"
    if isinstance(node, Scripts):
        s = _ast_to_simple_html(node.base)
        if node.sub is not None:
            s += f"<sub>{_ast_to_simple_html(node.sub)}</sub>"
        if node.sup is not None:
            s += f"<sup>{_ast_to_simple_html(node.sup)}</sup>"
        return s
    return ""


class _Box:
    """Layout box measured relative to the math baseline (y=0; up=asc, down=desc)."""
    __slots__ = ("w", "asc", "desc", "draw_fn")

    def __init__(self, w: float, asc: float, desc: float, draw_fn):
        self.w = w
        self.asc = asc
        self.desc = desc
        self.draw_fn = draw_fn

    @property
    def h(self) -> float:
        return self.asc + self.desc


def _render_qt_image(latex: str, display: bool, base_px: float, fg_hex: str):
    """Render AST to a cached 2x HiDPI QImage and return (img, logical_w, logical_h)."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen

    px_int = max(12, int(round(base_px)))
    key = (latex, display, px_int, fg_hex)
    cached = _QT_IMG_CACHE.get(key)
    if cached is not None:
        _QT_IMG_CACHE.move_to_end(key)
        return cached

    ast = parse_latex(latex)
    fg = QColor(fg_hex)

    def make_font(px: float, italic: bool = False, bold: bool = False, text: bool = False) -> QFont:
        fam = "Segoe UI" if text else "Cambria Math"
        f = QFont(fam)
        if not text:
            f.setFamilies(["Cambria Math", "Segoe UI Symbol", "Segoe UI"])
        f.setPixelSize(max(14, int(round(px))))
        f.setHintingPreference(QFont.PreferNoHinting)
        f.setItalic(italic)
        f.setBold(bold)
        return f

    def glyph_box(text: str, px: float, italic: bool = False, bold: bool = False,
                  is_text: bool = False, extra_x: float = 0.0) -> _Box:
        f = make_font(px, italic=italic, bold=bold, text=is_text)
        fm = QFontMetricsF(f)
        rect = fm.tightBoundingRect(text)
        left_pad = max(0.0, -rect.left())
        w = left_pad + max(fm.horizontalAdvance(text), rect.right()) + extra_x * 2
        asc = max(fm.ascent() * 0.85, -rect.top() + 1.0)
        desc = max(fm.descent() * 0.75, rect.bottom() + 1.0)

        def draw(p: QPainter, x: float, y: float):
            p.setFont(f)
            p.setPen(fg)
            p.drawText(x + extra_x + left_pad, y, text)

        return _Box(w, asc, desc, draw)

    def hbox(children: list[_Box], gaps: list[float] | None = None) -> _Box:
        if not children:
            return _Box(0.0, px_int * 0.65, px_int * 0.2, lambda p, x, y: None)
        offsets: list[float] = []
        x = 0.0
        for i, b in enumerate(children):
            offsets.append(x)
            g = gaps[i] if gaps and i < len(gaps) else 0.0
            x += b.w + g
        asc = max(b.asc for b in children)
        desc = max(b.desc for b in children)

        def draw(p: QPainter, ox: float, oy: float):
            for b, dx in zip(children, offsets):
                b.draw_fn(p, ox + dx, oy)

        return _Box(max(0.0, x), asc, desc, draw)

    def delim_box(ch: str, target_asc: float, target_desc: float, px: float) -> _Box:
        if not ch:
            return _Box(0.0, target_asc, target_desc, lambda p, x, y: None)
        total_h = target_asc + target_desc
        if total_h <= px * 1.45 and ch not in "{}":
            return glyph_box(ch, px, italic=False)
        # Draw smooth vector delimiter spanning [-target_asc, +target_desc]
        stroke = max(1.15, px * 0.075)
        w = px * (0.48 if ch in "{}" else 0.36 if ch in "()⟨⟩" else 0.44 if ch == "‖" else 0.34)
        pad_x = px * 0.08
        total_w = w + pad_x * 2

        def draw(p: QPainter, x: float, y: float):
            pen = QPen(fg, stroke, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            top = y - target_asc + stroke
            bot = y + target_desc - stroke
            mid = 0.5 * (top + bot)
            lx = x + pad_x
            rx = lx + w
            path = QPainterPath()
            if ch == "(":
                h = bot - top
                path.moveTo(rx, top)
                path.cubicTo(lx, top + h * 0.22, lx, bot - h * 0.22, rx, bot)
                p.drawPath(path)
            elif ch == ")":
                h = bot - top
                path.moveTo(lx, top)
                path.cubicTo(rx, top + h * 0.22, rx, bot - h * 0.22, lx, bot)
                p.drawPath(path)
            elif ch == "[":
                path.moveTo(rx, top)
                path.lineTo(lx + stroke, top)
                path.lineTo(lx + stroke, bot)
                path.lineTo(rx, bot)
                p.drawPath(path)
            elif ch == "]":
                path.moveTo(lx, top)
                path.lineTo(rx - stroke, top)
                path.lineTo(rx - stroke, bot)
                path.lineTo(lx, bot)
                p.drawPath(path)
            elif ch == "{":
                xm = lx + w * 0.52
                path.moveTo(rx, top)
                path.cubicTo(xm, top, xm + w * 0.1, mid - (bot - top) * 0.06, lx, mid)
                path.cubicTo(xm + w * 0.1, mid + (bot - top) * 0.06, xm, bot, rx, bot)
                p.drawPath(path)
            elif ch == "}":
                xm = lx + w * 0.48
                path.moveTo(lx, top)
                path.cubicTo(xm, top, xm - w * 0.1, mid - (bot - top) * 0.06, rx, mid)
                path.cubicTo(xm - w * 0.1, mid + (bot - top) * 0.06, xm, bot, lx, bot)
                p.drawPath(path)
            elif ch == "|":
                cx = lx + w * 0.5
                p.drawLine(cx, top, cx, bot)
            elif ch == "‖":
                p.drawLine(lx + w * 0.30, top, lx + w * 0.30, bot)
                p.drawLine(lx + w * 0.70, top, lx + w * 0.70, bot)
            elif ch == "⟨":
                path.moveTo(rx, top)
                path.lineTo(lx, mid)
                path.lineTo(rx, bot)
                p.drawPath(path)
            elif ch == "⟩":
                path.moveTo(lx, top)
                path.lineTo(rx, mid)
                path.lineTo(lx, bot)
                p.drawPath(path)
            elif ch in "⌊⌈":
                if ch == "⌊":
                    path.moveTo(lx + stroke, top)
                    path.lineTo(lx + stroke, bot)
                    path.lineTo(rx, bot)
                else:
                    path.moveTo(rx, top)
                    path.lineTo(lx + stroke, top)
                    path.lineTo(lx + stroke, bot)
                p.drawPath(path)
            elif ch in "⌋⌉":
                if ch == "⌋":
                    path.moveTo(rx - stroke, top)
                    path.lineTo(rx - stroke, bot)
                    path.lineTo(lx, bot)
                else:
                    path.moveTo(lx, top)
                    path.lineTo(rx - stroke, top)
                    path.lineTo(rx - stroke, bot)
                p.drawPath(path)

        return _Box(total_w, target_asc, target_desc, draw)

    def layout(node: Any, px: float, style_disp: bool) -> _Box:
        axis_h = px * 0.28
        if isinstance(node, Row):
            if not node.items:
                return _Box(0.0, px * 0.65, px * 0.2, lambda p, x, y: None)
            boxes: list[_Box] = []
            gaps: list[float] = []
            prev = None
            for idx, item in enumerate(node.items):
                unary = isinstance(item, Sym) and item.unicode in "+−" and _is_unary_context(prev)
                if unary:
                    b = glyph_box(item.unicode, px, italic=False)
                else:
                    b = layout(item, px, style_disp)
                boxes.append(b)
                # Compute trailing gap after current item
                nxt = node.items[idx + 1] if idx + 1 < len(node.items) else None
                gap = 0.0
                if nxt is not None and not isinstance(item, Space) and not isinstance(nxt, Space):
                    if unary:
                        gap = 0.0
                    elif isinstance(item, Sym) and item.kind in ("bin", "rel"):
                        gap = px * (0.20 if item.kind == "bin" else 0.26)
                    elif isinstance(nxt, Sym) and nxt.kind in ("bin", "rel"):
                        gap = px * (0.20 if nxt.kind == "bin" else 0.26)
                    elif isinstance(item, Sym) and item.kind == "punct":
                        gap = px * 0.16
                    elif isinstance(item, Sym) and item.kind in ("op", "lim", "bigop"):
                        gap = px * 0.14
                    elif isinstance(item, Atom) and isinstance(nxt, Atom) and item.style == "num" and nxt.style == "math":
                        gap = px * 0.05
                gaps.append(gap)
                prev = item
            return hbox(boxes, gaps)

        if isinstance(node, Atom):
            italic = node.style in ("math", "it")
            bold = node.style == "bf"
            is_text = node.style == "text"
            return glyph_box(node.val, px, italic=italic, bold=bold, is_text=is_text)

        if isinstance(node, Sym):
            scale = 1.38 if (node.kind == "bigop" and style_disp) else 1.15 if node.kind == "bigop" else 1.0
            return glyph_box(node.unicode, px * scale, italic=False)

        if isinstance(node, Space):
            return _Box(max(0.0, node.em * px), px * 0.5, 0.0, lambda p, x, y: None)

        if isinstance(node, Frac):
            sub_px = max(10.0, px * (0.86 if style_disp else 0.78))
            nb = layout(node.num, sub_px, False)
            db = layout(node.den, sub_px, False)
            pad_x = px * 0.14
            gap_y = px * (0.18 if node.line else 0.14)
            bar_thick = max(1.0, px * 0.07)
            w = max(nb.w, db.w) + pad_x * 2
            asc = axis_h + gap_y + nb.h
            desc = max(px * 0.25, db.h + gap_y - axis_h)

            def draw(p: QPainter, x: float, y: float):
                bar_y = y - axis_h
                nb.draw_fn(p, x + (w - nb.w) * 0.5, bar_y - gap_y - nb.desc)
                if node.line:
                    p.setPen(Qt.NoPen)
                    p.setBrush(fg)
                    p.drawRect(x + 1.0, bar_y - bar_thick * 0.5, w - 2.0, bar_thick)
                db.draw_fn(p, x + (w - db.w) * 0.5, bar_y + gap_y + db.asc)

            return _Box(w, asc, desc, draw)

        if isinstance(node, Sqrt):
            rb = layout(node.rad, px, style_disp)
            ib = layout(node.index, max(9.0, px * 0.62), False) if node.index is not None else None
            stroke = max(1.15, px * 0.072)
            rad_hook_w = px * 0.56
            idx_extra = max(0.0, ib.w - rad_hook_w * 0.35) if ib else 0.0
            top_gap = px * 0.16
            r_asc = max(rb.asc + top_gap, px * 0.78)
            if ib:
                r_asc = max(r_asc, (rb.asc + rb.desc) * 0.35 + ib.h)
            r_desc = max(rb.desc + px * 0.06, px * 0.22)
            over_pad = px * 0.14
            w = idx_extra + rad_hook_w + rb.w + over_pad

            def draw(p: QPainter, x: float, y: float):
                rx = x + idx_extra
                top_y = y - r_asc + stroke
                bot_y = y + r_desc - stroke
                mid_y = y - (r_asc - r_desc) * 0.1
                path = QPainterPath()
                path.moveTo(rx, mid_y)
                path.lineTo(rx + rad_hook_w * 0.22, mid_y - px * 0.08)
                path.lineTo(rx + rad_hook_w * 0.55, bot_y)
                path.lineTo(rx + rad_hook_w * 0.92, top_y)
                path.lineTo(x + w - 1.0, top_y)
                p.setPen(QPen(fg, stroke, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                rb.draw_fn(p, rx + rad_hook_w + px * 0.04, y)
                if ib:
                    ib.draw_fn(p, x, mid_y - px * 0.14 - ib.desc)

            return _Box(w, r_asc, r_desc, draw)

        if isinstance(node, Scripts):
            bb = layout(node.base, px, style_disp)
            scr_px = max(9.0, px * 0.68)
            sb = layout(node.sub, scr_px, False) if node.sub is not None else None
            pb = layout(node.sup, scr_px, False) if node.sup is not None else None
            if node.limits and style_disp:
                w = max(bb.w, sb.w if sb else 0.0, pb.w if pb else 0.0)
                gap = px * 0.10
                asc = bb.asc + (gap + pb.h if pb else 0.0)
                desc = bb.desc + (gap + sb.h if sb else 0.0)

                def draw(p: QPainter, x: float, y: float):
                    bb.draw_fn(p, x + (w - bb.w) * 0.5, y)
                    if pb:
                        pb.draw_fn(p, x + (w - pb.w) * 0.5, y - bb.asc - gap - pb.desc)
                    if sb:
                        sb.draw_fn(p, x + (w - sb.w) * 0.5, y + bb.desc + gap + sb.asc)

                return _Box(w, asc, desc, draw)

            # Standard side scripts
            scr_w = max(sb.w if sb else 0.0, pb.w if pb else 0.0)
            sup_shift = max(px * 0.36, bb.asc * 0.55)
            sub_shift = max(px * 0.24, bb.desc * 0.70)
            # Prevent collision when both sub and sup exist on tall bases
            if pb and sb and (y_sup := sup_shift - pb.desc) <= (y_sub := -sub_shift + sb.asc) + px * 0.08:
                mid = 0.5 * (y_sup + y_sub)
                sup_shift = mid + pb.desc + px * 0.06
                sub_shift = -(mid - sb.asc - px * 0.06)
            asc = max(bb.asc, sup_shift + pb.asc if pb else 0.0)
            desc = max(bb.desc, sub_shift + sb.desc if sb else 0.0)
            gap_x = px * 0.04

            def draw(p: QPainter, x: float, y: float):
                bb.draw_fn(p, x, y)
                sx = x + bb.w + gap_x
                if pb:
                    pb.draw_fn(p, sx, y - sup_shift)
                if sb:
                    sb.draw_fn(p, sx, y + sub_shift)

            return _Box(bb.w + gap_x + scr_w, asc, desc, draw)

        if isinstance(node, Accent):
            bb = layout(node.base, px, style_disp)
            stroke = max(1.05, px * 0.068)
            gap = px * 0.14
            if node.acc == "underline":
                desc = bb.desc + gap + stroke * 1.5
                def draw(p: QPainter, x: float, y: float):
                    bb.draw_fn(p, x, y)
                    p.setPen(QPen(fg, stroke, Qt.SolidLine, Qt.RoundCap))
                    p.drawLine(x, y + bb.desc + gap * 0.7, x + bb.w, y + bb.desc + gap * 0.7)
                return _Box(bb.w, bb.asc, desc, draw)
            acc_h = px * 0.24
            asc = bb.asc + gap + acc_h
            def draw(p: QPainter, x: float, y: float):
                bb.draw_fn(p, x, y)
                top = y - bb.asc - gap * 0.55
                p.setPen(QPen(fg, stroke, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                if node.acc in ("bar", "overline"):
                    p.drawLine(x + 1.0, top, x + bb.w - 1.0, top)
                elif node.acc in ("hat", "widehat"):
                    span = bb.w if node.acc == "widehat" else min(bb.w - 2.0, px * 0.52)
                    cx = x + bb.w * 0.5
                    path = QPainterPath()
                    path.moveTo(cx - span * 0.5, top)
                    path.lineTo(cx, top - acc_h * 0.65)
                    path.lineTo(cx + span * 0.5, top)
                    p.drawPath(path)
                elif node.acc in ("tilde", "widetilde"):
                    span = bb.w if node.acc == "widetilde" else min(bb.w - 2.0, px * 0.55)
                    lx = x + (bb.w - span) * 0.5
                    path = QPainterPath()
                    path.moveTo(lx, top)
                    path.cubicTo(lx + span * 0.3, top - acc_h * 0.85, lx + span * 0.7, top + acc_h * 0.35, lx + span, top - acc_h * 0.3)
                    p.drawPath(path)
                elif node.acc in ("vec", "overrightarrow", "overleftarrow"):
                    p.drawLine(x + 1.0, top, x + bb.w - 1.5, top)
                    ax = (x + 1.5) if node.acc == "overleftarrow" else (x + bb.w - 1.5)
                    dx = px * 0.16 * (1 if node.acc == "overleftarrow" else -1)
                    p.drawLine(ax, top, ax + dx, top - px * 0.11)
                    p.drawLine(ax, top, ax + dx, top + px * 0.11)
                elif node.acc in ("dot", "ddot"):
                    p.setBrush(fg)
                    p.setPen(Qt.NoPen)
                    r = max(1.1, stroke * 0.9)
                    if node.acc == "dot":
                        p.drawEllipse(x + bb.w * 0.5 - r, top - r, r * 2, r * 2)
                    else:
                        p.drawEllipse(x + bb.w * 0.34 - r, top - r, r * 2, r * 2)
                        p.drawEllipse(x + bb.w * 0.66 - r, top - r, r * 2, r * 2)
                else:
                    p.drawLine(x + 1.0, top, x + bb.w - 1.0, top)
            return _Box(bb.w, asc, bb.desc, draw)

        if isinstance(node, Delim):
            bb = layout(node.body, px, style_disp)
            lb = delim_box(node.left, bb.asc, bb.desc, px)
            rb = delim_box(node.right, bb.asc, bb.desc, px)
            return hbox([lb, bb, rb], [px * 0.04, px * 0.04, 0.0])

        if isinstance(node, Table):
            cell_px = px * (0.94 if len(node.rows) > 1 else 1.0)
            grid = [[layout(c, cell_px, style_disp) for c in row] for row in node.rows]
            ncols = max((len(r) for r in grid), default=0)
            col_ws = [
                max((r[c].w for r in grid if c < len(r)), default=0.0)
                for c in range(ncols)
            ]
            row_ascs = [max((b.asc for b in r), default=px * 0.6) for r in grid]
            row_descs = [max((b.desc for b in r), default=px * 0.2) for r in grid]
            aligned = node.kind.startswith("align")
            cases = node.kind == "cases"
            col_gap = px * (0.14 if aligned else 0.75 if cases else 0.65)
            row_gap = px * 0.34
            inner_w = sum(col_ws) + col_gap * max(0, ncols - 1)
            inner_h = sum(a + d for a, d in zip(row_ascs, row_descs)) + row_gap * max(0, len(grid) - 1)
            # Center table vertically around the math axis (-axis_h)
            half_h = inner_h * 0.5
            asc = max(px * 0.65, half_h + axis_h)
            desc = max(px * 0.25, half_h - axis_h)

            def draw_table(p: QPainter, x: float, y: float):
                cur_y = y - axis_h - half_h
                for r_idx, row in enumerate(grid):
                    base_y = cur_y + row_ascs[r_idx]
                    cur_x = x
                    for c_idx in range(ncols):
                        if c_idx < len(row):
                            cb = row[c_idx]
                            if aligned:
                                dx = (col_ws[c_idx] - cb.w) if c_idx % 2 == 0 else 0.0
                            elif cases:
                                dx = 0.0
                            else:
                                dx = (col_ws[c_idx] - cb.w) * 0.5
                            cb.draw_fn(p, cur_x + dx, base_y)
                        cur_x += col_ws[c_idx] + col_gap
                    cur_y += row_ascs[r_idx] + row_descs[r_idx] + row_gap

            tbox = _Box(inner_w, asc, desc, draw_table)
            l_del, r_del = MATRIX_DELIMS.get(node.kind, ("", ""))
            if l_del or r_del:
                lb = delim_box(l_del, tbox.asc, tbox.desc, px)
                rb = delim_box(r_del, tbox.asc, tbox.desc, px)
                return hbox([lb, tbox, rb], [px * 0.08, px * 0.08, 0.0])
            return tbox

        return _Box(0.0, px * 0.6, px * 0.2, lambda p, x, y: None)

    scale = 2
    hi_px = float(px_int * scale)
    box = layout(ast, hi_px, display)
    pad_x = 4.0 * scale
    pad_y = 3.0 * scale
    if not display:
        # Balance image canvas around the math axis so `vertical-align: middle`
        # aligns the formula baseline cleanly with surrounding prose text.
        axis_h = hi_px * 0.28
        half_v = max(box.asc - axis_h, box.desc + axis_h) + pad_y
        hi_h = max(12.0 * scale, _pymath.ceil(half_v * 2.0))
        baseline_y = hi_h * 0.5 + axis_h
    else:
        hi_h = max(16.0 * scale, _pymath.ceil(box.asc + box.desc + pad_y * 2.0))
        baseline_y = pad_y + box.asc

    hi_w = max(8.0 * scale, _pymath.ceil(box.w + pad_x * 2.0))
    img = QImage(int(hi_w), int(hi_h), QImage.Format_ARGB32_Premultiplied)
    img.fill(Qt.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setRenderHint(QPainter.TextAntialiasing, True)
    box.draw_fn(painter, pad_x, baseline_y)
    painter.end()

    res = (img, float(hi_w) / scale, float(hi_h) / scale)
    _QT_IMG_CACHE[key] = res
    while len(_QT_IMG_CACHE) > _QT_CACHE_MAX:
        _QT_IMG_CACHE.popitem(last=False)
    return res


# ---------------------------------------------------------------------------
# Markdown -> HTML pipeline + public report renderers
# ---------------------------------------------------------------------------

_MD_PARSER = None


def _markdown_to_html(masked_md: str) -> str:
    global _MD_PARSER
    if _MD_PARSER is None:
        try:
            from markdown_it import MarkdownIt
            _MD_PARSER = MarkdownIt("commonmark", {"html": False, "breaks": False}).enable(
                ["table", "strikethrough"]
            )
        except Exception:
            _MD_PARSER = False
    if _MD_PARSER:
        return _MD_PARSER.render(masked_md)
    # Minimal fallback if markdown-it-py is ever unavailable
    paras = [f"<p>{html.escape(p).replace(chr(10), '<br>')}</p>" for p in masked_md.split("\n\n") if p.strip()]
    return "".join(paras)


_DISP_P_RE = re.compile(r"<p>\s*(XELQMATHD\d+X)\s*</p>")
_ANY_TOK_RE = re.compile(r"XELQMATH([DI])(\d+)X")


def _stitch_math(html_body: str, tokens: dict[str, tuple[str, bool]], render_fn) -> str:
    def wrap_block(tag: str) -> str:
        latex, _ = tokens[tag]
        return render_fn(latex, True)

    out = _DISP_P_RE.sub(lambda m: f'<div class="math-block">{wrap_block(m.group(1))}</div>', html_body)

    def repl(m: re.Match) -> str:
        tag = m.group(0)
        entry = tokens.get(tag)
        if not entry:
            return tag
        latex, disp = entry
        inner = render_fn(latex, disp)
        return f'<div class="math-block">{inner}</div>' if disp else inner

    return _ANY_TOK_RE.sub(repl, out)


_WEB_REPORT_CACHE: OrderedDict[str, str] = OrderedDict()


def render_report_web(md_text: str) -> str:
    """Render Markdown + LaTeX into HTML5 + native MathML Core for the Web UI."""
    cached = _WEB_REPORT_CACHE.get(md_text)
    if cached is not None:
        _WEB_REPORT_CACHE.move_to_end(md_text)
        return cached
    masked, tokens = _extract_math(md_text or "")
    body = _markdown_to_html(masked)
    res = _stitch_math(body, tokens, ast_to_mathml)
    _WEB_REPORT_CACHE[md_text] = res
    while len(_WEB_REPORT_CACHE) > 32:
        _WEB_REPORT_CACHE.popitem(last=False)
    return res


_QT_CSS = """
<style>
body { color: #ececf0; font-family: "Segoe UI", system-ui, sans-serif; line-height: 1.45; }
h1 { font-size: 18px; color: #ffffff; margin: 0 0 10px 0; }
h2 { font-size: 15px; color: #7db4ff; margin: 14px 0 6px 0; }
h3 { font-size: 14px; color: #c7d9ff; margin: 12px 0 4px 0; }
h4 { font-size: 13px; color: #b8b8c4; margin: 10px 0 4px 0; }
p { margin: 0 0 8px 0; }
ul, ol { margin: 4px 0 8px 0; }
li { margin: 2px 0; }
code { background-color: #23232b; color: #e2e8f0; font-family: Consolas, monospace; font-size: 12px; padding: 1px 4px; }
pre { background-color: #1c1c23; color: #e2e8f0; font-family: Consolas, monospace; font-size: 12px; padding: 8px 10px; margin: 8px 0; }
blockquote { margin: 6px 0 8px 0; padding: 4px 10px; color: #b8b8c4; background-color: #191920; }
table { border-collapse: collapse; margin: 8px 0; }
th { background-color: #22222c; color: #ffffff; font-weight: 600; padding: 5px 8px; border: 1px solid #383846; }
td { padding: 5px 8px; border: 1px solid #30303d; }
a { color: #7db4ff; }
.math-block { margin: 10px 0; text-align: center; }
</style>
"""


def render_report_qt(doc, label: str, md_text: str, max_width: int = 0) -> str:
    """Render Markdown + LaTeX to Qt Rich HTML and register math QImages on `doc`."""
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QTextDocument

    base_px = 15.0
    try:
        pt = doc.defaultFont().pointSizeF()
        px = doc.defaultFont().pixelSize()
        if px > 0:
            base_px = max(14.0, float(px) * 1.15)
        elif pt > 0:
            base_px = max(14.0, float(pt) * 1.48)
    except Exception:
        pass

    masked, tokens = _extract_math(md_text or "")
    body = _markdown_to_html(masked)
    fg_hex = "#ececf0"

    def render_qt_math(latex: str, display: bool) -> str:
        if not display:
            ast = parse_latex(latex)
            if _is_simple_inline_ast(ast):
                return f'<span style="font-family:\'Cambria Math\',\'Segoe UI Symbol\',serif;">{_ast_to_simple_html(ast)}</span>'
        img, w, h = _render_qt_image(latex, display, base_px, fg_hex)
        if max_width > 40 and w > max_width:
            ratio = max_width / w
            w = max_width
            h = max(10.0, h * ratio)
        digest = hashlib.sha1(f"{display}:{int(base_px)}:{latex}".encode("utf-8")).hexdigest()[:16]
        uri = f"math://{digest}"
        doc.addResource(QTextDocument.ResourceType.ImageResource, QUrl(uri), img)
        align = "" if display else ' style="vertical-align: middle;"'
        return f'<img src="{uri}" width="{int(round(w))}" height="{int(round(h))}"{align}>'

    rendered_body = _stitch_math(body, tokens, render_qt_math)
    rendered_body = rendered_body.replace(
        "<table>",
        '<table border="1" cellspacing="0" cellpadding="6" style="border-color:#353543;margin:8px 0;">',
    )
    rendered_body = re.sub(
        r'<div class="math-block">(.*?)</div>',
        r'<p align="center" style="margin:8px 0;">\1</p>',
        rendered_body,
        flags=re.DOTALL,
    )
    return f"<!DOCTYPE html><html><head>{_QT_CSS}</head><body><h1>{html.escape(label)}</h1>{rendered_body}</body></html>"
