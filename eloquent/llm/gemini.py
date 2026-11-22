"""Google Gemini / Gemma provider with Smart Free-Tier routing and automatic fallback.

In ``auto`` mode (default), each Eloquent task routes to the best fast Free-Tier model
with automatic cooldown + fallback on 429 (rate limit), 503/504 (congestion/timeout),
or 404 (deprecated):
- Live polishing : gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemma-4-26b-a4b-it -> gemma-4-31b-it
- Voice dictation: gemma-4-26b-a4b-it -> gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemma-4-31b-it
- Ask AI (cursor): gemini-3.6-flash -> gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemma-4-26b-a4b-it
- Reports        : gemini-3.6-flash -> gemini-3.5-flash-lite -> gemini-3.1-flash-lite -> gemini-4-hydrogen -> gemini-3.8-flash
"""

import threading
import time

from .base import LLMProvider
from .prompts import (build_compose_prompt, build_proofread_prompt, compose_system_prompt,
                      dictation_system_prompt, proofread_system_prompt)
from ..config import is_arabic_speech
from ..textutil import strip_wrappers

# Ordered fast-first Free-Tier routing chains (measured on Google AI Studio Free Tier)
AUTO_CHAINS: dict[str, tuple[str, ...]] = {
    "proofread": (
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-26b-a4b-it",
        "gemma-4-31b-it",
    ),
    "dictation": (
        "gemma-4-26b-a4b-it",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-31b-it",
    ),
    "compose": (
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemma-4-26b-a4b-it",
    ),
    "generate": (
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
    ),
}

# Gemma 4 hosted on AI Studio Free Tier has a 16K TPM cap; skip it when prompt alone is very large.
_GEMMA_MAX_INPUT_CHARS = 32_000


def _is_gemma(model: str) -> bool:
    return model.lower().startswith("gemma-")


def _thinking_config_for(types, model: str, desired: str):
    """Return a model-safe ThinkingConfig (or None) so no model rejects unsupported levels."""
    m = model.lower()
    if m.startswith("gemma-4"):
        return types.ThinkingConfig(thinking_level="minimal")
    if m.startswith("gemma-"):
        return None
    if m.startswith("gemini-4") or m.startswith("gemini-3.8"):
        # Gen 4 & 3.8 Flash reject thinking_level="minimal" with 400 INVALID_ARGUMENT
        lvl = "low" if desired in ("minimal", "low") else desired
        return types.ThinkingConfig(thinking_level=lvl)
    if m.startswith("gemini-3"):
        return types.ThinkingConfig(thinking_level=desired)
    if m.startswith("gemini-2.5"):
        if desired == "minimal":
            return types.ThinkingConfig(thinking_budget=0)
        return None
    return None


def _classify_error(exc: Exception) -> str:
    """Classify an API exception as 'auth', 'rate_limit', 'transient', 'not_found', 'thinking', or 'other'."""
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(k in msg for k in ("api_key_invalid", "api key not valid", "401", "403", "permission_denied", "unauthenticated")):
        return "auth"
    if any(k in msg for k in ("429", "resource_exhausted", "quota", "rate limit", "too many requests")):
        return "rate_limit"
    if any(k in msg for k in ("503", "504", "unavailable", "deadline_exceeded", "timed out", "timeout", "overloaded", "internal")):
        return "transient"
    if any(k in msg for k in ("404", "not_found", "is not found", "not supported for generatecontent")):
        return "not_found"
    if "thinking" in msg and ("400" in msg or "invalid_argument" in msg):
        return "thinking"
    return "other"


class GeminiProvider(LLMProvider):
    name = "Gemini"

    def __init__(self, api_key: str, model: str = "auto", keep_fillers: bool = False,
                 speech_language: str = "parakeet", custom_vocabulary: str = "",
                 timeout_ms: int = 12_000, log=None):
        from google import genai
        from google.genai import types
        self._types = types
        raw_model = (model or "auto").strip()
        self.auto_mode = raw_model.lower() in ("", "auto")
        self.model = "auto" if self.auto_mode else raw_model
        self.last_model = AUTO_CHAINS["proofread"][0] if self.auto_mode else self.model
        self.custom_vocabulary = custom_vocabulary
        self.log = log
        self._timeout_ms = timeout_ms
        self._ar = is_arabic_speech(speech_language)
        self._clean_think = "low" if self._ar else "minimal"
        self._proof_sys = proofread_system_prompt(keep_fillers, speech_language=speech_language)
        self._dict_sys = dictation_system_prompt(keep_fillers, speech_language=speech_language)
        self._comp_sys = compose_system_prompt()
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._rl_strikes: dict[str, int] = {}
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))

    def label(self) -> str:
        if self.auto_mode:
            return "Gemini Auto (Free Tier)"
        return f"{self.name} {self.model}".strip()

    def check(self) -> None:
        """Validate API key & primary model via models.get (0 RPD quota spent)."""
        probe = AUTO_CHAINS["proofread"][0] if self.auto_mode else self.model
        self.client.models.get(model=probe)

    def _candidates(self, task: str, prompt_chars: int = 0) -> list[str]:
        if not self.auto_mode:
            return [self.model]
        chain = list(AUTO_CHAINS.get(task, AUTO_CHAINS["proofread"]))
        if prompt_chars > _GEMMA_MAX_INPUT_CHARS:
            filtered = [m for m in chain if not _is_gemma(m)]
            if filtered:
                chain = filtered
        now = time.monotonic()
        with self._lock:
            ready = [m for m in chain if self._cooldowns.get(m, 0.0) <= now]
            cooling = sorted(
                (m for m in chain if self._cooldowns.get(m, 0.0) > now),
                key=lambda m: self._cooldowns.get(m, 0.0),
            )
        return ready + cooling

    def _mark_cooldown(self, model: str, kind: str):
        now = time.monotonic()
        with self._lock:
            if kind == "rate_limit":
                strikes = self._rl_strikes.get(model, 0) + 1
                self._rl_strikes[model] = strikes
                # 1st hit: 75s (RPM/TPM window); repeated hits: escalate for daily RPD exhaustion
                dur = 75.0 if strikes == 1 else (900.0 if strikes == 2 else 2700.0)
            elif kind == "not_found":
                dur = 3600.0
            else:
                dur = 90.0
            self._cooldowns[model] = now + dur

    def _mark_success(self, model: str):
        self.last_model = model
        with self._lock:
            self._cooldowns.pop(model, None)
            self._rl_strikes.pop(model, None)

    def _build_config(self, model: str, system_instruction: str, desired_think: str,
                      max_tokens: int, timeout_ms: int, allow_thinking: bool = True):
        t = self._types
        kwargs = dict(
            system_instruction=system_instruction,
            max_output_tokens=max_tokens,
            http_options=t.HttpOptions(timeout=timeout_ms),
        )
        if allow_thinking:
            tc = _thinking_config_for(t, model, desired_think)
            if tc is not None:
                kwargs["thinking_config"] = tc
        return t.GenerateContentConfig(**kwargs)

    def _timeout_for(self, model: str, base_timeout_ms: int) -> int:
        m = model.lower()
        if m.startswith(("gemini-4", "gemini-3.8", "gemma-4-31b")):
            return max(base_timeout_ms, 22_000)
        return base_timeout_ms

    def _call(self, task: str, contents: str, system_instruction: str,
              desired_think: str, max_tokens: int, timeout_ms: int) -> str:
        candidates = self._candidates(task, prompt_chars=len(contents) + len(system_instruction))
        last_exc = None
        for idx, model in enumerate(candidates):
            eff_timeout = self._timeout_for(model, timeout_ms)
            cfg = self._build_config(model, system_instruction, desired_think, max_tokens, eff_timeout)
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=cfg)
                self._mark_success(model)
                return strip_wrappers(resp.text or "")
            except Exception as exc:
                kind = _classify_error(exc)
                if kind == "thinking":
                    try:
                        cfg_plain = self._build_config(
                            model, system_instruction, desired_think, max_tokens, eff_timeout, allow_thinking=False
                        )
                        resp = self.client.models.generate_content(model=model, contents=contents, config=cfg_plain)
                        self._mark_success(model)
                        return strip_wrappers(resp.text or "")
                    except Exception as exc2:
                        exc = exc2
                        kind = _classify_error(exc2)
                last_exc = exc
                if kind == "auth" or not self.auto_mode or idx + 1 >= len(candidates):
                    raise
                if kind in ("rate_limit", "transient", "not_found"):
                    self._mark_cooldown(model, kind)
                    nxt = candidates[idx + 1]
                    if self.log:
                        self.log(f"[llm] {model} skipped ({kind}), falling back to {nxt}")
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        return ""

    def proofread(self, raw: str, context: str, mode: str = "proofread", upcoming: str = "") -> str:
        dictating = (mode == "dictation")
        task = "dictation" if dictating else "proofread"
        sys_inst = self._dict_sys if dictating else self._proof_sys
        max_tok = 4096 if dictating else 2048
        contents = build_proofread_prompt(
            raw, context, mode=mode, custom_vocabulary=self.custom_vocabulary, upcoming=upcoming,
        )
        return self._call(task, contents, sys_inst, self._clean_think, max_tok, self._timeout_ms)

    def compose(self, instruction: str, selected: str = "", history: list = None) -> str:
        contents = build_compose_prompt(
            instruction, selected, history, custom_vocabulary=self.custom_vocabulary,
        )
        return self._call("compose", contents, self._comp_sys, "low", 4096, 20_000)

    def generate(self, instruction: str, text: str, max_tokens: int = 8192) -> str:
        return self._call("generate", "TRANSCRIPT:\n" + text, instruction, "low", max_tokens, 60_000)
