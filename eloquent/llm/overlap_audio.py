"""Dedicated Gemini multimodal audio provider for separating overlapping speech.

Completely independent from the text LLM provider (`GeminiProvider` / `settings.llm_model`):
- Uses its own audio-capable Gemini model chain so non-audio text models (such as Gemma 4
  or future local/OpenAI-compatible text providers) never break overlap separation.
- Encodes 16 kHz mono WAV clips strictly in RAM (no temporary files on disk).
- Supports short per-speaker voice anchors (kept in RAM for the current session) so Gemini
  matches simultaneous voices to the right speaker names even during 100% crosstalk.
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import wave
from typing import Optional

import numpy as np

from ..config import DEFAULT_OVERLAP_AUDIO_MODEL, is_arabic_speech
from ..textutil import parse_custom_vocabulary

# Ordered accuracy-first Free-Tier Gemini models verified to accept native audio/wav input
AUDIO_OVERLAP_CHAIN: tuple[str, ...] = (
    "gemini-3.6-flash",
    "gemini-4-hydrogen",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)


def samples_to_wav_bytes(samples: np.ndarray, sample_rate: int = 16000, max_sec: float = 25.0) -> bytes:
    """Encode a 1D float32 audio array into 16-bit mono PCM WAV bytes purely in memory."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return b""
    max_n = max(1, int(max_sec * sample_rate))
    if len(x) > max_n:
        x = x[:max_n]
    # Gently lift very quiet clips so faint overlapping speakers remain audible without clipping
    rms = float(np.sqrt(np.mean(x * x)))
    if 1e-4 <= rms < 0.05:
        gain = min(6.0, 0.05 / rms)
        x = np.clip(x * gain, -1.0, 1.0)
    else:
        x = np.clip(x, -1.0, 1.0)
    pcm = np.round(x * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _trim_anchor_clip(samples: np.ndarray, sample_rate: int = 16000, target_sec: float = 2.6) -> np.ndarray:
    """Extract up to *target_sec* from the center of a clean speaker clip."""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    max_n = int(target_sec * sample_rate)
    if len(x) <= max_n:
        return x
    mid = len(x) // 2
    half = max_n // 2
    return x[mid - half:mid - half + max_n]


def _classify_audio_error(exc: Exception) -> str:
    msg = f"{type(exc).__name__}: {exc}".lower()
    if any(k in msg for k in ("api_key_invalid", "api key not valid", "401", "403", "permission_denied", "unauthenticated")):
        return "auth"
    if any(k in msg for k in ("audio input modality", "modality is not enabled", "unsupported mime", "audio is not supported")):
        return "modality"
    if any(k in msg for k in ("429", "resource_exhausted", "quota", "rate limit", "too many requests")):
        return "rate_limit"
    if any(k in msg for k in ("503", "504", "unavailable", "deadline_exceeded", "timed out", "timeout", "overloaded", "internal")):
        return "transient"
    if any(k in msg for k in ("404", "not_found", "is not found", "not supported for generatecontent")):
        return "not_found"
    if "thinking" in msg and ("400" in msg or "invalid_argument" in msg):
        return "thinking"
    return "other"


def _audio_thinking_config(types, model: str):
    """Use low thinking for acoustic separation (significantly sharper than minimal on crosstalk)."""
    m = model.lower()
    if m.startswith(("gemini-4", "gemini-3")):
        return types.ThinkingConfig(thinking_level="low")
    if m.startswith("gemini-2.5"):
        return types.ThinkingConfig(thinking_budget=1024)
    return None


def _overlap_system_instruction(speech_language: str) -> str:
    lang_rule = (
        "Speech may mix Arabic (including Tunisian/Maghrebi/Levantine/Egyptian dialects) with French or English. "
        "Write Arabic speech in Arabic script and French/English words naturally as spoken; do not translate."
        if is_arabic_speech(speech_language)
        else "Transcribe each speaker in the exact language they spoke; do not translate."
    )
    return (
        "You separate simultaneous overlapping speech in a short meeting audio clip.\n"
        "You may receive short REFERENCE VOICE clips for known speakers, followed by the TARGET overlapping audio clip.\n"
        "Rules:\n"
        "1. Listen to the ENTIRE duration of the TARGET clip from start to finish (0.0s through the final second) and "
        "transcribe EVERY simultaneous voice you actually hear. Do not stop early when one speaker finishes.\n"
        "2. NEVER repeat sentences from the reference voice clips or prior transcript context, and NEVER fabricate phrases for unintelligible fragments.\n"
        "3. STRICT SPEAKER MATCHING: The first voice in the TARGET clip is NOT necessarily Speaker 1. "
        "Compare each voice's vocal timbre (pitch, tone, cadence) against the REFERENCE VOICE clips and check whether a voice "
        "continues an unfinished sentence from the recent transcript. Use a provided speaker label ONLY when that voice truly matches that speaker.\n"
        "4. UNKNOWN / NEW VOICES: If a voice in the TARGET clip does NOT match any provided reference voice clip, you MUST label it "
        "\"NEW_SPEAKER\" (or \"NEW_SPEAKER_2\" if a second distinct unknown voice appears). NEVER reuse \"Speaker 1\" or \"Speaker 2\" as generic placeholders for an unmatched voice.\n"
        f"5. {lang_rule}\n"
        "6. Return ONLY a valid JSON array of objects: "
        "[{\"speaker\": \"<matched speaker label or NEW_SPEAKER>\", \"text\": \"<verbatim separated turn>\"}] "
        "ordered by when each speaker starts talking in the TARGET clip. Omit empty turns."
    )


def _parse_separated_json(raw_text: str) -> list[dict[str, str]]:
    """Parse JSON array returned by Gemini into `[{'speaker': str, 'text': str}, ...]`."""
    s = (raw_text or "").strip()
    if not s:
        return []
    # Strip optional markdown code fences
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", s, flags=re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        for key in ("turns", "speakers", "segments", "transcript", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        return []

    out: list[dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        spk = str(item.get("speaker") or item.get("name") or "").strip()
        txt = re.sub(r"\s+", " ", str(item.get("text") or item.get("transcript") or "").strip())
        if not txt:
            continue
        out.append({"speaker": spk, "text": txt})
    return out


def _next_speaker_number(known_speakers: list[tuple[str, str]], hint: int = 0) -> int:
    if hint and hint >= 1:
        return int(hint)
    max_n = 0
    for cid, _ in known_speakers:
        m = re.match(r"^speaker\s+(\d+)$", (cid or "").strip().lower())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return max_n + 1


def _resolve_speaker_id(
    raw_label: str,
    known_speakers: list[tuple[str, str]],
    default_speaker: str = "",
    new_speaker_map: Optional[dict[str, str]] = None,
    next_num_ref: Optional[list[int]] = None,
) -> str:
    """Map Gemini's returned speaker label back to an existing canonical `Speaker N` ID when matched."""
    lbl = (raw_label or "").strip()
    low = lbl.lower()
    if not lbl:
        return default_speaker or (known_speakers[0][0] if known_speakers else "Speaker 1")

    # Map explicit unknown/new speaker sentinels to sequential unused Speaker N IDs
    if re.match(r"^(?:new[\s_-]*speaker|unknown(?:[\s_-]*speaker)?|other(?:[\s_-]*speaker)?)(?:[\s_-]*\d+)?$", low):
        key = re.sub(r"[\s_-]+", "_", low)
        if new_speaker_map is not None and key in new_speaker_map:
            return new_speaker_map[key]
        if next_num_ref is None:
            next_num_ref = [_next_speaker_number(known_speakers)]
        cid = f"Speaker {next_num_ref[0]}"
        next_num_ref[0] += 1
        if new_speaker_map is not None:
            new_speaker_map[key] = cid
        return cid

    lookup: dict[str, str] = {}
    for cid, disp in known_speakers:
        if cid:
            lookup[cid.strip().lower()] = cid
        if disp:
            lookup[disp.strip().lower()] = cid
        if cid and disp and disp != cid:
            lookup[f"{disp.strip()} ({cid.strip()})".lower()] = cid

    if low in lookup:
        return lookup[low]

    # Handle patterns like "Sarah (Speaker 1)" or "Speaker 1 - Sarah"
    m_paren = re.search(r"\(([^)]+)\)", lbl)
    if m_paren and m_paren.group(1).strip().lower() in lookup:
        return lookup[m_paren.group(1).strip().lower()]
    head = re.split(r"\s*[\(\-:]\s*", lbl, maxsplit=1)[0].strip().lower()
    if head in lookup:
        return lookup[head]

    # Normalize bare "speaker 3" -> "Speaker 3"
    m_spk = re.match(r"^speaker\s+(\d+)$", low)
    if m_spk:
        return f"Speaker {int(m_spk.group(1))}"
    return lbl


class GeminiOverlapAudioProvider:
    """Dedicated Gemini Audio provider for separating simultaneous overlapping speakers."""

    name = "Gemini Audio"

    def __init__(self, api_key: str, model: str = DEFAULT_OVERLAP_AUDIO_MODEL,
                 speech_language: str = "parakeet", custom_vocabulary: str = "",
                 timeout_ms: int = 14_000, log=None):
        from google import genai
        from google.genai import types

        self._types = types
        raw_model = (model or DEFAULT_OVERLAP_AUDIO_MODEL).strip()
        self.auto_mode = raw_model.lower() in ("", "auto")
        self.model = "auto" if self.auto_mode else raw_model
        self.last_model = AUDIO_OVERLAP_CHAIN[0] if self.auto_mode else self.model
        self.speech_language = speech_language
        self.custom_vocabulary = custom_vocabulary
        self.timeout_ms = int(timeout_ms)
        self.log = log
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._rl_strikes: dict[str, int] = {}
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=self.timeout_ms))

    def label(self) -> str:
        if self.auto_mode:
            return "Gemini Audio (Auto)"
        return f"{self.name} {self.model}".strip()

    def _candidates(self) -> list[str]:
        if not self.auto_mode:
            return [self.model]
        now = time.monotonic()
        with self._lock:
            ready = [m for m in AUDIO_OVERLAP_CHAIN if self._cooldowns.get(m, 0.0) <= now]
            cooling = sorted(
                (m for m in AUDIO_OVERLAP_CHAIN if self._cooldowns.get(m, 0.0) > now),
                key=lambda m: self._cooldowns.get(m, 0.0),
            )
        return ready + cooling

    def _mark_cooldown(self, model: str, kind: str):
        now = time.monotonic()
        with self._lock:
            if kind == "rate_limit":
                strikes = self._rl_strikes.get(model, 0) + 1
                self._rl_strikes[model] = strikes
                dur = 75.0 if strikes == 1 else (900.0 if strikes == 2 else 2700.0)
            elif kind in ("not_found", "modality"):
                dur = 3600.0
            else:
                dur = 90.0
            self._cooldowns[model] = now + dur

    def _mark_success(self, model: str):
        self.last_model = model
        with self._lock:
            self._cooldowns.pop(model, None)
            self._rl_strikes.pop(model, None)

    def _build_config(self, model: str, system_instruction: str, allow_thinking: bool = True):
        t = self._types
        eff_timeout = max(self.timeout_ms, 16_000) if model.lower().startswith(("gemini-4", "gemini-3.8")) else self.timeout_ms
        kwargs = dict(
            system_instruction=system_instruction,
            max_output_tokens=1536,
            response_mime_type="application/json",
            http_options=t.HttpOptions(timeout=eff_timeout),
        )
        if allow_thinking:
            tc = _audio_thinking_config(t, model)
            if tc is not None:
                kwargs["thinking_config"] = tc
        return t.GenerateContentConfig(**kwargs)

    def separate_overlap(
        self,
        target_samples: np.ndarray,
        *,
        sample_rate: int = 16000,
        anchors: Optional[list[tuple[str, str, np.ndarray]]] = None,
        known_speakers: Optional[list[tuple[str, str]]] = None,
        context: str = "",
        local_text: str = "",
        default_speaker: str = "",
        prev_speaker: str = "",
        next_speaker_num: int = 0,
        custom_vocabulary: Optional[str] = None,
        speech_language: Optional[str] = None,
    ) -> list[dict[str, str]]:
        """Send an overlapping audio clip (+ optional speaker voice anchors) to Gemini Audio.

        Returns `[{'speaker': canonical_id, 'text': separated_text}, ...]`.
        """
        target_wav = samples_to_wav_bytes(target_samples, sample_rate=sample_rate, max_sec=25.0)
        if not target_wav:
            return []

        lang = speech_language if speech_language is not None else self.speech_language
        vocab = custom_vocabulary if custom_vocabulary is not None else self.custom_vocabulary
        sys_inst = _overlap_system_instruction(lang)

        t = self._types
        contents: list = []

        # Attach up to 4 clean speaker voice anchors (kept strictly in RAM)
        spk_list: list[tuple[str, str]] = list(known_speakers or ())
        seen_cids = {cid for cid, _ in spk_list}
        anchor_labels: list[str] = []
        for cid, disp, clip in (anchors or ())[:4]:
            trimmed = _trim_anchor_clip(clip, sample_rate=sample_rate, target_sec=2.6)
            wav_b = samples_to_wav_bytes(trimmed, sample_rate=sample_rate, max_sec=3.0)
            if not wav_b:
                continue
            if cid and cid not in seen_cids:
                spk_list.append((cid, disp or cid))
                seen_cids.add(cid)
            lbl = disp or cid
            anchor_labels.append(f'"{lbl}"')
            prev_hint = " (speaking right before this clip)" if (prev_speaker and cid == prev_speaker) else ""
            contents.append(f'REFERENCE VOICE for "{lbl}"{prev_hint}:')
            contents.append(t.Part.from_bytes(data=wav_b, mime_type="audio/wav"))

        meta_lines: list[str] = []
        if anchor_labels:
            meta_lines.append(
                "Allowed matched speaker labels (ONLY use if the voice matches that reference clip): "
                + ", ".join(anchor_labels)
                + '.\nFor ANY voice in the TARGET clip that does NOT match one of those reference voices, label it "NEW_SPEAKER".'
            )
        elif spk_list:
            names = [f'"{disp or cid}"' for cid, disp in spk_list]
            meta_lines.append(
                "Known speakers so far: " + ", ".join(names)
                + '. Use "NEW_SPEAKER" for any additional voice.'
            )
        if vocab:
            terms, pairs = parse_custom_vocabulary(vocab)
            if terms:
                meta_lines.append("Important names/terms: " + ", ".join(terms[:30]))
            if pairs:
                meta_lines.append("Preferred spellings: " + "; ".join(f"{w} -> {r}" for w, r in pairs[:20]))
        if context and context.strip():
            meta_lines.append("Recent transcript before this overlap:\n" + context.strip()[-700:])
        meta_lines.append("TARGET overlapping audio clip to transcribe and separate (listen from 0.0s to the very end):")
        contents.append("\n\n".join(meta_lines))
        contents.append(t.Part.from_bytes(data=target_wav, mime_type="audio/wav"))

        candidates = self._candidates()
        last_exc: Optional[Exception] = None
        raw_reply = ""
        for idx, model in enumerate(candidates):
            cfg = self._build_config(model, sys_inst, allow_thinking=True)
            try:
                resp = self.client.models.generate_content(model=model, contents=contents, config=cfg)
                self._mark_success(model)
                raw_reply = resp.text or ""
                break
            except Exception as exc:
                kind = _classify_audio_error(exc)
                if kind == "thinking":
                    try:
                        cfg_plain = self._build_config(model, sys_inst, allow_thinking=False)
                        resp = self.client.models.generate_content(model=model, contents=contents, config=cfg_plain)
                        self._mark_success(model)
                        raw_reply = resp.text or ""
                        break
                    except Exception as exc2:
                        exc = exc2
                        kind = _classify_audio_error(exc2)
                last_exc = exc
                if kind == "modality" and not self.auto_mode:
                    raise RuntimeError(
                        f"Model {model} does not support audio input. Pick Auto (Gemini Audio) or a Gemini Flash model."
                    ) from exc
                if kind == "auth" or not self.auto_mode or idx + 1 >= len(candidates):
                    raise
                if kind in ("rate_limit", "transient", "not_found", "modality"):
                    self._mark_cooldown(model, kind)
                    nxt = candidates[idx + 1]
                    if self.log:
                        self.log(f"[overlap-audio] {model} skipped ({kind}), falling back to {nxt}")
                    continue
                raise
        else:
            if last_exc is not None:
                raise last_exc
            return []

        parsed = _parse_separated_json(raw_reply)
        if not parsed:
            return []

        # Map speaker names back to canonical IDs and merge consecutive same-speaker items
        new_spk_map: dict[str, str] = {}
        next_num_ref = [_next_speaker_number(spk_list, hint=next_speaker_num)]
        resolved: list[dict[str, str]] = []
        for item in parsed:
            cid = _resolve_speaker_id(
                item["speaker"],
                spk_list,
                default_speaker=default_speaker,
                new_speaker_map=new_spk_map,
                next_num_ref=next_num_ref,
            )
            txt = item["text"].strip()
            if not txt:
                continue
            if resolved and resolved[-1]["speaker"] == cid:
                resolved[-1]["text"] = f"{resolved[-1]['text']} {txt}".strip()
            else:
                resolved.append({"speaker": cid, "text": txt})
        return resolved
