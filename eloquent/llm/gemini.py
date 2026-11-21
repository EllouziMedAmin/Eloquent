"""Google Gemini provider (default: gemini-3.1-flash-lite, free key from Google AI Studio)."""

from .base import LLMProvider
from .prompts import (build_compose_prompt, build_proofread_prompt, compose_system_prompt,
                      dictation_system_prompt, proofread_system_prompt)
from ..config import is_arabic_speech
from ..textutil import strip_wrappers


class GeminiProvider(LLMProvider):
    name = "Gemini"

    def __init__(self, api_key: str, model: str, keep_fillers: bool,
                 speech_language: str = "parakeet", custom_vocabulary: str = "",
                 timeout_ms: int = 12_000):
        from google import genai
        from google.genai import types
        self._types = types
        self.model = model
        self.custom_vocabulary = custom_vocabulary
        ar = is_arabic_speech(speech_language)
        clean_think = "low" if ar else "minimal"
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        self.proof_cfg = types.GenerateContentConfig(
            system_instruction=proofread_system_prompt(keep_fillers, speech_language=speech_language),
            thinking_config=types.ThinkingConfig(thinking_level=clean_think),
            max_output_tokens=2048,
            # Gemini 3 works best at its default temperature, so we leave it alone.
        )
        self.dict_cfg = types.GenerateContentConfig(
            system_instruction=dictation_system_prompt(keep_fillers, speech_language=speech_language),
            thinking_config=types.ThinkingConfig(thinking_level=clean_think),
            max_output_tokens=4096,
        )
        self.comp_cfg = types.GenerateContentConfig(
            system_instruction=compose_system_prompt(),
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            max_output_tokens=4096,
            http_options=types.HttpOptions(timeout=25_000),
        )

    def check(self) -> None:
        t = self._types
        self.client.models.generate_content(
            model=self.model, contents="ping",
            config=t.GenerateContentConfig(
                max_output_tokens=16, thinking_config=t.ThinkingConfig(thinking_level="minimal")))

    def proofread(self, raw: str, context: str, mode: str = "proofread", upcoming: str = "") -> str:
        cfg = self.dict_cfg if mode == "dictation" else self.proof_cfg
        resp = self.client.models.generate_content(
            model=self.model,
            contents=build_proofread_prompt(
                raw, context, mode=mode, custom_vocabulary=self.custom_vocabulary, upcoming=upcoming,
            ),
            config=cfg,
        )
        return strip_wrappers(resp.text or "")

    def compose(self, instruction: str, selected: str = "", history: list = None) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=build_compose_prompt(instruction, selected, history, custom_vocabulary=self.custom_vocabulary),
            config=self.comp_cfg,
        )
        return strip_wrappers(resp.text or "")

    def generate(self, instruction: str, text: str, max_tokens: int = 8192) -> str:
        t = self._types
        resp = self.client.models.generate_content(
            model=self.model, contents="TRANSCRIPT:\n" + text,
            config=t.GenerateContentConfig(
                system_instruction=instruction,
                max_output_tokens=max_tokens,
                thinking_config=t.ThinkingConfig(thinking_level="low"),
                http_options=t.HttpOptions(timeout=60_000),
            ))
        return strip_wrappers(resp.text or "")
