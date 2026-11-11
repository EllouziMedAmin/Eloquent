"""Google Gemini provider (default: gemini-3.1-flash-lite, free key from Google AI Studio)."""

from .base import LLMProvider
from .prompts import build_proofread_prompt, proofread_system_prompt
from ..textutil import strip_wrappers


class GeminiProvider(LLMProvider):
    name = "Gemini"

    def __init__(self, api_key: str, model: str, keep_fillers: bool, timeout_ms: int = 12_000):
        from google import genai
        from google.genai import types
        self._types = types
        self.model = model
        self.client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
        self.proof_cfg = types.GenerateContentConfig(
            system_instruction=proofread_system_prompt(keep_fillers),
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            max_output_tokens=2048,
            # Gemini 3 works best at its default temperature, so we leave it alone.
        )

    def check(self) -> None:
        t = self._types
        self.client.models.generate_content(
            model=self.model, contents="ping",
            config=t.GenerateContentConfig(
                max_output_tokens=16, thinking_config=t.ThinkingConfig(thinking_level="minimal")))

    def proofread(self, raw: str, context: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model, contents=build_proofread_prompt(raw, context), config=self.proof_cfg)
        return strip_wrappers(resp.text or "")

    def generate(self, instruction: str, text: str, max_tokens: int = 4096) -> str:
        t = self._types
        resp = self.client.models.generate_content(
            model=self.model, contents="TRANSCRIPT:\n" + text,
            config=t.GenerateContentConfig(
                system_instruction=instruction, max_output_tokens=max_tokens,
                thinking_config=t.ThinkingConfig(thinking_level="low")))
        return (resp.text or "").strip()
