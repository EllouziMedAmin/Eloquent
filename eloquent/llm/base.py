"""LLM provider interface.

A provider only has to turn text into text. Proofreading during transcription
uses `proofread()`; post-session actions (summary, bullet points, report...)
will use `generate()`.
"""


class LLMProvider:
    name = "LLM"
    model = ""

    def check(self) -> None:
        """Cheap request that raises if the key/model/endpoint is not usable."""
        raise NotImplementedError

    def proofread(self, raw: str, context: str) -> str:
        """Return the corrected text. Raise on network/quota errors."""
        raise NotImplementedError

    def generate(self, instruction: str, text: str, max_tokens: int = 4096) -> str:
        """Free-form generation on a transcript (summary, bullet points, report...)."""
        raise NotImplementedError

    def label(self) -> str:
        return f"{self.name} {self.model}".strip()
