"""LLM providers. Gemini is the default; local / OpenAI-compatible providers plug in here later."""

from .base import LLMProvider


def make_provider(settings):
    """Return (provider or None, reason). Never raises."""
    if not settings.cleanup:
        return None, "cleanup off"
    key = settings.resolved_api_key()
    if not key:
        return None, "no Gemini API key (raw text only)"
    try:
        from .gemini import GeminiProvider
    except ImportError:
        return None, "google-genai not installed (pip install google-genai)"
    try:
        return GeminiProvider(key, settings.llm_model, settings.keep_fillers), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


__all__ = ["LLMProvider", "make_provider"]
