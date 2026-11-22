"""LLM providers. Gemini is the default; local / OpenAI-compatible providers plug in here later."""

from .base import LLMProvider
from ..config import DEFAULT_OVERLAP_AUDIO_MODEL, normalize_overlap_mode


def make_provider(settings, force: bool = False):
    """Return (provider or None, reason). Never raises."""
    if not settings.cleanup and not force:
        return None, "cleanup off"
    key = settings.resolved_api_key()
    if not key:
        return None, "no Gemini API key (raw text only)"
    try:
        from .gemini import GeminiProvider
    except ImportError:
        return None, "google-genai not installed (pip install google-genai)"
    try:
        speech_lang = settings.resolved_speech_language() if hasattr(settings, "resolved_speech_language") else getattr(settings, "speech_language", "parakeet")
        vocab = getattr(settings, "custom_vocabulary", "")
        return GeminiProvider(key, settings.llm_model, settings.keep_fillers, speech_language=speech_lang, custom_vocabulary=vocab), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


def make_overlap_provider(settings, force: bool = False):
    """Return (GeminiOverlapAudioProvider or None, reason). Independent of text LLM model."""
    mode = normalize_overlap_mode(getattr(settings, "overlap_mode", "off"))
    if mode == "off" and not force:
        return None, "overlap separation off"
    key = settings.resolved_api_key()
    if not key:
        return None, "no Gemini API key (add one in Settings -> AI & Polishing)"
    try:
        from .overlap_audio import GeminiOverlapAudioProvider
    except ImportError:
        return None, "google-genai not installed (pip install google-genai)"
    try:
        speech_lang = (
            settings.resolved_speech_language()
            if hasattr(settings, "resolved_speech_language")
            else getattr(settings, "speech_language", "parakeet")
        )
        vocab = getattr(settings, "custom_vocabulary", "")
        audio_model = getattr(settings, "overlap_audio_model", DEFAULT_OVERLAP_AUDIO_MODEL)
        return GeminiOverlapAudioProvider(
            key, model=audio_model, speech_language=speech_lang, custom_vocabulary=vocab,
        ), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)[:140]}"


__all__ = ["LLMProvider", "make_provider", "make_overlap_provider"]
