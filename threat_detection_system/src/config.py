"""
Runtime configuration for every agent, loaded from environment variables / .env.

OpenAI is the default LLM provider for local development and testing (per project
convention). Swapping providers later (e.g. to Anthropic, or a self-hosted model)
only requires changing LLM_MODEL / LLM_SUMMARY_MODEL here — no agent code changes,
since agents never reference "OpenAI" directly, only these settings.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (text reasoning) ---------------------------------------------------
    openai_api_key: str | None = None
    # PydanticAI model string, used by the structured-output detection/decision agents.
    llm_model: str = "openai:gpt-4o-mini"
    # LiteLLM model string, used by the correlation agents' plain-text summaries.
    llm_summary_model: str = "gpt-4o-mini"

    # --- Speech-to-Text (faster-whisper, local) ---------------------------------
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"

    # --- Speaker Diarization (pyannote.audio, gated HF model) -------------------
    # "HUGGINGFACEHUB_API_TOKEN" is the conventional env var name used by the
    # huggingface_hub / transformers ecosystem, so it's accepted alongside our
    # own "HUGGINGFACE_TOKEN" name.
    huggingface_token: str | None = Field(default=None, validation_alias="HUGGINGFACEHUB_API_TOKEN")
    diarization_model: str = "pyannote/speaker-diarization-3.1"

    # --- Emotion Recognition (SpeechBrain / Hugging Face audio models) ----------
    emotion_model: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so .env is only parsed once."""
    return Settings()
