"""
Runtime configuration for every agent, loaded from environment variables / .env.

OpenAI is the default LLM provider for local development and testing, matching
Project 2's convention. Swapping providers later only requires changing
LLM_MODEL/LLM_SUMMARY_MODEL here -- no agent code changes, since agents never
reference "OpenAI" directly, only these settings (see llm_client.py).
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (text generation) --------------------------------------------------
    openai_api_key: str | None = None
    # PydanticAI model string, used by the structured-output generation agents.
    llm_model: str = "openai:gpt-4o-mini"
    # LiteLLM model string, used by simpler plain-text generation calls.
    llm_text_model: str = "gpt-4o-mini"

    # --- TTS (Piper, runs locally) -----------------------------------------------
    piper_voices_dir: str = "configs/voices"   # directory of downloaded .onnx voice models
    default_piper_voice: str = "en_US-amy-medium"

    # --- Output -------------------------------------------------------------------
    dataset_output_dir: str = "datasets"

    # --- Reproducibility ------------------------------------------------------------
    default_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton. Cached so .env is only parsed once."""
    return Settings()


def dataset_output_path() -> Path:
    """Resolved output directory for benchmark releases, created on first use."""
    path = Path(get_settings().dataset_output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
