"""Application configuration and environment variable loading."""

from dataclasses import dataclass
import logging
import os
from typing import Optional
from dotenv import load_dotenv

from chatbot.errors import ConfigError

logger = logging.getLogger(__name__)

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.5
DEFAULT_MAX_OUTPUT_TOKENS: int = 1024
DEFAULT_MAX_CONTEXT_CHARS: int = 24000
DEFAULT_MAX_INPUT_CHARS: int = 4000
DEFAULT_SESSIONS_DIR: str = "data/sessions"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings container.

    Attributes:
        llm_api_key: API key for the LLM service.
        llm_base_url: Optional custom base URL for OpenAI-compatible endpoints.
        llm_model: Target model name.
        llm_temperature: Sampling temperature (0.0 to 2.0).
        llm_max_output_tokens: Maximum tokens in completion output.
        max_context_chars: Context window character budget for history trimming.
        max_input_chars: Maximum character limit for user input.
        sessions_dir: Directory path for persisting conversation sessions.
    """

    llm_api_key: str
    llm_base_url: Optional[str] = None
    llm_model: str = DEFAULT_MODEL
    llm_temperature: float = DEFAULT_TEMPERATURE
    llm_max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    sessions_dir: str = DEFAULT_SESSIONS_DIR


def _parse_float(var_name: str, raw_value: Optional[str], default: float) -> float:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        logger.warning(
            "Invalid float value for %s: '%s'. Falling back to default: %s",
            var_name,
            raw_value,
            default,
        )
        return default


def _parse_int(var_name: str, raw_value: Optional[str], default: int) -> int:
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        logger.warning(
            "Invalid integer value for %s: '%s'. Falling back to default: %s",
            var_name,
            raw_value,
            default,
        )
        return default


def load_settings(dotenv_path: Optional[str] = None) -> Settings:
    """Load configuration from environment variables and .env file.

    Args:
        dotenv_path: Optional specific path to a .env file.

    Returns:
        Settings: Validated settings object.

    Raises:
        ConfigError: If LLM_API_KEY is missing or empty.
    """
    load_dotenv(dotenv_path=dotenv_path)

    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise ConfigError(
            message="LLM_API_KEY is missing or blank.",
            user_message="LLM_API_KEY is not configured. Please create a .env file from .env.example and set your API key.",
        )

    base_url_raw = os.getenv("LLM_BASE_URL", "").strip()
    base_url = base_url_raw if base_url_raw else None

    model = os.getenv("LLM_MODEL", "").strip() or DEFAULT_MODEL
    temperature = _parse_float("LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE"), DEFAULT_TEMPERATURE)
    max_output_tokens = _parse_int("LLM_MAX_OUTPUT_TOKENS", os.getenv("LLM_MAX_OUTPUT_TOKENS"), DEFAULT_MAX_OUTPUT_TOKENS)
    max_context_chars = _parse_int("MAX_CONTEXT_CHARS", os.getenv("MAX_CONTEXT_CHARS"), DEFAULT_MAX_CONTEXT_CHARS)
    max_input_chars = _parse_int("MAX_INPUT_CHARS", os.getenv("MAX_INPUT_CHARS"), DEFAULT_MAX_INPUT_CHARS)
    sessions_dir = os.getenv("SESSIONS_DIR", "").strip() or DEFAULT_SESSIONS_DIR

    return Settings(
        llm_api_key=api_key,
        llm_base_url=base_url,
        llm_model=model,
        llm_temperature=temperature,
        llm_max_output_tokens=max_output_tokens,
        max_context_chars=max_context_chars,
        max_input_chars=max_input_chars,
        sessions_dir=sessions_dir,
    )
