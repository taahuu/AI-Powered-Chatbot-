"""Unit tests for configuration loading and validation."""

import pytest

from chatbot.errors import ConfigError
from config import (
    DEFAULT_MAX_CONTEXT_CHARS,
    DEFAULT_MAX_INPUT_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    load_settings,
)


def test_load_settings_missing_key_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing or empty LLM_API_KEY must raise ConfigError."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ConfigError) as exc_info:
        load_settings(dotenv_path="/dev/null")
    assert "LLM_API_KEY" in str(exc_info.value)
    assert exc_info.value.user_message is not None

    monkeypatch.setenv("LLM_API_KEY", "   ")
    with pytest.raises(ConfigError):
        load_settings(dotenv_path="/dev/null")


def test_load_settings_defaults_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid key with no optional settings applies sensible defaults."""
    monkeypatch.setenv("LLM_API_KEY", "sk-valid-key-abc")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLM_MAX_OUTPUT_TOKENS", raising=False)
    monkeypatch.delenv("MAX_CONTEXT_CHARS", raising=False)
    monkeypatch.delenv("MAX_INPUT_CHARS", raising=False)

    settings = load_settings(dotenv_path="/dev/null")
    assert settings.llm_api_key == "sk-valid-key-abc"
    assert settings.llm_base_url is None
    assert settings.llm_model == DEFAULT_MODEL
    assert settings.llm_temperature == DEFAULT_TEMPERATURE
    assert settings.llm_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert settings.max_context_chars == DEFAULT_MAX_CONTEXT_CHARS
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS


def test_load_settings_bad_numeric_values_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid numeric environment variables safely fall back to defaults."""
    monkeypatch.setenv("LLM_API_KEY", "sk-valid-key-abc")
    monkeypatch.setenv("LLM_TEMPERATURE", "not-a-float")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "invalid_int")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "xyz")
    monkeypatch.setenv("MAX_INPUT_CHARS", "bad_value")

    settings = load_settings(dotenv_path="/dev/null")
    assert settings.llm_temperature == DEFAULT_TEMPERATURE
    assert settings.llm_max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert settings.max_context_chars == DEFAULT_MAX_CONTEXT_CHARS
    assert settings.max_input_chars == DEFAULT_MAX_INPUT_CHARS


def test_load_settings_custom_values_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid custom values should be properly parsed and stored in Settings."""
    monkeypatch.setenv("LLM_API_KEY", "custom-key")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LLM_MODEL", "llama3.1")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.9")
    monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "2048")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "50000")
    monkeypatch.setenv("MAX_INPUT_CHARS", "8000")

    settings = load_settings(dotenv_path="/dev/null")
    assert settings.llm_api_key == "custom-key"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "llama3.1"
    assert settings.llm_temperature == 0.9
    assert settings.llm_max_output_tokens == 2048
    assert settings.max_context_chars == 50000
    assert settings.max_input_chars == 8000
