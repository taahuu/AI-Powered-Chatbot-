"""Unit tests for LLMClient: error mapping, retry logic, and stream chunk handling."""

from unittest.mock import MagicMock, patch
import openai
import pytest

from chatbot.errors import (
    AuthenticationFailedError,
    ConnectionFailedError,
    LLMError,
    ModelError,
    RateLimitedError,
)
from chatbot.llm_client import LLMClient
from config import Settings


def _create_mock_chunk(content: str | None = None, empty_choices: bool = False) -> MagicMock:
    """Helper to build a mock OpenAI chat completion chunk."""
    chunk = MagicMock()
    if empty_choices:
        chunk.choices = []
    else:
        choice = MagicMock()
        delta = MagicMock()
        delta.content = content
        choice.delta = delta
        chunk.choices = [choice]
    return chunk


def _make_openai_error(cls: type[Exception], message: str = "test error") -> Exception:
    """Construct a real, raiseable OpenAI exception instance for tests."""
    if cls is openai.APIConnectionError:
        return openai.APIConnectionError(request=MagicMock())
    if cls is openai.APITimeoutError:
        return openai.APITimeoutError(request=MagicMock())
    if cls is openai.APIError:
        return openai.APIError(message, request=MagicMock(), body=None)
    mock_resp = MagicMock()
    mock_resp.request = MagicMock()
    mock_resp.status_code = 400
    mock_resp.headers = {}
    return cls(message=message, response=mock_resp, body=None)


@pytest.mark.parametrize(
    "sdk_error_cls,expected_custom_err",
    [
        (openai.AuthenticationError, AuthenticationFailedError),
        (openai.RateLimitError, RateLimitedError),
        (openai.APIConnectionError, ConnectionFailedError),
        (openai.APITimeoutError, ConnectionFailedError),
        (openai.NotFoundError, ModelError),
        (openai.BadRequestError, ModelError),
        (openai.APIError, LLMError),
    ],
)
def test_map_exception_correct_mapping(
    sdk_error_cls: type[Exception], expected_custom_err: type[Exception]
) -> None:
    """Verify SDK exceptions map to the corresponding custom ChatbotError."""
    sdk_err = _make_openai_error(sdk_error_cls)
    mapped = LLMClient._map_exception(sdk_err)
    assert isinstance(mapped, expected_custom_err)
    assert mapped.user_message is not None


def test_retry_on_rate_limit_and_succeeds(test_settings: Settings) -> None:
    """Verify rate-limit errors trigger retry before first chunk and succeed."""
    mock_openai = MagicMock()
    good_chunk1 = _create_mock_chunk("Hello")
    good_chunk2 = _create_mock_chunk(" World")

    rate_limit_err = _make_openai_error(openai.RateLimitError, "rate limit")
    mock_openai.chat.completions.create.side_effect = [
        rate_limit_err,
        [good_chunk1, good_chunk2],
    ]

    client = LLMClient(settings=test_settings, client=mock_openai, retry_delays=(0.01, 0.02))

    with patch("time.sleep") as mock_sleep:
        chunks = list(client.stream([{"role": "user", "content": "Hi"}]))
        assert chunks == ["Hello", " World"]
        assert mock_openai.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(0.01)


def test_no_retry_on_authentication_error(test_settings: Settings) -> None:
    """Authentication errors must never be retried and fail immediately."""
    mock_openai = MagicMock()
    auth_err = _make_openai_error(openai.AuthenticationError, "invalid api key")
    mock_openai.chat.completions.create.side_effect = auth_err

    client = LLMClient(settings=test_settings, client=mock_openai, retry_delays=(0.01, 0.02))

    with patch("time.sleep") as mock_sleep:
        with pytest.raises(AuthenticationFailedError):
            list(client.stream([{"role": "user", "content": "Hi"}]))
        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()


def test_empty_delta_chunks_skipped(test_settings: Settings) -> None:
    """Chunks with empty choices or None/empty delta content should be skipped."""
    mock_openai = MagicMock()
    chunks_sequence = [
        _create_mock_chunk(empty_choices=True),
        _create_mock_chunk(content=None),
        _create_mock_chunk(content=""),
        _create_mock_chunk(content="Actual"),
        _create_mock_chunk(empty_choices=True),
        _create_mock_chunk(content=" text"),
        _create_mock_chunk(content=None),
    ]
    mock_openai.chat.completions.create.return_value = chunks_sequence

    client = LLMClient(settings=test_settings, client=mock_openai)
    result = list(client.stream([{"role": "user", "content": "Hi"}]))
    assert result == ["Actual", " text"]


def test_mid_stream_failure_is_not_retried(test_settings: Settings) -> None:
    """Errors occurring mid-stream (after first chunk yielded) must not trigger retry."""
    mock_openai = MagicMock()

    def chunk_generator():
        yield _create_mock_chunk(content="First part")
        raise _make_openai_error(openai.RateLimitError, "interrupted mid stream")

    mock_openai.chat.completions.create.return_value = chunk_generator()

    client = LLMClient(settings=test_settings, client=mock_openai, retry_delays=(0.01, 0.02))

    with patch("time.sleep") as mock_sleep:
        stream = client.stream([{"role": "user", "content": "Hi"}])
        first = next(stream)
        assert first == "First part"

        with pytest.raises(RateLimitedError):
            next(stream)

        # No retry should have been attempted
        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()
