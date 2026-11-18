"""LLM client interface, OpenAI SDK integration, error mapping, and streaming logic."""

from abc import ABC, abstractmethod
import logging
import time
from typing import Any, Iterator, Optional, Sequence
import openai

from chatbot.errors import (
    AuthenticationFailedError,
    ChatbotError,
    ConnectionFailedError,
    LLMError,
    ModelError,
    RateLimitedError,
)

logger = logging.getLogger(__name__)


class BaseLLM(ABC):
    """Abstract base class for LLM completion providers."""

    @abstractmethod
    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Stream response tokens for the given message list.

        Args:
            messages: Formatted conversation message dictionaries.

        Yields:
            Token strings as they arrive from the model.
        """
        pass


class LLMClient(BaseLLM):
    """Wrapper around the OpenAI SDK supporting streaming, retry backoff, and error mapping."""

    def __init__(
        self,
        settings: Any,
        client: Optional[openai.OpenAI] = None,
        retry_delays: tuple[float, ...] = (1.0, 2.0),
    ) -> None:
        self.model: str = getattr(settings, "llm_model", "gpt-4o-mini")
        self.temperature: float = getattr(settings, "llm_temperature", 0.5)
        self.max_tokens: int = getattr(settings, "llm_max_output_tokens", 1024)
        self._retry_delays: tuple[float, ...] = retry_delays

        if client is not None:
            self.client = client
        else:
            api_key = getattr(settings, "llm_api_key", "")
            base_url = getattr(settings, "llm_base_url", None)
            client_kwargs: dict[str, object] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**client_kwargs)

    @staticmethod
    def _map_exception(exc: Exception) -> ChatbotError:
        """Map SDK and network exceptions to application-specific ChatbotError subclasses."""
        if isinstance(exc, openai.AuthenticationError):
            return AuthenticationFailedError(
                message=f"Authentication failed: {exc}",
                user_message="Authentication failed. Please check your LLM_API_KEY in the .env file.",
            )
        if isinstance(exc, openai.RateLimitError):
            return RateLimitedError(
                message=f"Rate limit exceeded: {exc}",
                user_message="Rate limit reached. Please wait a moment before sending another message.",
            )
        if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
            return ConnectionFailedError(
                message=f"Connection/timeout error: {exc}",
                user_message="Failed to connect to the LLM service. Please check your network connection or base URL.",
            )
        if isinstance(exc, (openai.NotFoundError, openai.BadRequestError)):
            return ModelError(
                message=f"Model error or bad request: {exc}",
                user_message="The specified model was not found or the request was invalid. Please check your LLM_MODEL configuration.",
            )
        if isinstance(exc, ChatbotError):
            return exc
        return LLMError(
            message=f"Unexpected LLM error: {exc}",
            user_message="An error occurred while communicating with the LLM service. Please try again.",
        )

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Stream completion chunks with retry on transient errors before first chunk.

        Args:
            messages: List of message dictionaries.

        Yields:
            Text content chunks from delta responses.
        """
        attempt = 0
        max_retries = len(self._retry_delays)
        delays = self._retry_delays

        first_chunk_content: Optional[str] = None
        stream_iter = None

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=list(messages),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                stream_iter = iter(response)
                # Read chunks until first non-empty content chunk is found
                for chunk in stream_iter:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            first_chunk_content = delta.content
                            break
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as exc:
                if attempt < max_retries:
                    delay = delays[attempt]
                    logger.warning(
                        "Transient error on attempt %d: %s. Retrying in %ss...",
                        attempt + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    attempt += 1
                else:
                    logger.error(
                        "Exhausted retries on attempt %d: %s",
                        attempt + 1,
                        exc,
                        exc_info=True,
                    )
                    raise self._map_exception(exc) from exc
            except Exception as exc:
                logger.error("LLM call failed with non-retryable error: %s", exc, exc_info=True)
                raise self._map_exception(exc) from exc

        if first_chunk_content is not None:
            yield first_chunk_content

        if stream_iter is not None:
            try:
                for chunk in stream_iter:
                    if chunk.choices:
                        delta = chunk.choices[0].delta
                        if delta and getattr(delta, "content", None):
                            yield delta.content
            except Exception as exc:
                logger.error("Stream interrupted mid-generation: %s", exc, exc_info=True)
                raise self._map_exception(exc) from exc
