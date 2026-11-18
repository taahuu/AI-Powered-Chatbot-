"""Unit tests for prompt construction, input cleaning, and system prompt formatting."""

import pytest

from chatbot.errors import InvalidInputError
from chatbot.models import Message
from chatbot.prompts import PromptBuilder, clean_user_input, SYSTEM_PROMPT


def test_clean_user_input_strip_and_collapse_lines() -> None:
    """Test whitespace stripping and collapse of runs of 3+ blank lines."""
    raw = "   \n\n\nHello\n\n\n\n\nWorld\n\n\n   "
    cleaned = clean_user_input(raw, max_chars=100)
    assert cleaned == "Hello\n\nWorld"


def test_clean_user_input_preserves_single_blank_line() -> None:
    """Paragraph breaks with a single blank line should be preserved."""
    raw = "Paragraph 1\n\nParagraph 2"
    assert clean_user_input(raw) == "Paragraph 1\n\nParagraph 2"


def test_clean_user_input_empty_raises_error() -> None:
    """Empty or whitespace-only inputs must raise InvalidInputError."""
    with pytest.raises(InvalidInputError):
        clean_user_input("")

    with pytest.raises(InvalidInputError):
        clean_user_input("    \n\t  ")


def test_clean_user_input_over_length_raises_error() -> None:
    """Inputs exceeding max_chars must raise InvalidInputError."""
    with pytest.raises(InvalidInputError):
        clean_user_input("a" * 50, max_chars=40)


def test_clean_user_input_non_string_raises_error() -> None:
    """Non-string inputs must raise InvalidInputError."""
    with pytest.raises(InvalidInputError):
        clean_user_input(None)  # type: ignore[arg-type]


def test_prompt_builder_order_of_messages() -> None:
    """Verify prompt builder output order: system prompt, history, user input."""
    builder = PromptBuilder()
    history = [
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi! How can I help?"),
    ]
    messages = builder.build(history_messages=history, user_input="Explain quantum computing")

    assert len(messages) == 4
    # 1. System prompt first
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    # 2. History in order
    assert messages[1] == {"role": "user", "content": "Hello"}
    assert messages[2] == {"role": "assistant", "content": "Hi! How can I help?"}
    # 3. Cleaned user input last
    assert messages[3] == {"role": "user", "content": "Explain quantum computing"}


def test_prompt_builder_with_retrieved_context() -> None:
    """Verify retrieved context block is included when provided."""
    builder = PromptBuilder()
    history = [Message(role="user", content="Previous query")]
    retrieved = "Doc 1 content\nDoc 2 content"

    messages = builder.build(
        history_messages=history,
        user_input="Current query",
        retrieved_context=retrieved,
    )

    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    # Second message is the context wrapper
    assert messages[1]["role"] == "system"
    assert "<context>" in messages[1]["content"]
    assert "</context>" in messages[1]["content"]
    assert "Doc 1 content" in messages[1]["content"]
    assert "prefer this context" in messages[1]["content"].lower()
    # History
    assert messages[2] == {"role": "user", "content": "Previous query"}
    # Current user query
    assert messages[3] == {"role": "user", "content": "Current query"}


def test_prompt_builder_without_retrieved_context() -> None:
    """Verify context block is omitted when retrieved_context is None or empty."""
    builder = PromptBuilder()

    messages_none = builder.build(history_messages=[], user_input="Hi", retrieved_context=None)
    assert len(messages_none) == 2
    assert messages_none[0]["role"] == "system"
    assert messages_none[1]["role"] == "user"

    messages_empty = builder.build(history_messages=[], user_input="Hi", retrieved_context="   ")
    assert len(messages_empty) == 2
    assert messages_empty[0]["role"] == "system"
    assert messages_empty[1]["role"] == "user"


def test_system_prompt_contents() -> None:
    """Ensure SYSTEM_PROMPT contains all mandated guidelines and rules."""
    assert "helpful, accurate AI assistant" in SYSTEM_PROMPT
    assert "clear, concise" in SYSTEM_PROMPT
    assert "Markdown" in SYSTEM_PROMPT
    assert "code blocks" in SYSTEM_PROMPT
    assert "same tone every turn" in SYSTEM_PROMPT
    assert "user's language" in SYSTEM_PROMPT
    assert "I don't know" in SYSTEM_PROMPT
    assert "earlier messages" in SYSTEM_PROMPT
    assert "Today's date is" in SYSTEM_PROMPT
