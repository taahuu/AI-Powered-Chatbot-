"""System prompts and prompt construction logic."""

from datetime import date
import re
from typing import Sequence

from chatbot.errors import InvalidInputError
from chatbot.models import Message

DEFAULT_MAX_INPUT_CHARS: int = 4000

_TODAY: str = date.today().strftime("%Y-%m-%d")

SYSTEM_PROMPT: str = (
    f"You are a helpful, accurate AI assistant. Today's date is {_TODAY}.\n\n"
    "Guidelines:\n"
    "- Style: clear, concise by default. Use Markdown formatting. Use code blocks for code. "
    "Provide step-by-step explanations when asked.\n"
    "- Consistency: Maintain the same tone every turn. Always answer in the user's language.\n"
    "- Honesty: Say 'I don't know' rather than invent facts.\n"
    "- Context: Use earlier messages in this conversation to resolve references like 'it', 'that', 'the previous one'."
)


def clean_user_input(text: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> str:
    """Validate and clean user input text.

    Strips leading/trailing whitespace, collapses runs of 3+ blank lines,
    and checks against length limits.

    Args:
        text: Raw user input text.
        max_chars: Maximum allowed characters.

    Returns:
        Cleaned input text.

    Raises:
        InvalidInputError: If text is empty or exceeds max_chars.
    """
    if not isinstance(text, str):
        raise InvalidInputError("User input must be a string.")

    stripped = text.strip()
    if not stripped:
        raise InvalidInputError("User input cannot be empty.")

    # Collapse runs of 3 or more blank lines (or newlines) down to 2 newlines
    cleaned = re.sub(r'(\r?\n[ \t]*){3,}', '\n\n', stripped)

    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"User input exceeds maximum length of {max_chars} characters (received {len(cleaned)})."
        )

    return cleaned


class PromptBuilder:
    """Builds structured message lists for LLM chat completion calls."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, max_input_chars: int = DEFAULT_MAX_INPUT_CHARS) -> None:
        self.system_prompt = system_prompt
        self.max_input_chars = max_input_chars

    def build(
        self,
        history_messages: Sequence[Message],
        user_input: str,
        retrieved_context: str | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the complete prompt list in fixed order.

        Order:
        1. Base system prompt
        2. Retrieved context wrapped in <context>...</context> (if provided and non-empty)
        3. History messages in chronological order
        4. Cleaned user input

        Args:
            history_messages: Prior conversation messages to include.
            user_input: Current user query.
            retrieved_context: Optional context from a retriever.

        Returns:
            List of message dictionaries for the chat completion API.
        """
        cleaned_user = clean_user_input(user_input, max_chars=self.max_input_chars)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if retrieved_context and retrieved_context.strip():
            context_block = (
                "Context information is below.\n"
                f"<context>\n{retrieved_context.strip()}\n</context>\n"
                "Please prefer this context when relevant to the user query."
            )
            messages.append({"role": "system", "content": context_block})

        for msg in history_messages:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": cleaned_user})
        return messages
