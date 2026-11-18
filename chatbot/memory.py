"""Conversation memory management and context trimming."""

from chatbot.models import ChatSession, Message

DEFAULT_MAX_CONTEXT_CHARS: int = 24000


class ConversationMemory:
    """Manages conversation context window and message trimming for LLM calls."""

    def __init__(self, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> None:
        self.max_context_chars = max_context_chars

    def get_context(self, session: ChatSession) -> list[Message]:
        """Extract the most recent messages fitting within the context limit.

        Rules:
        - Exclude messages marked as incomplete.
        - If the total content length is within max_context_chars, return all messages.
        - Otherwise, walk newest to oldest until the character limit is reached.
        - Return messages in chronological order.
        - If trimming causes the context to start with an assistant message, drop it.

        Args:
            session: Current chat session.

        Returns:
            List of Message objects to include in the LLM context.
        """
        valid_messages = [m for m in session.messages if not m.incomplete]

        total_length = sum(len(m.content) for m in valid_messages)
        if total_length <= self.max_context_chars:
            candidate = list(valid_messages)
        else:
            accumulated_chars = 0
            selected: list[Message] = []
            for msg in reversed(valid_messages):
                msg_len = len(msg.content)
                if accumulated_chars + msg_len <= self.max_context_chars:
                    selected.append(msg)
                    accumulated_chars += msg_len
                else:
                    break
            # If newest message exceeds limit alone, send it anyway (never silently empty)
            if not selected and valid_messages:
                selected.append(valid_messages[-1])
            candidate = list(reversed(selected))

        # Context must never begin with an assistant message
        while candidate and candidate[0].role == "assistant":
            candidate.pop(0)

        return candidate
