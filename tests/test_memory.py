"""Unit tests for conversation memory context management and message trimming."""

from chatbot.memory import ConversationMemory
from chatbot.models import ChatSession


def test_memory_returns_everything_under_limit() -> None:
    """All messages are returned when the total character count is under the budget."""
    memory = ConversationMemory(max_context_chars=1000)
    session = ChatSession()
    session.add("user", "Hello there")
    session.add("assistant", "General Kenobi")
    session.add("user", "How are you?")

    context = memory.get_context(session)
    assert len(context) == 3
    assert [m.content for m in context] == ["Hello there", "General Kenobi", "How are you?"]


def test_memory_trims_oldest_first_when_over_limit() -> None:
    """When exceeding character limit, oldest messages are trimmed first."""
    # 4 messages of 10 chars each = 40 chars total
    # limit = 25 chars -> can fit at most 2 messages (20 chars)
    memory = ConversationMemory(max_context_chars=25)
    session = ChatSession()
    session.add("user", "1234567890")       # msg 0
    session.add("assistant", "abcdefghij")  # msg 1
    session.add("user", "1234567890")       # msg 2
    session.add("assistant", "abcdefghij")  # msg 3

    context = memory.get_context(session)
    # Newest two are msg 2 (user) and msg 3 (assistant).
    # Since msg 2 is user, candidate starts with user, so no leading assistant dropped.
    assert len(context) == 2
    assert context[0].role == "user"
    assert context[1].role == "assistant"
    assert context[0].content == "1234567890"
    assert context[1].content == "abcdefghij"


def test_memory_never_begins_with_assistant_message() -> None:
    """If trimming results in a leading assistant message, it must be dropped."""
    # If limit fits only 1 message (10 chars), the newest message is msg 3 (assistant).
    # Since context must never start with an assistant message, it should be dropped.
    memory = ConversationMemory(max_context_chars=15)
    session = ChatSession()
    session.add("user", "1234567890")       # msg 0
    session.add("assistant", "abcdefghij")  # msg 1
    session.add("user", "1234567890")       # msg 2
    session.add("assistant", "abcdefghij")  # msg 3

    context = memory.get_context(session)
    # The candidate walked backwards: msg 3 (10 chars).
    # But msg 3 is assistant, so dropping leading assistant leaves empty context.
    assert context == []


def test_memory_excludes_incomplete_messages() -> None:
    """Messages marked as incomplete must be excluded from context."""
    memory = ConversationMemory(max_context_chars=1000)
    session = ChatSession()
    session.add("user", "First question")
    session.add("assistant", "Interrupted ans...", incomplete=True)
    session.add("user", "Second question")
    session.add("assistant", "Complete answer")

    context = memory.get_context(session)
    assert len(context) == 3
    assert [m.content for m in context] == [
        "First question",
        "Second question",
        "Complete answer",
    ]
    assert all(not m.incomplete for m in context)


def test_memory_preserves_chronological_order() -> None:
    """Messages returned must always maintain chronological sequence."""
    memory = ConversationMemory(max_context_chars=1000)
    session = ChatSession()
    turns = [
        ("user", "msg1"),
        ("assistant", "msg2"),
        ("user", "msg3"),
        ("assistant", "msg4"),
        ("user", "msg5"),
    ]
    for role, text in turns:
        session.add(role, text)

    context = memory.get_context(session)
    assert [m.content for m in context] == ["msg1", "msg2", "msg3", "msg4", "msg5"]


def test_memory_newest_message_exceeding_limit_alone_is_sent() -> None:
    """If the newest message alone exceeds the limit, it must be sent anyway (never silently empty)."""
    memory = ConversationMemory(max_context_chars=10)
    session = ChatSession()
    session.add("user", "Hello earlier message")
    session.add("assistant", "Hi earlier reply")
    session.add("user", "A very long user prompt that exceeds ten chars")

    context = memory.get_context(session)
    assert len(context) == 1
    assert context[0].role == "user"
    assert context[0].content == "A very long user prompt that exceeds ten chars"


def test_memory_empty_session_returns_empty_list() -> None:
    """An empty session returns an empty context list."""
    memory = ConversationMemory(max_context_chars=100)
    session = ChatSession()
    assert memory.get_context(session) == []
