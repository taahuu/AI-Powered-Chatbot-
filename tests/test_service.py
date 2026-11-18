"""Unit tests for ChatService: context preservation, error recovery, regeneration, and retrieval."""

import pytest

from chatbot.errors import ConnectionFailedError, InvalidInputError, RateLimitedError
from chatbot.memory import ConversationMemory
from chatbot.models import ChatSession, load_session
from chatbot.prompts import PromptBuilder
from chatbot.retriever import Document, NullRetriever
from chatbot.service import ChatService
from tests.conftest import FakeLLM, FakeRetriever, RecordingLLM


def test_multi_turn_conversation_exact_context_recording(tmp_path) -> None:
    """Verify exact message list and roles sent to the LLM on turns 2, 3, and 4."""
    recording_llm = RecordingLLM(
        responses=[
            ["Paris", " is ", "the capital."],
            ["It", " means ", "Paris."],
            ["def add(a, b):\n    return a + b"],
            ["Photosynthesis", " is how plants make food."],
        ]
    )
    service = ChatService(
        llm=recording_llm,
        memory=ConversationMemory(max_context_chars=10000),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
        sessions_dir=str(tmp_path),
    )
    session = ChatSession()

    # Turn 1: Factual
    list(service.send_message(session, "What is the capital of France?"))
    assert len(recording_llm.history) == 1
    assert len(session.messages) == 2

    # Turn 2: Follow-up
    list(service.send_message(session, "Explain that simpler"))
    assert len(recording_llm.history) == 2
    assert len(session.messages) == 4

    turn2_messages = recording_llm.history[1]
    assert len(turn2_messages) == 4
    assert [m["role"] for m in turn2_messages] == ["system", "user", "assistant", "user"]
    assert turn2_messages[0]["role"] == "system"
    assert turn2_messages[1] == {"role": "user", "content": "What is the capital of France?"}
    assert turn2_messages[2] == {"role": "assistant", "content": "Paris is the capital."}
    assert turn2_messages[3] == {"role": "user", "content": "Explain that simpler"}

    # Turn 3: Coding
    list(service.send_message(session, "Write a Python function to add two numbers"))
    assert len(recording_llm.history) == 3
    assert len(session.messages) == 6

    turn3_messages = recording_llm.history[2]
    assert len(turn3_messages) == 6
    assert [m["role"] for m in turn3_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert turn3_messages[1] == {"role": "user", "content": "What is the capital of France?"}
    assert turn3_messages[2] == {"role": "assistant", "content": "Paris is the capital."}
    assert turn3_messages[3] == {"role": "user", "content": "Explain that simpler"}
    assert turn3_messages[4] == {"role": "assistant", "content": "It means Paris."}
    assert turn3_messages[5] == {
        "role": "user",
        "content": "Write a Python function to add two numbers",
    }

    # Turn 4: Topic switch
    list(service.send_message(session, "What is photosynthesis?"))
    assert len(recording_llm.history) == 4
    assert len(session.messages) == 8

    turn4_messages = recording_llm.history[3]
    assert len(turn4_messages) == 8
    assert [m["role"] for m in turn4_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert turn4_messages[5] == {
        "role": "user",
        "content": "Write a Python function to add two numbers",
    }
    assert turn4_messages[6] == {
        "role": "assistant",
        "content": "def add(a, b):\n    return a + b",
    }
    assert turn4_messages[7] == {"role": "user", "content": "What is photosynthesis?"}

    # Verify session persisted to disk
    saved = load_session(tmp_path / f"{session.id}.json")
    assert saved is not None
    assert len(saved.messages) == 8


def test_service_rejects_empty_and_whitespace_input() -> None:
    """ChatService raises InvalidInputError for empty or whitespace-only inputs."""
    service = ChatService(
        llm=FakeLLM(),
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    with pytest.raises(InvalidInputError):
        list(service.send_message(session, ""))

    with pytest.raises(InvalidInputError):
        list(service.send_message(session, "   \n\t  "))


def test_service_rejects_input_exceeding_character_limit() -> None:
    """ChatService raises InvalidInputError when input exceeds max_input_chars."""
    service = ChatService(
        llm=FakeLLM(),
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(max_input_chars=50),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    with pytest.raises(InvalidInputError):
        list(service.send_message(session, "X" * 51))


def test_assistant_reply_stored_after_streaming() -> None:
    """On complete streaming, assistant reply must be appended with incomplete=False."""
    fake_llm = FakeLLM(responses=[["Hello", " there!"]])
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    list(service.send_message(session, "Hi"))

    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Hi"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Hello there!"
    assert session.messages[1].incomplete is False


def test_mid_stream_failure_stores_incomplete_message() -> None:
    """If an error occurs mid-stream, partial output must be saved with incomplete=True."""
    fake_llm = FakeLLM(
        responses=[["Partial", " chunk", " never reached"]],
        error_to_raise=RateLimitedError("Rate limit hit"),
        error_after_chunks=1,  # fails after yielding "Partial"
    )
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    with pytest.raises(RateLimitedError):
        list(service.send_message(session, "Tell me something"))

    # Session should contain the user message and an incomplete assistant message
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Partial"
    assert session.messages[1].incomplete is True


def test_failure_before_any_chunk_removes_user_message() -> None:
    """If LLM fails before yielding any chunk, the added user message must be removed."""
    fake_llm = FakeLLM(
        error_to_raise=ConnectionFailedError("Cannot connect"),
        error_after_chunks=0,
    )
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    with pytest.raises(ConnectionFailedError):
        list(service.send_message(session, "Hello world"))

    # Session must remain empty since no tokens were generated
    assert len(session.messages) == 0


def test_regenerate_replaces_last_answer() -> None:
    """Regenerate should remove the previous assistant answer and stream a replacement."""
    fake_llm = FakeLLM(
        responses=[
            ["First", " answer"],
            ["Regenerated", " better", " answer"],
        ]
    )
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    # Initial answer
    list(service.send_message(session, "Write a poem"))
    assert session.messages[-1].content == "First answer"
    assert len(session.messages) == 2

    # Regenerate
    regen_chunks = list(service.regenerate(session))
    assert "".join(regen_chunks) == "Regenerated better answer"
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[0].content == "Write a poem"
    assert session.messages[1].role == "assistant"
    assert session.messages[1].content == "Regenerated better answer"


def test_regenerate_empty_session_raises_invalid_input_error() -> None:
    """Regenerate without any prior user turn must raise InvalidInputError."""
    service = ChatService(
        llm=FakeLLM(),
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    with pytest.raises(InvalidInputError):
        service.regenerate(session)


def test_retriever_output_appears_in_prompt_when_using_fake_retriever() -> None:
    """Retrieved document text appears wrapped in <context> when retriever returns results."""
    fake_retriever = FakeRetriever(
        documents=[Document(content="Internal company policy doc.", source="kb_01")]
    )
    fake_llm = FakeLLM()
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=fake_retriever,
    )
    session = ChatSession()

    list(service.send_message(session, "What is the policy?"))

    assert len(fake_llm.history) == 1
    prompt_sent = fake_llm.history[0]
    # Check that a system message with <context> is present
    context_msgs = [m for m in prompt_sent if "<context>" in m.get("content", "")]
    assert len(context_msgs) == 1
    assert "Internal company policy doc." in context_msgs[0]["content"]
    assert "[Source: kb_01]" in context_msgs[0]["content"]


def test_retriever_output_absent_with_null_retriever() -> None:
    """With NullRetriever, no <context> message is added to the prompt."""
    fake_llm = FakeLLM()
    service = ChatService(
        llm=fake_llm,
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )
    session = ChatSession()

    list(service.send_message(session, "Hello there"))

    prompt_sent = fake_llm.history[0]
    context_msgs = [m for m in prompt_sent if "<context>" in m.get("content", "")]
    assert len(context_msgs) == 0
