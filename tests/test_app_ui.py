"""UI automated tests using streamlit.testing.v1.AppTest."""

from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest

from chatbot.memory import ConversationMemory
from chatbot.models import ChatSession, save_session
from chatbot.prompts import PromptBuilder
from chatbot.retriever import NullRetriever
from chatbot.service import ChatService
from tests.conftest import FakeLLM, RecordingLLM

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture
def fake_service() -> ChatService:
    """Fixture providing a mock ChatService backed by FakeLLM."""
    return ChatService(
        llm=FakeLLM(
            responses=[
                ["First", " AI", " response"],
                ["Second", " AI", " response"],
                ["Third", " AI", " response"],
                ["Fourth", " AI", " response"],
            ]
        ),
        memory=ConversationMemory(),
        prompt_builder=PromptBuilder(),
        retriever=NullRetriever(),
    )


def test_app_starts_with_valid_environment(monkeypatch: pytest.MonkeyPatch, fake_service: ChatService) -> None:
    """App initializes properly when valid environment is configured."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        assert len(at.exception) == 0
        assert len(at.error) == 0
        assert len(at.chat_input) == 1
        assert at.chat_input[0].placeholder == "Type your message..."


def test_app_sending_message_shows_both_bubbles(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService
) -> None:
    """Sending a single message displays both the user and assistant bubbles."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Hello assistant!").run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 2
        assert at.chat_message[0].markdown[0].value == "Hello assistant!"
        assert at.chat_message[1].markdown[0].value == "First AI response"


def test_app_second_message_keeps_first_in_history(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService
) -> None:
    """Subsequent messages maintain earlier turns in the rendered conversation."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Turn 1 Question").run()
        at.chat_input[0].set_value("Turn 2 Question").run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 4
        assert at.chat_message[0].markdown[0].value == "Turn 1 Question"
        assert at.chat_message[1].markdown[0].value == "First AI response"
        assert at.chat_message[2].markdown[0].value == "Turn 2 Question"
        assert at.chat_message[3].markdown[0].value == "Second AI response"


def test_app_multi_turn_recording_asserts_exact_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Offline AppTest verifies exact message lists and roles received on turns 2, 3, and 4."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    recording_llm = RecordingLLM(
        responses=[
            ["Paris", " is ", "the capital."],
            ["It", " means ", "Paris."],
            ["def add(a, b):\n    return a + b"],
            ["Photosynthesis", " is how plants make food."],
        ]
    )

    with patch("chatbot.llm_client.LLMClient", return_value=recording_llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        # Turn 1
        at.chat_input[0].set_value("What is the capital of France?").run()
        assert len(at.chat_message) == 2

        # Turn 2
        at.chat_input[0].set_value("Explain that simpler").run()
        assert len(at.chat_message) == 4

        # Turn 3
        at.chat_input[0].set_value("Write a Python function to add two numbers").run()
        assert len(at.chat_message) == 6

        # Turn 4
        at.chat_input[0].set_value("What is photosynthesis?").run()
        assert len(at.chat_message) == 8

        # Assert Turn 2 exact messages
        turn2_prompt = recording_llm.history[1]
        assert len(turn2_prompt) == 4
        assert [m["role"] for m in turn2_prompt] == ["system", "user", "assistant", "user"]
        assert turn2_prompt[1] == {"role": "user", "content": "What is the capital of France?"}
        assert turn2_prompt[2] == {"role": "assistant", "content": "Paris is the capital."}
        assert turn2_prompt[3] == {"role": "user", "content": "Explain that simpler"}

        # Assert Turn 3 exact messages
        turn3_prompt = recording_llm.history[2]
        assert len(turn3_prompt) == 6
        assert [m["role"] for m in turn3_prompt] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert turn3_prompt[4] == {"role": "assistant", "content": "It means Paris."}
        assert turn3_prompt[5] == {
            "role": "user",
            "content": "Write a Python function to add two numbers",
        }

        # Assert Turn 4 exact messages
        turn4_prompt = recording_llm.history[3]
        assert len(turn4_prompt) == 8
        assert [m["role"] for m in turn4_prompt] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert turn4_prompt[6] == {
            "role": "assistant",
            "content": "def add(a, b):\n    return a + b",
        }
        assert turn4_prompt[7] == {"role": "user", "content": "What is photosynthesis?"}


def test_app_new_chat_empties_chat(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService
) -> None:
    """Clicking 'New chat' button resets the conversation session to empty."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Message to be wiped").run()
        assert len(at.chat_message) == 2

        new_chat_btn = [b for b in at.button if b.label == "New chat"][0]
        new_chat_btn.click().run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 0


def test_app_clear_chat_empties_chat(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService
) -> None:
    """Clicking 'Clear chat' button removes all messages in the active session."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Another message to clear").run()
        assert len(at.chat_message) == 2

        clear_chat_btn = [b for b in at.button if b.label == "Clear chat"][0]
        clear_chat_btn.click().run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 0


def test_app_delete_chat_removes_session_file(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService, tmp_path: Path
) -> None:
    """Clicking 'Delete chat' removes the session file and resets the view."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path))

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Message to delete").run()
        assert len(at.chat_message) == 2

        # Verify a session JSON file was created
        files_before = list(tmp_path.glob("*.json"))
        assert len(files_before) == 1

        delete_btn = [b for b in at.button if b.label == "Delete chat"][0]
        delete_btn.click().run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 0
        files_after = list(tmp_path.glob("*.json"))
        assert len(files_after) == 0


def test_app_regenerate_replaces_last_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clicking the Regenerate button replaces the last assistant response."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")

    recording_llm = RecordingLLM(
        responses=[
            ["First", " response"],
            ["Regenerated", " fresh", " response"],
        ]
    )

    with patch("chatbot.llm_client.LLMClient", return_value=recording_llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        at.chat_input[0].set_value("Hello").run()
        assert len(at.chat_message) == 2
        assert at.chat_message[1].markdown[0].value == "First response"

        regen_btn = [b for b in at.button if "Regenerate" in b.label][0]
        regen_btn.click().run()

        assert len(at.exception) == 0
        assert len(at.chat_message) == 2
        assert at.chat_message[1].markdown[0].value == "Regenerated fresh response"


def test_app_saved_chat_switching(
    monkeypatch: pytest.MonkeyPatch, fake_service: ChatService, tmp_path: Path
) -> None:
    """Saved chats listed in sidebar can be switched by clicking."""
    monkeypatch.setenv("LLM_API_KEY", "mock-test-key")
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path))

    # Pre-populate two sessions on disk
    s1 = ChatSession(id="chat-1", title="First Historical Topic")
    s1.add("user", "First Historical Topic")
    s1.add("assistant", "First Historical Response")
    save_session(s1, sessions_dir=str(tmp_path))

    s2 = ChatSession(id="chat-2", title="Second Historical Topic")
    s2.add("user", "Second Historical Topic")
    s2.add("assistant", "Second Historical Response")
    save_session(s2, sessions_dir=str(tmp_path))

    with patch("chatbot.llm_client.LLMClient", return_value=fake_service.llm):
        at = AppTest.from_file(APP_FILE)
        at.run()

        # The newest (s2) is loaded initially
        assert len(at.chat_message) == 2
        assert at.chat_message[0].markdown[0].value == "Second Historical Topic"

        # Click button for s1
        s1_btn = [b for b in at.button if "First Historical Topic" in b.label][0]
        s1_btn.click().run()

        # Now s1 is loaded
        assert len(at.chat_message) == 2
        assert at.chat_message[0].markdown[0].value == "First Historical Topic"


def test_app_missing_key_shows_error_and_no_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing API key renders a user-friendly error banner and raises no unhandled exceptions."""
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setattr("config.load_dotenv", lambda *args, **kwargs: None)

    at = AppTest.from_file(APP_FILE)
    at.run()

    assert len(at.exception) == 0
    assert len(at.error) >= 1
    assert "LLM_API_KEY" in at.error[0].value
