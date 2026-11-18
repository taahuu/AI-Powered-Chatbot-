"""Unit tests for conversation session model and atomic JSON disk persistence."""

import json
from pathlib import Path
import pytest

from chatbot.models import (
    ChatSession,
    Message,
    delete_session,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)


def test_session_initial_state() -> None:
    """A fresh session has a unique id, default title, and empty messages."""
    session = ChatSession()
    assert session.id is not None
    assert session.title == "New chat"
    assert session.messages == []
    assert session.created_at is not None
    assert session.updated_at is not None


def test_session_title_derived_from_first_user_message() -> None:
    """The session title is set from the first user message (up to 40 chars)."""
    session = ChatSession()
    session.add("user", "What is the airspeed velocity of an unladen swallow?")
    assert len(session.title) <= 40
    assert session.title == "What is the airspeed velocity of an unla"

    # Second user message must not overwrite title
    session.add("assistant", "African or European swallow?")
    session.add("user", "Another question that should not change title")
    assert session.title == "What is the airspeed velocity of an unla"


def test_session_clear_resets_messages_and_title() -> None:
    """Clearing a session removes all messages and resets the title."""
    session = ChatSession()
    session.add("user", "Hello there")
    session.add("assistant", "General Kenobi")
    assert len(session.messages) == 2

    session.clear()
    assert len(session.messages) == 0
    assert session.title == "New chat"


def test_session_save_and_load_roundtrip(tmp_path: Path) -> None:
    """A session can be atomically saved and cleanly reloaded from disk."""
    session = ChatSession()
    session.add("user", "Test question")
    session.add("assistant", "Test answer")

    file_path = save_session(session, sessions_dir=str(tmp_path))
    assert file_path.is_file()

    loaded = load_session(file_path)
    assert loaded is not None
    assert loaded.id == session.id
    assert loaded.title == session.title
    assert len(loaded.messages) == 2
    assert loaded.messages[0].role == "user"
    assert loaded.messages[0].content == "Test question"
    assert loaded.messages[1].role == "assistant"
    assert loaded.messages[1].content == "Test answer"


def test_load_session_handles_corrupt_files_gracefully(tmp_path: Path) -> None:
    """Corrupted JSON files or non-dict files do not crash the application."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{this is not valid json!}", encoding="utf-8")

    result = load_session(bad_json)
    assert result is None

    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]", encoding="utf-8")

    result_arr = load_session(array_json)
    assert result_arr is None

    nonexistent = tmp_path / "does_not_exist.json"
    assert load_session(nonexistent) is None


def test_list_and_get_latest_sessions(tmp_path: Path) -> None:
    """Sessions are listed in newest-first order by updated_at."""
    sess1 = ChatSession(id="sess-1", title="Chat 1")
    sess1.add("user", "First")
    save_session(sess1, sessions_dir=str(tmp_path))

    sess2 = ChatSession(id="sess-2", title="Chat 2")
    sess2.add("user", "Second")
    save_session(sess2, sessions_dir=str(tmp_path))

    sessions = list_sessions(str(tmp_path))
    assert len(sessions) == 2
    # sess2 was created/updated later
    assert sessions[0].id == "sess-2"
    assert sessions[1].id == "sess-1"

    latest = get_latest_session(str(tmp_path))
    assert latest is not None
    assert latest.id == "sess-2"


def test_delete_session_removes_file(tmp_path: Path) -> None:
    """delete_session removes the file from disk and returns True."""
    session = ChatSession(id="to-delete")
    session.add("user", "Delete me")
    target = save_session(session, sessions_dir=str(tmp_path))
    assert target.is_file()

    assert delete_session("to-delete", str(tmp_path)) is True
    assert not target.exists()

    # Deleting again returns False
    assert delete_session("to-delete", str(tmp_path)) is False
