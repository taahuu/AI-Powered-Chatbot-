"""Data models and persistence for chat messages and conversation sessions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
from typing import Optional
import uuid

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR: str = "data/sessions"


@dataclass
class Message:
    """Represents a single message in the chat history.

    Attributes:
        role: Message role ('user' or 'assistant'). System prompts are never stored in history.
        content: The text content of the message.
        incomplete: Whether message generation was interrupted mid-stream.
    """

    role: str
    content: str
    incomplete: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "incomplete": self.incomplete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Message":
        """Reconstruct message from a dictionary."""
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            incomplete=bool(data.get("incomplete", False)),
        )


@dataclass
class ChatSession:
    """Represents a conversation session maintaining message history.

    Attributes:
        id: Unique UUID string identifying this session.
        title: Short title summarizing the conversation (derived from first user message).
        messages: List of Message objects in chronological order.
        created_at: Datetime when the session was initialized.
        updated_at: Datetime when the session was last updated.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New chat"
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, role: str, content: str, incomplete: bool = False) -> Message:
        """Add a new message to the session history and update metadata."""
        msg = Message(role=role, content=content, incomplete=incomplete)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)

        # Derive title from first user message if currently default
        if (self.title == "New chat" or not self.title) and role == "user":
            first_line = content.strip().split("\n")[0].strip()
            self.title = first_line[:40] if first_line else "New chat"

        return msg

    def clear(self) -> None:
        """Clear all messages from the session history."""
        self.messages.clear()
        self.title = "New chat"
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, object]:
        """Serialize session to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ChatSession":
        """Deserialize session from dictionary."""
        created_raw = data.get("created_at")
        updated_raw = data.get("updated_at")

        try:
            created_at = (
                datetime.fromisoformat(str(created_raw))
                if created_raw
                else datetime.now(timezone.utc)
            )
        except Exception:
            created_at = datetime.now(timezone.utc)

        try:
            updated_at = (
                datetime.fromisoformat(str(updated_raw))
                if updated_raw
                else created_at
            )
        except Exception:
            updated_at = created_at

        raw_msgs = data.get("messages", [])
        messages = [
            Message.from_dict(m) for m in raw_msgs if isinstance(m, dict)
        ]

        return cls(
            id=str(data.get("id", str(uuid.uuid4()))),
            title=str(data.get("title", "New chat")),
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
        )

    def save(self, sessions_dir: str = DEFAULT_SESSIONS_DIR) -> Path:
        """Save this session to disk atomically."""
        return save_session(self, sessions_dir=sessions_dir)


def save_session(session: ChatSession, sessions_dir: str = DEFAULT_SESSIONS_DIR) -> Path:
    """Save session to a JSON file atomically using a temporary file and rename.

    Args:
        session: ChatSession to persist.
        sessions_dir: Target directory path.

    Returns:
        Path to the saved session JSON file.
    """
    os.makedirs(sessions_dir, exist_ok=True)
    target_path = Path(sessions_dir) / f"{session.id}.json"

    # Write to temporary file in the same directory, then atomic rename
    fd, temp_path = tempfile.mkstemp(dir=sessions_dir, prefix="tmp_sess_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(session.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(temp_path, target_path)
    except Exception:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

    return target_path


def load_session(filepath: Path | str) -> Optional[ChatSession]:
    """Load a session from a JSON file.

    Corrupted or unreadable files do not raise exceptions, but log a warning and return None.

    Args:
        filepath: Path to the JSON file.

    Returns:
        ChatSession if successfully read, None otherwise.
    """
    path = Path(filepath)
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("Session file %s does not contain a JSON object", path)
            return None
        return ChatSession.from_dict(data)
    except Exception as exc:
        logger.warning("Corrupted or unreadable session file %s: %s", path, exc)
        return None


def list_sessions(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> list[ChatSession]:
    """List all valid sessions from the storage directory, sorted newest first.

    Args:
        sessions_dir: Directory containing session JSON files.

    Returns:
        List of ChatSession objects ordered by updated_at descending.
    """
    path = Path(sessions_dir)
    if not path.exists():
        return []

    sessions: list[ChatSession] = []
    for json_file in path.glob("*.json"):
        session = load_session(json_file)
        if session is not None:
            sessions.append(session)

    # Newest first by updated_at
    sessions.sort(key=lambda s: s.updated_at, reverse=True)
    return sessions


def get_latest_session(sessions_dir: str = DEFAULT_SESSIONS_DIR) -> Optional[ChatSession]:
    """Retrieve the most recently updated session, or None if no sessions exist."""
    sessions = list_sessions(sessions_dir)
    return sessions[0] if sessions else None


def delete_session(session_id: str, sessions_dir: str = DEFAULT_SESSIONS_DIR) -> bool:
    """Delete a session JSON file from disk.

    Args:
        session_id: UUID of the session to delete.
        sessions_dir: Storage directory.

    Returns:
        True if deleted, False if file did not exist.
    """
    target = Path(sessions_dir) / f"{session_id}.json"
    if target.is_file():
        try:
            target.unlink()
            return True
        except OSError as exc:
            logger.warning("Failed to delete session file %s: %s", target, exc)
            return False
    return False
