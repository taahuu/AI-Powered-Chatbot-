"""Shared test fixtures, FakeLLM, and FakeRetriever implementations."""

from pathlib import Path
import sys

# Ensure root directory is in sys.path for testing
_ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from typing import Iterator, Optional, Sequence
import pytest

from chatbot.llm_client import BaseLLM
from chatbot.retriever import BaseRetriever, Document
from config import Settings


class FakeLLM(BaseLLM):
    """Test double for BaseLLM that yields predefined chunks and records prompts."""

    def __init__(
        self,
        responses: Optional[list[list[str]]] = None,
        error_to_raise: Optional[Exception] = None,
        error_after_chunks: int = -1,
        model: str = "mock-gpt",
    ) -> None:
        self.model = model
        self.responses: list[list[str]] = responses or [["Hello", " ", "world!"]]
        self.call_count: int = 0
        self.history: list[list[dict[str, str]]] = []
        self.error_to_raise = error_to_raise
        self.error_after_chunks = error_after_chunks

    def stream(self, messages: Sequence[dict[str, str]]) -> Iterator[str]:
        """Record received messages and yield scripted token chunks."""
        self.history.append([dict(m) for m in messages])
        current_call = self.call_count
        self.call_count += 1

        if self.error_to_raise and self.error_after_chunks == 0:
            raise self.error_to_raise

        chunks = (
            self.responses[current_call]
            if current_call < len(self.responses)
            else ["Echo", " ", "chunk"]
        )

        yielded = 0
        for chunk in chunks:
            if self.error_to_raise and yielded == self.error_after_chunks:
                raise self.error_to_raise
            yield chunk
            yielded += 1


class RecordingLLM(FakeLLM):
    """Test double that records all received prompt histories for exact assertion."""
    pass


class FakeRetriever(BaseRetriever):
    """Test double for BaseRetriever that returns fixed test documents."""

    def __init__(self, documents: Optional[list[Document]] = None) -> None:
        self.documents: list[Document] = (
            documents
            if documents is not None
            else [Document(content="Artificial intelligence reference data.", source="kb_ref_1")]
        )
        self.queries_received: list[str] = []

    def retrieve(self, query: str, k: int = 3) -> list[Document]:
        """Record the search query and return test documents."""
        self.queries_received.append(query)
        return self.documents[:k]


@pytest.fixture
def fake_llm() -> FakeLLM:
    """Fixture providing a fresh FakeLLM."""
    return FakeLLM()


@pytest.fixture
def recording_llm() -> RecordingLLM:
    """Fixture providing a fresh RecordingLLM."""
    return RecordingLLM()


@pytest.fixture
def fake_retriever() -> FakeRetriever:
    """Fixture providing a fresh FakeRetriever."""
    return FakeRetriever()


@pytest.fixture(autouse=True)
def isolated_sessions_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Ensure every test runs with an isolated session directory."""
    sessions_path = tmp_path / "sessions"
    sessions_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SESSIONS_DIR", str(sessions_path))
    return sessions_path


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing standard test settings."""
    return Settings(
        llm_api_key="sk-test-mock-key-12345",
        llm_base_url="https://api.mock.test/v1",
        llm_model="mock-gpt",
        llm_temperature=0.7,
        llm_max_output_tokens=512,
        max_context_chars=1000,
        max_input_chars=500,
    )
