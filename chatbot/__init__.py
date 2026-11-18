"""AI Chatbot package."""

from chatbot.errors import (
    AuthenticationFailedError,
    ChatbotError,
    ConfigError,
    ConnectionFailedError,
    InvalidInputError,
    LLMError,
    ModelError,
    RateLimitedError,
)
from chatbot.llm_client import BaseLLM, LLMClient
from chatbot.memory import ConversationMemory
from chatbot.models import (
    ChatSession,
    DEFAULT_SESSIONS_DIR,
    Message,
    delete_session,
    get_latest_session,
    list_sessions,
    load_session,
    save_session,
)
from chatbot.prompts import PromptBuilder, clean_user_input, SYSTEM_PROMPT
from chatbot.retriever import BaseRetriever, Document, NullRetriever
from chatbot.service import ChatService

__all__ = [
    "AuthenticationFailedError",
    "BaseLLM",
    "BaseRetriever",
    "ChatSession",
    "ChatService",
    "ChatbotError",
    "ConfigError",
    "ConnectionFailedError",
    "ConversationMemory",
    "DEFAULT_SESSIONS_DIR",
    "Document",
    "InvalidInputError",
    "LLMClient",
    "LLMError",
    "Message",
    "ModelError",
    "NullRetriever",
    "PromptBuilder",
    "RateLimitedError",
    "SYSTEM_PROMPT",
    "clean_user_input",
    "delete_session",
    "get_latest_session",
    "list_sessions",
    "load_session",
    "save_session",
]
