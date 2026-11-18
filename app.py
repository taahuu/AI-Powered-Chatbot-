"""Streamlit user interface for the AI-Powered Chatbot."""

from collections.abc import Iterator
import logging
from typing import Optional
import streamlit as st

from chatbot.errors import ChatbotError, ConfigError
from chatbot.llm_client import LLMClient
from chatbot.memory import ConversationMemory
from chatbot.models import ChatSession, delete_session, get_latest_session, list_sessions
from chatbot.prompts import PromptBuilder
from chatbot.retriever import NullRetriever
from chatbot.service import ChatService
from config import Settings, load_settings

logger = logging.getLogger(__name__)


def build_service(settings: Optional[Settings] = None) -> ChatService:
    """Instantiate and wire application services from configuration."""
    if settings is None:
        settings = load_settings()
    llm = LLMClient(settings)
    memory = ConversationMemory(max_context_chars=settings.max_context_chars)
    prompt_builder = PromptBuilder(max_input_chars=settings.max_input_chars)
    retriever = NullRetriever()
    return ChatService(
        llm=llm,
        memory=memory,
        prompt_builder=prompt_builder,
        retriever=retriever,
        sessions_dir=settings.sessions_dir,
    )


def stream_with_status(generator: Iterator[str], placeholder: st.delta_generator.DeltaGenerator) -> Iterator[str]:
    """Yield chunks while showing status placeholder until the first chunk arrives."""
    first = True
    try:
        for chunk in generator:
            if first:
                placeholder.empty()
                first = False
            yield chunk
    finally:
        placeholder.empty()


def on_new_chat() -> None:
    """Callback to start a new chat session."""
    st.session_state.session = ChatSession()


def on_clear_chat() -> None:
    """Callback to empty messages in the active chat session and update disk."""
    if "session" in st.session_state:
        st.session_state.session.clear()
        settings = getattr(st.session_state, "settings", None)
        if settings:
            st.session_state.session.save(settings.sessions_dir)


def on_delete_chat() -> None:
    """Callback to delete the current chat session from disk and switch to another."""
    if "session" in st.session_state:
        current_id = st.session_state.session.id
        settings = getattr(st.session_state, "settings", None)
        sessions_dir = settings.sessions_dir if settings else "data/sessions"
        delete_session(current_id, sessions_dir)
        latest = get_latest_session(sessions_dir)
        st.session_state.session = latest if latest is not None else ChatSession()


def init_state() -> None:
    """Initialize session state objects once per browser session."""
    if "settings" not in st.session_state:
        try:
            st.session_state.settings = load_settings()
        except ConfigError as err:
            st.error(
                f"Configuration Error: {err.user_message}\n\n"
                "Please create a `.env` file from `.env.example` and set your `LLM_API_KEY`:\n\n"
                "```bash\ncp .env.example .env\n```"
            )
            st.stop()

    settings: Settings = st.session_state.settings

    if "session" not in st.session_state:
        latest = get_latest_session(settings.sessions_dir)
        st.session_state.session = latest if latest is not None else ChatSession()

    if "service" not in st.session_state:
        try:
            st.session_state.service = build_service(settings)
        except Exception as exc:
            logger.exception("Unexpected failure during service setup: %s", exc)
            st.error("Failed to initialize chat service. Please check application configuration.")
            st.stop()


def main() -> None:
    """Render the Streamlit chatbot application."""
    st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

    init_state()

    settings: Settings = st.session_state.settings
    session: ChatSession = st.session_state.session
    service: ChatService = st.session_state.service

    # Sidebar configuration and controls
    st.sidebar.title("AI Chatbot")
    model_name = getattr(service.llm, "model", "Unknown")
    st.sidebar.caption(f"**Model:** `{model_name}`")

    st.sidebar.button("New chat", on_click=on_new_chat, use_container_width=True)
    st.sidebar.button("Clear chat", on_click=on_clear_chat, use_container_width=True)
    st.sidebar.button("Delete chat", on_click=on_delete_chat, use_container_width=True)

    message_count = len(session.messages)
    st.sidebar.metric(label="Messages", value=message_count)

    # Saved sessions list
    saved_sessions = list_sessions(settings.sessions_dir)
    if saved_sessions:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Saved Chats")
        for s in saved_sessions:
            label = s.title if s.title else "New chat"
            btn_label = f"👉 {label}" if s.id == session.id else f"💬 {label}"
            if st.sidebar.button(btn_label, key=f"session_btn_{s.id}", use_container_width=True):
                st.session_state.session = s
                st.rerun()

    # Main conversation view
    if not session.messages:
        st.info("👋 Welcome! Ask me anything to get started.")

    for msg in session.messages:
        with st.chat_message(msg.role):
            st.markdown(msg.content)
            if msg.incomplete:
                st.warning("Response was interrupted")

    # Show Regenerate button only when the last message is from the assistant
    if session.messages and session.messages[-1].role == "assistant":
        if st.button("🔄 Regenerate", key="regenerate_btn"):
            regen_success = False
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown("_Thinking..._")
                try:
                    st.write_stream(stream_with_status(service.regenerate(session), placeholder))
                    regen_success = True
                    session.save(settings.sessions_dir)
                except ChatbotError as err:
                    st.error(err.user_message)
                except Exception:
                    logger.exception("Unexpected error during regeneration")
                    st.error("An unexpected error occurred. Please try again.")

            if regen_success:
                st.rerun()

    # User input handling
    user_input = st.chat_input("Type your message...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        stream_success = False
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking..._")
            try:
                st.write_stream(stream_with_status(service.send_message(session, user_input), placeholder))
                stream_success = True
                session.save(settings.sessions_dir)
            except ChatbotError as err:
                st.error(err.user_message)
            except Exception:
                logger.exception("Unexpected error during message generation")
                st.error("An unexpected error occurred. Please try again.")

        if stream_success:
            st.rerun()


if __name__ == "__main__":
    main()
