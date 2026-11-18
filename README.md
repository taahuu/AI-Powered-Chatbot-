# AI-Powered Chatbot using LLMs & Streamlit

A production-grade conversational AI chatbot built with Streamlit and modern OpenAI-compatible LLM APIs, featuring context-aware conversation memory, dynamic streaming, atomic disk session persistence, and pluggable knowledge retrieval hooks.

---

## 1. Features

- **F1 - Python & Streamlit Core:** Full chatbot application with LLM-based response generation and conversation memory (`streamlit run app.py`).
- **F2 - Interactive Interface:** Streamlit UI featuring distinct chat bubbles for user and assistant messages, anchored chat input bar, and sidebar controls.
- **F3 - Context-Aware LLM Integration:** Follow-up awareness across multiple conversation turns using real-time OpenAI-compatible LLM completions, with automatic retry on transient connection errors before chunk 1.
- **F4 - Prompt Engineering:** Strict prompt builder assembling system prompt, optional `<context>` knowledge block, chronological conversation history, and user input.
- **F5 - Conversation Memory:** Context window budgeting with reverse chronological trimming that never begins with an assistant message, excludes interrupted replies, and ensures the context is never silently empty.
- **F6 - Decoupled Python Logic:** Separation of concerns with `ChatService` orchestrating memory, retrieval, prompt assembly, and streaming without any LLM or SDK code inside the Streamlit frontend.
- **F7 - Dynamic Response Display:** Real-time token streaming (`st.write_stream`), transient status indicator until first token arrives, and instant regeneration of previous answers.
- **F8 - Robust Session Management:** Full multi-session persistence saved atomically (`tempfile.mkstemp` + `os.replace`) to `data/sessions/<id>.json` after each turn. Supports saved chat listing (newest first, titled from first user turn up to 40 chars), session switching, browser refresh resume, and dedicated "New chat", "Clear chat", and "Delete chat" controls.
- **F9 - Comprehensive Offline Testing:** 100% offline unit, service, model, and UI test suite (58 tests) using `RecordingLLM` verifying exact prompt histories across 4 conversational turns, session persistence, error mapping, and corruption tolerance.
- **F10 - Extensible Architecture for Future RAG:** Decoupled retriever interface (`BaseRetriever`, `Document`) wired directly into the prompt pipeline with zero-overhead `NullRetriever`.

---

## 2. Architecture & Data Flow

```text
[ User Input in UI ]
        │
        ▼
[ Streamlit app.py ] ──► calls ChatService.send_message(session, input)
        │                       │
        │                       ├──► 1. clean_user_input(text) (validates & strips)
        │                       ├──► 2. ConversationMemory.get_context(session)
        │                       ├──► 3. BaseRetriever.retrieve(query)
        │                       ├──► 4. PromptBuilder.build(history, input, retrieved_context)
        │                       ├──► 5. session.add(role="user", content=input)
        │                       ├──► 6. LLMClient.stream(messages)
        │                       │            │ (Streams tokens with retry backoff)
        ▼                       ▼            ▼
[ Streamlit UI: st.write_stream ] ◄── streams tokens live to user
        │
        ▼
[ Completion & Persistence ]
        │
        ├──► 7. session.add(role="assistant", content=full_text)
        └──► 8. session.save(sessions_dir)
                     │
                     ▼
           [ Atomic File Write ] ──► data/sessions/<id>.json
```

---

## 3. Project Structure

```text
Chatbot/
├── app.py                  # Streamlit UI, sidebar session list & buttons, streaming display
├── config.py               # Reads env vars, validates, exposes Settings dataclass
├── chatbot/
│   ├── __init__.py         # Package exports
│   ├── models.py           # Message and ChatSession models, atomic JSON persistence
│   ├── prompts.py          # System prompt + PromptBuilder + clean_user_input
│   ├── memory.py           # ConversationMemory (history context window trimming)
│   ├── llm_client.py       # LLMClient wrapper around openai SDK with retry & error mapping
│   ├── retriever.py        # BaseRetriever + NullRetriever (future RAG hooks)
│   ├── service.py          # ChatService: Python application logic coordinating UI and LLM
│   └── errors.py           # Custom exceptions with user-friendly messages
├── data/
│   └── sessions/           # JSON files for saved conversation sessions
├── tests/
│   ├── conftest.py         # FakeLLM, RecordingLLM, FakeRetriever, isolated sessions fixtures
│   ├── test_prompts.py     # Prompt assembly and input cleaning tests
│   ├── test_memory.py      # Context window trimming, assistant removal, and limit tests
│   ├── test_session.py     # Atomic save, reload, corrupt file tolerance, list, delete tests
│   ├── test_service.py     # Service orchestrator, multi-turn follow-up, and error recovery tests
│   ├── test_llm_client.py  # Error mapping and retry logic tests with mocked SDK
│   ├── test_config.py      # Configuration validation and fallback tests
│   └── test_app_ui.py      # Streamlit AppTest based UI tests with RecordingLLM
├── requirements.txt        # Pinned minimal dependencies
├── .env.example            # Environment configuration template
├── .gitignore              # Ignored files, virtual environments, and session storage
└── README.md               # Complete project documentation
```

---

## 4. Prerequisites

- **Python:** Version 3.10 or higher (Python 3.10, 3.11, or 3.12).
- An API key from OpenAI or any OpenAI-compatible provider (e.g. Groq, OpenRouter), or a locally running Ollama instance.

---

## 5. Setup & Installation

### Linux & macOS
```bash
# 1. Navigate to the project directory
cd /home/dell/Desktop/Chatbot

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env from template and configure your API key
cp .env.example .env
```

### Windows (PowerShell)
```powershell
# 1. Navigate to project directory
cd C:\path\to\Chatbot

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create .env from template
copy .env.example .env
```

Open `.env` in any text editor and supply your `LLM_API_KEY`:
```env
LLM_API_KEY=sk-your-openai-api-key-here
```

---

## 6. Using Other OpenAI-Compatible Providers or Local Ollama

The chatbot connects via the official `openai` SDK, enabling instant compatibility with alternative providers without modifying any application code.

### Local Ollama
```env
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

### Groq
```env
LLM_API_KEY=gsk_your_groq_api_key_here
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-70b-versatile
```

### OpenRouter
```env
LLM_API_KEY=sk-or-your-openrouter-key-here
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=anthropic/claude-3.5-sonnet
```

---

## 7. Run the App

With your virtual environment activated:

```bash
streamlit run app.py
```

The application will start and open automatically in your browser at `http://localhost:8501`.

---

## 8. Run the Automated Tests

Execute all offline automated tests with `pytest`:

```bash
pytest -q
```

All 58 unit, service, model, and UI tests execute completely offline without network requests or external credentials.

---

## 9. Manual Test Table

| # | Test Category | Input / Action | Expected Behaviour |
|---|---|---|---|
| 1 | Simple Factual Question | "What is the capital of Japan?" | Answers clearly and accurately: "Tokyo". |
| 2 | Coding Request | "Write a Python function to check for palindromes." | Provides concise code formatted inside a Markdown ` ```python ` block. |
| 3 | Multi-Turn Follow-Up | Turn 1: "Explain quantum entanglement."<br>Turn 2: "Explain that in simpler words." | Assistant references quantum entanglement from Turn 1 without restating the topic from scratch. |
| 4 | Topic Switch & Return | Turn 1: Discuss recipe.<br>Turn 2: Ask about math.<br>Turn 3: "Back to the recipe, how much salt?" | Assistant successfully recalls the recipe ingredients and context from Turn 1. |
| 5 | Very Long Input | Paste a block of text > 4000 characters. | Input validation raises error; UI displays clear message: "User input exceeds maximum length of 4000 characters". |
| 6 | Empty Input | Submit blank spaces or press Enter without typing. | `st.chat_input` prevents submission; backend rejects empty string with `InvalidInputError`. |
| 7 | Wrong API Key | Set `LLM_API_KEY=sk-invalid` in `.env` and send a message. | Displays friendly error: "Authentication failed. Please check your LLM_API_KEY in the .env file." (no traceback). |
| 8 | No Internet | Disconnect network adapter and submit a query. | After retry attempts, displays: "Failed to connect to the LLM service. Please check your network connection or base URL." |
| 9 | "New Chat" Starts Clean | Click "New chat" in sidebar. | Conversation clears immediately, message counter resets to 0, welcome screen appears. |
| 10 | "Clear Chat" Empties History | Click "Clear chat" in sidebar. | Current session messages are emptied and the session file on disk is updated. |
| 11 | "Delete Chat" Removes File | Click "Delete chat" in sidebar. | Session file is deleted from `data/sessions/`, active chat switches to remaining session or new chat. |
| 12 | Browser Refresh Resume | Refresh the browser tab during a conversation. | The most recent session is loaded automatically from disk with all previous messages preserved. |
| 13 | Saved Chat Switching | Click any conversation listed under "Saved Chats" in the sidebar. | Loads the full context and messages of that specific historical session. |

---

## 10. Configuration Table

| Environment Variable | Default Value | Description |
|---|---|---|
| `LLM_API_KEY` | *(Required)* | Authentication key for OpenAI or target API provider. Can be a dummy value (e.g. `ollama`) for local services. |
| `LLM_BASE_URL` | `None` (OpenAI default) | Optional base URL for OpenAI-compatible endpoints (e.g. `http://localhost:11434/v1`). |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier to request from the API provider. |
| `LLM_TEMPERATURE` | `0.5` | Sampling temperature between `0.0` and `2.0`. Falls back to default if non-numeric. |
| `LLM_MAX_OUTPUT_TOKENS` | `1024` | Maximum number of tokens generated in completion response. Falls back to default if non-numeric. |
| `MAX_CONTEXT_CHARS` | `24000` | Maximum character budget for conversational history context trimming. |
| `MAX_INPUT_CHARS` | `4000` | Maximum allowed character length for a single user input prompt. |
| `SESSIONS_DIR` | `data/sessions` | Directory path for persisting conversation sessions as JSON files. |

---

## 11. Design Decisions

- **Strict Separation of Concerns:** Streamlit UI code (`app.py`) contains zero LLM client logic. All orchestration is handled by `ChatService`.
- **Atomic File Persistence:** Sessions are persisted to disk using temporary files in the target directory (`tempfile.mkstemp`) and atomic replacement (`os.replace`), preventing corrupted JSON files during sudden crashes or power loss.
- **Fail-Safe History Trimming:** Trimming is computed dynamically when assembling the LLM context. Full conversational history is preserved in `session.messages` and on disk, while `ConversationMemory` ensures the prompt budget is respected and never begins with an assistant message.
- **Corrupt File Resilience:** `load_session` logs a warning and returns `None` on unparseable JSON files, ensuring that damaged disk files never crash the Streamlit interface.
- **Single Standard Client:** Uses the official `openai` Python SDK (v1+), ensuring zero dependency on complex frameworks like LangChain while maintaining full compatibility across OpenAI, Groq, OpenRouter, and Ollama.

---

## 12. Future Enhancements: Adding Document Q&A / RAG / Knowledge Base

The architecture includes a built-in extension point for Retrieval-Augmented Generation (RAG). To connect an external knowledge base or vector database (e.g., Chroma, FAISS, Pinecone):

1. **Implement `BaseRetriever`** in a new module or file:
   ```python
   from chatbot.retriever import BaseRetriever, Document

   class ChromaRetriever(BaseRetriever):
       def __init__(self, collection):
           self.collection = collection

       def retrieve(self, query: str, k: int = 3) -> list[Document]:
           results = self.collection.query(query_texts=[query], n_results=k)
           return [
               Document(content=doc, source=meta.get("source", ""))
               for doc, meta in zip(results["documents"][0], results["metadatas"][0])
           ]
   ```

2. **Inject into `ChatService`** in `app.py`:
   ```python
   # Replace NullRetriever with your custom retriever instance:
   retriever = ChromaRetriever(my_collection)
   service = ChatService(
       llm=llm,
       memory=memory,
       prompt_builder=prompt_builder,
       retriever=retriever,
       sessions_dir=settings.sessions_dir,
   )
   ```

`ChatService` and `PromptBuilder` automatically format retrieved documents into structured `<context>` blocks without requiring any changes to the core prompt pipeline.
