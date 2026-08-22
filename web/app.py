"""FastAPI backend for the browser chat UI.

Loads one model + adapter (and optional knowledge store) at startup via `tiny_lora.chat`'s shared
`load_chat_backend`, then hands each browser tab its own `ChatSession` so requests from different
tabs don't share history. Run through `poetry run tiny-lora serve ...` (see `tiny_lora/cli.py`),
which builds the app with `create_app` and serves it with uvicorn.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tiny_lora.chat import ChatSession, load_chat_backend

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ResetRequest(BaseModel):
    session_id: str


def create_app(
    adapter_path: Path,
    base_model_override: str | None = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    summarize_after_turns: int = 6,
    keep_recent_turns: int = 2,
    memory_dir: Path = Path("outputs/chat_memory"),
    db_path: Path | None = None,
) -> FastAPI:
    """Load the model once and return a FastAPI app serving the chat UI and its `/api` routes."""
    model, tokenizer, knowledge_store = load_chat_backend(
        adapter_path, base_model_override, load_in_4bit, trust_remote_code, db_path
    )

    sessions: dict[str, ChatSession] = {}
    # `model.generate` isn't safe to call concurrently from multiple request threads; uvicorn runs
    # sync endpoints in a thread pool, so serialize generation across all sessions with one lock.
    generate_lock = threading.Lock()

    def _new_session() -> ChatSession:
        return ChatSession(
            model=model,
            tokenizer=tokenizer,
            adapter_path=adapter_path,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            summarize_after_turns=summarize_after_turns,
            keep_recent_turns=keep_recent_turns,
            memory_dir=memory_dir,
            knowledge_store=knowledge_store,
        )

    app = FastAPI(title="TinyLoRA Chat")

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        text = request.message.strip()
        if not text:
            raise HTTPException(status_code=400, detail="message must not be empty")

        session_id = request.session_id or str(uuid.uuid4())
        session = sessions.get(session_id)
        if session is None:
            session = _new_session()
            sessions[session_id] = session

        with generate_lock:
            reply = session.send(text)
        return ChatResponse(session_id=session_id, reply=reply)

    @app.post("/api/reset")
    def reset(request: ResetRequest) -> dict[str, bool]:
        sessions.pop(request.session_id, None)
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app
