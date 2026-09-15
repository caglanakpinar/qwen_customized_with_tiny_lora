"""KServe model server exposing chat as a payload-in, response-out inference API.

Wraps `tiny_lora.chat.ChatSession` -- the same generation/summarization/retrieval logic the CLI
REPL (`chat`) and the browser UI (`serve`) already use -- behind KServe's V1 inference protocol, so
it can run as a standard KServe `InferenceService`:

    POST /v1/models/<name>:predict
        {"instances": [{"message": "...", "session_id": "optional"}]}
        -> {"predictions": [{"session_id": "...", "reply": "..."}]}

`kserve` is not a project dependency: it pins `protobuf>=6`, which conflicts with the
transformers/sentencepiece stack this project already depends on (`protobuf ^4.25.0`). Install it
separately in whichever environment runs `chat_api`, e.g. `pip install kserve` -- the same
arrangement `train_grpo.py` uses for `vllm`.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path
from typing import Any

import kserve

from tiny_lora.chat import ChatSession, load_chat_backend


class ChatModel(kserve.Model):
    """One loaded adapter, fanned out across per-`session_id` `ChatSession`s.

    `load()` resolves the base model and loads the tokenizer/adapter/knowledge-store once;
    `predict()` looks up (or creates) the session named by each instance's `session_id` and calls
    `ChatSession.send` on it, identically to a REPL turn or a `web/app.py` request.
    """

    def __init__(
        self,
        name: str,
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
    ) -> None:
        super().__init__(name)
        self.adapter_path = adapter_path
        self.base_model_override = base_model_override
        self.load_in_4bit = load_in_4bit
        self.trust_remote_code = trust_remote_code
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.summarize_after_turns = summarize_after_turns
        self.keep_recent_turns = keep_recent_turns
        self.memory_dir = memory_dir
        self.db_path = db_path

        self.model: Any = None
        self.tokenizer: Any = None
        self.knowledge_store: Any = None
        self.sessions: dict[str, ChatSession] = {}
        # `model.generate` isn't safe to call concurrently; serialize generation across sessions.
        self._generate_lock = threading.Lock()

    def load(self) -> bool:
        self.model, self.tokenizer, self.knowledge_store = load_chat_backend(
            self.adapter_path, self.base_model_override, self.load_in_4bit, self.trust_remote_code, self.db_path
        )
        self.ready = True
        return self.ready

    def _session_for(self, session_id: str) -> ChatSession:
        session = self.sessions.get(session_id)
        if session is None:
            session = ChatSession(
                model=self.model,
                tokenizer=self.tokenizer,
                adapter_path=self.adapter_path,
                system_prompt=self.system_prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                summarize_after_turns=self.summarize_after_turns,
                keep_recent_turns=self.keep_recent_turns,
                memory_dir=self.memory_dir,
                knowledge_store=self.knowledge_store,
            )
            self.sessions[session_id] = session
        return session

    def _send_sync(self, session_id: str, message: str) -> dict[str, str]:
        session = self._session_for(session_id)
        with self._generate_lock:
            reply = session.send(message)
        return {"session_id": session_id, "reply": reply}

    async def predict(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        instances = payload.get("instances", [])
        if not instances:
            raise ValueError("payload must include a non-empty 'instances' list")

        predictions = []
        for instance in instances:
            message = (instance.get("message") or "").strip()
            if not message:
                raise ValueError("each instance must include a non-empty 'message'")
            session_id = instance.get("session_id") or str(uuid.uuid4())
            predictions.append(await asyncio.to_thread(self._send_sync, session_id, message))
        return {"predictions": predictions}


def run_chat_api(
    adapter_path: Path,
    base_model_override: str | None,
    load_in_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool = False,
    summarize_after_turns: int = 6,
    keep_recent_turns: int = 2,
    memory_dir: Path = Path("outputs/chat_memory"),
    db_path: Path | None = None,
    model_name: str = "tinylora-chat",
    http_port: int = 8080,
) -> None:
    """Load the adapter and serve it behind KServe's V1 predict protocol until interrupted."""
    model = ChatModel(
        model_name,
        adapter_path=adapter_path,
        base_model_override=base_model_override,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        summarize_after_turns=summarize_after_turns,
        keep_recent_turns=keep_recent_turns,
        memory_dir=memory_dir,
        db_path=db_path,
    )
    model.load()
    kserve.ModelServer(http_port=http_port).start([model])
