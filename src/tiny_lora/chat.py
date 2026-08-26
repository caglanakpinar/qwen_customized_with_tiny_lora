"""Interactive chat REPL against a trained TinyLoRA adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click
import torch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tiny_lora.chat_memory import WRITE_EVERY_N_SUMMARIES, record_summary
from tiny_lora.knowledge_db import KnowledgeStore, load_knowledge_store
from tiny_lora.model import load_adapter_tokenizer, load_peft_adapter, resolve_adapter_base_model

EXIT_WORDS = {"exit", "quit"}
KNOWLEDGE_TOP_K = 4

_ROLE_BY_MESSAGE_TYPE = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
}

_SUMMARY_INSTRUCTION = (
    "Summarize the key points of the conversation so far as short bullet points. "
    "Keep facts, decisions, and user preferences that later replies will need. Omit small talk."
)

# Vendor and model-family names the assistant must not put in a reply. The base model underneath
# the adapter is Qwen2.5, and left to itself it introduces itself as "Qwen, created by Alibaba
# Cloud" the moment it is asked who it is -- which is both wrong for this assistant and leaks
# where it came from. Deliberately NOT in this list: "google", "meta", "gemini" and friends are
# ordinary vocabulary for a data-science assistant (Google Colab, meta-learning), so banning them
# would delete useful answers. Add them here if that trade-off goes the other way for you.
BANNED_TERMS = (
    "qwen",
    "qwen2",
    "qwen2.5",
    "tongyi",
    "qianwen",
    "alibaba",
    "alibaba cloud",
    "anthropic",
    "claude",
    "openai",
    "chatgpt",
)

# How many times to re-ask for a clean reply before falling back to deleting sentences. Each retry
# is a full generation, so this is a latency budget as much as a safety one.
GUARDRAIL_RETRIES = 2

# Sent when every retry still leaked and sentence-stripping removed the entire reply.
GUARDRAIL_FALLBACK = "I can't answer that one. Ask me something else and I'll help."

# The assistant's answer to "what are you" -- returned verbatim, not generated. An identity question
# has exactly one correct answer, so there is nothing to gain by sampling one, and a great deal to
# lose: injecting this same text as a system instruction and letting the 0.5B base model paraphrase
# it produced "I'm a data scientist trained at Oxford University" and "I'm an AI assistant here at
# Google" -- no banned term in either, both false. A model this size does not follow a persona
# instruction reliably enough to be the last word on what it is. Edit this one string to change the
# persona; pass identity_reply=None to a ChatSession to generate the answer instead.
IDENTITY_REPLY = (
    "I'm a data-science assistant. I can help with analysis, statistics, machine learning, and the "
    "Python data stack -- pandas, NumPy, scikit-learn and the rest. What are you working on?"
)

# Questions that get the model talking about itself, which is exactly when the base model
# volunteers "I am Qwen, developed by Alibaba Cloud". Matched loosely on purpose: a false positive
# only prepends a short identity note to a turn that was already about the assistant, while a false
# negative sends the leak-prone question through with no guidance at all.
_IDENTITY_QUESTION_RE = re.compile(
    r"""(
          who\s+(are|r)\s+(you|u)
        | what\s+(are|r)\s+(you|u)\s*[?!.,]|what\s+(are|r)\s+(you|u)$
        | what\s+(kind\s+of\s+)?(model|llm|ai|bot|assistant|system)\s+(are|r)\s+(you|u)
        | which\s+(model|llm|ai|company|lab)\b
        | describe\s+your\s*self
        | tell\s+me\s+about\s+your\s*self
        | introduce\s+your\s*self
        | who\s+(made|created|built|trained|developed|owns)\s+(you|u)
        | what\s*('?s|\s+is)\s+your\s+name
        | are\s+(you|u)\s+(a\s+|an\s+)?(human|person|robot|ai|bot|llm|gpt|chatgpt|qwen|claude|llama)
        | where\s+(do|did)\s+(you|u)\s+come\s+from
        | what\s+can\s+(you|u)\s+do
    )""",
    re.IGNORECASE | re.VERBOSE,
)


def _is_identity_question(text: str) -> bool:
    """True when `text` asks the assistant what or who it is."""
    return _IDENTITY_QUESTION_RE.search(text) is not None


_BANNED_TERM_RE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in BANNED_TERMS) + r")\b", re.IGNORECASE)

# A sentence is everything up to a terminator, a newline, or the end of the text. Matching rather
# than splitting keeps each terminator attached to its sentence, so dropping one sentence from a
# reply leaves the rest spaced and punctuated as the model wrote it.
_SENTENCE_RE = re.compile(r"[^.!?\n]*(?:[.!?]+|\n+|$)")


def _banned_terms_in(text: str) -> list[str]:
    """Return the distinct banned terms `text` uses, in the order they first appear."""
    seen: dict[str, None] = {}
    for match in _BANNED_TERM_RE.finditer(text):
        seen.setdefault(match.group(0).lower(), None)
    return list(seen)


def _strip_banned_sentences(text: str) -> str:
    """Drop whole sentences containing a banned term, keeping the rest of `text` intact."""
    kept = [m.group(0) for m in _SENTENCE_RE.finditer(text) if m.group(0) and not _BANNED_TERM_RE.search(m.group(0))]
    return "".join(kept).strip()


def _guardrail_instruction(found: list[str]) -> str:
    return (
        "Your previous answer used these forbidden words: "
        + ", ".join(found)
        + ". Rewrite it without them. Never name the model, company, or research lab behind you, "
        "and do not substitute a different one. If you are asked what you are, say only that you "
        "are a data-science assistant."
    )


def _to_chat_template_messages(messages: list) -> list[dict[str, str]]:
    return [{"role": _ROLE_BY_MESSAGE_TYPE[type(m)], "content": m.content} for m in messages]


def _retrieved_context_message(knowledge_store: KnowledgeStore, query: str) -> SystemMessage | None:
    """Look up `query` in `knowledge_store` and, if it found anything, wrap the hits as a SystemMessage."""
    hits = knowledge_store.search(query, top_k=KNOWLEDGE_TOP_K)
    if not hits:
        return None
    snippets = "\n\n".join(f"[{hit['metadata'].get('title', hit['id'])}]\n{hit['text']}" for hit in hits)
    return SystemMessage(content=f"Relevant knowledge from {knowledge_store.name}:\n\n{snippets}")


def _generate(model, tokenizer, messages: list, max_new_tokens: int, temperature: float = 0.0) -> str:
    prompt_text = tokenizer.apply_chat_template(
        _to_chat_template_messages(messages), tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def _generate_guarded(
    model,
    tokenizer,
    messages: list,
    max_new_tokens: int,
    temperature: float,
    retries: int = GUARDRAIL_RETRIES,
) -> str:
    """Generate a reply that names no vendor or model family.

    Regenerates up to `retries` times, each time telling the model which words it just used and
    to rewrite without them -- a retry is preferred over editing because a reply that has had its
    identity sentence cut out often no longer answers the question. Only if the last attempt still
    leaks are the offending sentences dropped outright.
    """
    reply = _generate(model, tokenizer, messages, max_new_tokens, temperature)
    for _ in range(retries):
        found = _banned_terms_in(reply)
        if not found:
            return reply
        retry_messages = [*messages, SystemMessage(content=_guardrail_instruction(found))]
        reply = _generate(model, tokenizer, retry_messages, max_new_tokens, temperature)

    if not _banned_terms_in(reply):
        return reply
    return _strip_banned_sentences(reply) or GUARDRAIL_FALLBACK


def _summarize_key_points(model, tokenizer, running_summary: str | None, messages_to_fold: list) -> str:
    """Fold `messages_to_fold` (plus any prior running summary) into an updated key-points summary."""
    context = []
    if running_summary:
        context.append(SystemMessage(content=f"Summary of the conversation so far:\n{running_summary}"))
    context.extend(messages_to_fold)
    context.append(HumanMessage(content=_SUMMARY_INSTRUCTION))
    return _generate(model, tokenizer, context, max_new_tokens=200)


def _compact_history(
    history: InMemoryChatMessageHistory,
    model,
    tokenizer,
    base_system_prompt: str | None,
    running_summary: str | None,
    summarize_after_turns: int,
    keep_recent_turns: int,
) -> tuple[str | None, bool]:
    """If history has grown past `summarize_after_turns`, fold the oldest turns into `running_summary`.

    Returns the (possibly updated) running summary and whether folding happened, and rewrites
    `history` in place to hold only the system/summary message plus the most recent
    `keep_recent_turns` exchanges.
    """
    turns = [m for m in history.messages if not isinstance(m, SystemMessage)]
    if len(turns) <= summarize_after_turns * 2:
        return running_summary, False

    keep_count = keep_recent_turns * 2
    to_fold, to_keep = turns[:-keep_count], turns[-keep_count:]
    running_summary = _summarize_key_points(model, tokenizer, running_summary, to_fold)

    summary_content = f"Key points from earlier in this conversation:\n{running_summary}"
    if base_system_prompt:
        summary_content = f"{base_system_prompt}\n\n{summary_content}"

    history.clear()
    history.add_message(SystemMessage(content=summary_content))
    for message in to_keep:
        history.add_message(message)
    return running_summary, True


def load_chat_backend(
    adapter_path: Path,
    base_model_override: str | None,
    load_in_4bit: bool,
    trust_remote_code: bool = False,
    db_path: Path | None = None,
) -> tuple[Any, Any, KnowledgeStore | None]:
    """Resolve the base model, load the tokenizer + adapter, and optionally a knowledge store.

    Shared by the CLI REPL (`run_chat`) and the web backend (`web/app.py`), so both talk to the
    model the same way instead of duplicating the loading sequence.
    """
    base_model_name = base_model_override or resolve_adapter_base_model(adapter_path)
    tokenizer = load_adapter_tokenizer(adapter_path, base_model_name, trust_remote_code)
    model = load_peft_adapter(
        base_model_name,
        str(adapter_path),
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    knowledge_store = load_knowledge_store(db_path) if db_path is not None else None
    return model, tokenizer, knowledge_store


class ChatSession:
    """One conversation's worth of state: history, running summary, and its persistence bookkeeping.

    `send` performs exactly the steps the CLI REPL's loop body used to run inline (retrieve
    context, generate, compact history, persist the summary every `WRITE_EVERY_N_SUMMARIES`th
    fold) so a web backend can hold one `ChatSession` per browser session and get identical
    behavior to the terminal chat.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        adapter_path: Path,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        summarize_after_turns: int = 6,
        keep_recent_turns: int = 2,
        memory_dir: Path = Path("outputs/chat_memory"),
        knowledge_store: KnowledgeStore | None = None,
        identity_reply: str | None = IDENTITY_REPLY,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.adapter_path = adapter_path
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.summarize_after_turns = summarize_after_turns
        self.keep_recent_turns = keep_recent_turns
        self.memory_dir = memory_dir
        self.knowledge_store = knowledge_store
        self.identity_reply = identity_reply

        self.history = InMemoryChatMessageHistory()
        if system_prompt:
            self.history.add_message(SystemMessage(content=system_prompt))
        self.running_summary: str | None = None
        self.summary_event_count = 0

    def send(self, text: str) -> str:
        """Add `text` as a user turn and return the generated reply."""
        self.history.add_message(HumanMessage(content=text))

        if self.identity_reply is not None and _is_identity_question(text):
            # Answered without generating at all -- see IDENTITY_REPLY for why. The answer still
            # joins `history`, so a follow-up ("what else can you do?") reads as a normal turn.
            reply = self.identity_reply
        else:
            # A system message that applies to this turn only, slotted in just before the user's
            # message so it is the last thing the model reads. Never added to `history`: it is
            # about this question, and carrying it forward would skew every later turn.
            generation_messages = self.history.messages
            if self.knowledge_store is not None:
                context_message = _retrieved_context_message(self.knowledge_store, text)
                if context_message is not None:
                    generation_messages = [*self.history.messages[:-1], context_message, self.history.messages[-1]]

            reply = _generate_guarded(
                self.model, self.tokenizer, generation_messages, self.max_new_tokens, self.temperature
            )
        self.history.add_message(AIMessage(content=reply))

        self.running_summary, folded = _compact_history(
            self.history,
            self.model,
            self.tokenizer,
            self.system_prompt,
            self.running_summary,
            self.summarize_after_turns,
            self.keep_recent_turns,
        )
        if folded:
            self.summary_event_count += 1
            if self.summary_event_count % WRITE_EVERY_N_SUMMARIES == 0:
                record_summary(
                    self.model,
                    self.tokenizer,
                    self.running_summary,
                    self.summary_event_count,
                    self.memory_dir,
                    source=str(self.adapter_path),
                )
        return reply


def run_chat(
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
) -> None:
    """Load a trained adapter and hand control to an interactive read-generate-print loop."""
    click.echo(f"Loading adapter {adapter_path} ...")
    if db_path is not None:
        click.echo(f"Loading knowledge base from {db_path} ...")
    model, tokenizer, knowledge_store = load_chat_backend(
        adapter_path, base_model_override, load_in_4bit, trust_remote_code, db_path
    )

    session = ChatSession(
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
    click.echo("Chat ready. Type a prompt, or 'exit'/'quit' to leave (Ctrl-D also works).\n")

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ")
        except (EOFError, click.exceptions.Abort):
            click.echo("\nExiting chat.")
            return

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            click.echo("Exiting chat.")
            return

        reply = session.send(text)
        click.echo(f"Assistant> {reply}\n")
