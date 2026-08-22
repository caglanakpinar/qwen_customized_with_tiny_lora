"""Interactive chat REPL against a trained TinyLoRA adapter."""

from __future__ import annotations

from pathlib import Path

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
    base_model_name = base_model_override or resolve_adapter_base_model(adapter_path)

    click.echo(f"Loading {base_model_name} with adapter {adapter_path} ...")
    tokenizer = load_adapter_tokenizer(adapter_path, base_model_name, trust_remote_code)
    model = load_peft_adapter(
        base_model_name,
        str(adapter_path),
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    knowledge_store = None
    if db_path is not None:
        click.echo(f"Loading knowledge base from {db_path} ...")
        knowledge_store = load_knowledge_store(db_path)

    history = InMemoryChatMessageHistory()
    if system_prompt:
        history.add_message(SystemMessage(content=system_prompt))
    running_summary: str | None = None
    summary_event_count = 0
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

        history.add_message(HumanMessage(content=text))

        generation_messages = history.messages
        if knowledge_store is not None:
            context_message = _retrieved_context_message(knowledge_store, text)
            if context_message is not None:
                generation_messages = [*history.messages[:-1], context_message, history.messages[-1]]

        reply = _generate(model, tokenizer, generation_messages, max_new_tokens, temperature)
        click.echo(f"Assistant> {reply}\n")
        history.add_message(AIMessage(content=reply))

        running_summary, folded = _compact_history(
            history,
            model,
            tokenizer,
            system_prompt,
            running_summary,
            summarize_after_turns,
            keep_recent_turns,
        )
        if folded:
            summary_event_count += 1
            if summary_event_count % WRITE_EVERY_N_SUMMARIES == 0:
                record_summary(
                    model,
                    tokenizer,
                    running_summary,
                    summary_event_count,
                    memory_dir,
                    source=str(adapter_path),
                )
