"""Interactive chat REPL against a trained TinyLoRA adapter."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tiny_lora.model import load_adapter_tokenizer, load_peft_adapter, resolve_adapter_base_model

EXIT_WORDS = {"exit", "quit"}

_ROLE_BY_MESSAGE_TYPE = {
    HumanMessage: "user",
    AIMessage: "assistant",
    SystemMessage: "system",
}


def _to_chat_template_messages(history: InMemoryChatMessageHistory) -> list[dict[str, str]]:
    return [{"role": _ROLE_BY_MESSAGE_TYPE[type(m)], "content": m.content} for m in history.messages]


def run_chat(
    adapter_path: Path,
    base_model_override: str | None,
    load_in_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool = False,
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

    history = InMemoryChatMessageHistory()
    if system_prompt:
        history.add_message(SystemMessage(content=system_prompt))
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
        prompt_text = tokenizer.apply_chat_template(
            _to_chat_template_messages(history), tokenize=False, add_generation_prompt=True
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
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        click.echo(f"Assistant> {reply}\n")
        history.add_message(AIMessage(content=reply))
