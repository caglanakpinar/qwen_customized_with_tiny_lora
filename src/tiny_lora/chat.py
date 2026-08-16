"""Interactive chat REPL against a trained TinyLoRA adapter."""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from tiny_lora.model import load_peft_adapter, load_tokenizer

EXIT_WORDS = {"exit", "quit"}


def _resolve_base_model_name(adapter_path: Path) -> str:
    """Read the base model id out of the adapter's own config.

    Every saved adapter -- the final one under `<output_dir>/adapter` and each periodic
    `checkpoint-N` -- carries an `adapter_config.json` with `base_model_name_or_path`, so the
    base model normally never needs to be typed by hand.
    """
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. Point --adapter at a saved "
            "adapter or checkpoint directory, e.g. outputs/sft-ds-assistant/adapter or "
            "outputs/sft-ds-assistant/checkpoint-500."
        )
    base_model_name = json.loads(adapter_config_path.read_text()).get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"{adapter_config_path} has no base_model_name_or_path.")
    return base_model_name


def _load_chat_tokenizer(adapter_path: Path, base_model_name: str, trust_remote_code: bool):
    try:
        return load_tokenizer(str(adapter_path), trust_remote_code=trust_remote_code)
    except OSError:
        # Periodic checkpoint-N dirs may not carry tokenizer files -- the final `adapter`
        # dir does (train_sft.py saves it there explicitly), but fall back to the base
        # model's tokenizer so checkpoint-N still works.
        return load_tokenizer(base_model_name, trust_remote_code=trust_remote_code)


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
    base_model_name = base_model_override or _resolve_base_model_name(adapter_path)

    click.echo(f"Loading {base_model_name} with adapter {adapter_path} ...")
    tokenizer = _load_chat_tokenizer(adapter_path, base_model_name, trust_remote_code)
    model = load_peft_adapter(
        base_model_name,
        str(adapter_path),
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
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

        messages.append({"role": "user", "content": text})
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        messages.append({"role": "assistant", "content": reply})
