"""Click CLI for the TinyLoRA training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tiny_lora import __version__
from tiny_lora.config import (
    GRPOTrainingConfig,
    PipelineConfig,
    SFTTrainingConfig,
    build_pipeline_config,
    config_to_dict,
    load_yaml_config,
)
from tiny_lora.model import build_tinylora_config, load_tinylora_model, load_tokenizer
from tiny_lora.train_sft import run_sft_from_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = PROJECT_ROOT / "configs" / "sft_default.yaml"
DEFAULT_GRPO_CONFIG = PROJECT_ROOT / "configs" / "grpo_default.yaml"


def _apply_common_overrides(
    overrides: dict,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> dict:
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if u is not None:
        overrides.setdefault("tinylora", {})["u"] = u
    if r is not None:
        overrides.setdefault("tinylora", {})["r"] = r
    if weight_tying is not None:
        overrides.setdefault("tinylora", {})["weight_tying"] = weight_tying
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir
    return overrides


@click.group()
@click.version_option(__version__, prog_name="tiny-lora")
def cli() -> None:
    """TinyLoRA training pipeline — SFT and GRPO with PEFT."""


@cli.command("sft")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run supervised fine-tuning with TinyLoRA."""
    overrides = _apply_common_overrides({}, model, u, r, weight_tying, max_samples, output_dir, no_quant)
    adapter_path = run_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Adapter saved to: {adapter_path}")


@cli.command("grpo")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GRPO_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def grpo_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run GRPO reinforcement learning with TinyLoRA."""
    from tiny_lora.train_grpo import run_grpo_from_yaml  # local: trl's GRPOTrainer needs vllm

    overrides = _apply_common_overrides({}, model, u, r, weight_tying, max_samples, output_dir, no_quant)
    adapter_path = run_grpo_from_yaml(config_path, overrides)
    click.echo(f"GRPO complete. Adapter saved to: {adapter_path}")


@cli.command("chat")
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option("--model", "base_model", default=None, help="Override base model (default: read from the adapter).")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
@click.option("--max-new-tokens", type=int, default=512, show_default=True, help="Max tokens generated per reply.")
@click.option("--temperature", type=float, default=0.7, show_default=True, help="Sampling temperature; 0 for greedy.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt to prepend.")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for the base model.")
def chat_cmd(
    adapter_path: Path,
    base_model: str | None,
    no_quant: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool,
) -> None:
    """Open an interactive chat REPL with a trained TinyLoRA adapter."""
    from tiny_lora.chat import run_chat

    run_chat(
        adapter_path=adapter_path,
        base_model_override=base_model,
        load_in_4bit=not no_quant,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        trust_remote_code=trust_remote_code,
    )


@cli.command("eval")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (supplies the eval split, base model, and eval batch/seq settings).",
)
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option(
    "--max-eval-samples",
    type=int,
    default=None,
    help="Cap the eval split to this many rows (overrides data.max_eval_samples).",
)
def eval_cmd(config_path: Path, adapter_path: Path, max_eval_samples: int | None) -> None:
    """Compare base-model vs checkpoint eval loss/perplexity on the configured eval split."""
    from tiny_lora.eval import run_eval

    overrides: dict = {}
    if max_eval_samples is not None:
        overrides.setdefault("data", {})["max_eval_samples"] = max_eval_samples
    run_eval(config_path, adapter_path, overrides)


@cli.command("info")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GRPO_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization.")
@click.option("--json", "as_json", is_flag=True, help="Print config as JSON.")
def info_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    no_quant: bool,
    as_json: bool,
) -> None:
    """Print resolved config and trainable parameter count."""
    raw = load_yaml_config(config_path)
    overrides = _apply_common_overrides({}, model, u, r, weight_tying, None, None, no_quant)
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    training_cls = GRPOTrainingConfig if "grpo" in config_path.name else SFTTrainingConfig
    config: PipelineConfig = build_pipeline_config(raw, training_cls)

    if as_json:
        click.echo(json.dumps(config_to_dict(config), indent=2))
        return

    click.echo("Model config:")
    click.echo(f"  base: {config.model.model_name_or_path}")
    click.echo(f"  4-bit quant: {config.model.load_in_4bit}")
    click.echo("TinyLoRA config:")
    click.echo(f"  r={config.tinylora.r}, u={config.tinylora.u}, weight_tying={config.tinylora.weight_tying}")
    click.echo(f"  targets: {config.tinylora.target_modules}")

    click.echo("Loading model to count trainable parameters...")
    model_obj = load_tinylora_model(config.model, config.tinylora)
    model_obj.print_trainable_parameters()


@cli.command("show-config")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show_config_cmd(config_path: Path) -> None:
    """Print a YAML config file as formatted JSON."""
    raw = load_yaml_config(config_path)
    click.echo(json.dumps(raw, indent=2))


if __name__ == "__main__":
    cli()
