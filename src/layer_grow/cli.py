"""Click CLI for the layer-growth training pipeline.

    poetry run layer_grow sft --config configs/sft_layer.yaml

Reuses layer_expand.cli's layer-range parsing (`parse_layers`) -- "the position(s) this round's
new block(s) take" means the same thing in both.
"""

from __future__ import annotations

from pathlib import Path

import click
from layer_expand.cli import parse_layers

from layer_grow import __version__
from layer_grow.train_sft import run_sft_from_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = PROJECT_ROOT / "configs" / "sft_layer.yaml"


@click.group()
@click.version_option(__version__, prog_name="layer_grow")
def cli() -> None:
    """Layer growth — add another block on top of an already-expanded stack and keep every
    added block trainable, not just the newest."""


@cli.command("sft")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (reads its layer_grow: section).",
)
@click.option(
    "--previous-checkpoint",
    default=None,
    help="Finished layer_expand/layer_grow checkpoint (or its run dir) to grow from. "
    "Overrides layer_grow.previous_checkpoint.",
)
@click.option(
    "--layers",
    default=None,
    help="Position(s) for this round's new block(s) in the final stack, e.g. '25'. "
    "Overrides layer_grow.layers.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option(
    "--hidden-size",
    type=int,
    default=None,
    help="Width of this round's new block. Defaults to the previous round's own width.",
)
@click.option(
    "--intermediate-size", type=int, default=None, help="MLP width of this round's new block."
)
@click.option(
    "--num-attention-heads",
    type=int,
    default=None,
    help="Query heads in this round's new block. Must satisfy hidden_size = heads * head_dim.",
)
@click.option(
    "--num-key-value-heads", type=int, default=None, help="KV heads in this round's new block."
)
@click.option(
    "--init",
    type=click.Choice(["identity", "random"]),
    default=None,
    help="identity (default) zeroes the new block's output projections so it starts as a "
    "passthrough; random does not.",
)
@click.option(
    "--init-from-checkpoint",
    default=None,
    help="Continue *this round* from a finished attempt at it: a path, 'auto' for "
    "<output-dir>/model, or 'none'.",
)
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--max-steps", type=int, default=None, help="Cap total optimizer steps.")
@click.option("--learning-rate", type=float, default=None, help="Peak learning rate.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_cmd(
    config_path: Path,
    previous_checkpoint: str | None,
    layers: str | None,
    model: str | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
    init: str | None,
    init_from_checkpoint: str | None,
    max_samples: int | None,
    max_steps: int | None,
    learning_rate: float | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Grow an already-expanded stack by one more block, fine-tuning it and every earlier one."""
    overrides: dict = {}
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if previous_checkpoint is not None:
        overrides.setdefault("layer_grow", {})["previous_checkpoint"] = previous_checkpoint
    if layers is not None:
        overrides.setdefault("layer_grow", {})["layers"] = parse_layers(layers)
    for key, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("init", init),
        ("init_from_checkpoint", init_from_checkpoint),
    ):
        if value is not None:
            overrides.setdefault("layer_grow", {})[key] = value
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = max_steps
    if learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = learning_rate
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir

    model_path = run_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Grown model saved to: {model_path}")


if __name__ == "__main__":
    cli()
