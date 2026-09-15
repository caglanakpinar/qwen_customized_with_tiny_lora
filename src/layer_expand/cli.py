"""Click CLI for the layer-expansion training pipeline.

    poetry run layer_expand sft --config configs/sft_layer_expand.yaml

No adapter options here -- no rank, no alpha, no target modules. A new transformer block has no
host projection to adapt, so the knobs are its shape and where its starting weights come from.
"""

from __future__ import annotations

from pathlib import Path

import click

from layer_expand import __version__
from layer_expand.train_sft import run_sft_from_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = PROJECT_ROOT / "configs" / "sft_layer_expand.yaml"


def parse_layers(spec: str) -> list[int]:
    """Parse a position list like "24", "24,25" or "24-26" into layer indices.

    Ranges are inclusive on both ends, since they name positions rather than slice them --
    "24-26" means three new blocks, not two.
    """
    layers: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            start_text, _, end_text = chunk.partition("-")
            try:
                start, end = int(start_text), int(end_text)
            except ValueError:
                raise click.BadParameter(f"{chunk!r} is not a layer range like '24-26'.") from None
            if start > end:
                raise click.BadParameter(f"{chunk!r} runs backwards; write it as '{end}-{start}'.")
            layers.update(range(start, end + 1))
        else:
            try:
                layers.add(int(chunk))
            except ValueError:
                raise click.BadParameter(f"{chunk!r} is not a layer index.") from None

    if not layers:
        raise click.BadParameter("no layers given.")
    return sorted(layers)


@click.group()
@click.version_option(__version__, prog_name="layer_expand")
def cli() -> None:
    """Layer expansion — add whole transformer blocks and train only those."""


@cli.command("sft")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (reads its layer_expand: section).",
)
@click.option(
    "--layers",
    default=None,
    help="Positions for the new block(s) in the final stack, e.g. '24' or '24-25'. "
    "Overrides layer_expand.layers.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option(
    "--hidden-size",
    type=int,
    default=None,
    help="Width of the new block. Defaults to the base model's; a different value wraps the "
    "block in projections and makes the saved stack non-uniform.",
)
@click.option(
    "--intermediate-size", type=int, default=None, help="MLP width of the new block."
)
@click.option(
    "--num-attention-heads",
    type=int,
    default=None,
    help="Query heads in the new block. Must satisfy hidden_size = heads * head_dim.",
)
@click.option(
    "--num-key-value-heads", type=int, default=None, help="KV heads in the new block."
)
@click.option(
    "--init",
    type=click.Choice(["identity", "random"]),
    default=None,
    help="identity (default) zeroes the block's output projections so it starts as a "
    "passthrough; random does not.",
)
@click.option(
    "--base-adapters",
    default=None,
    help="Comma-separated adapters merged into the frozen base, in order, e.g. "
    "'outputs/sft-ds-assistant,outputs/sft-layer-lora'. Each is an adapter dir or the run dir "
    "holding one. Pass 'none' to expand the stock base model.",
)
@click.option(
    "--init-from-checkpoint",
    default=None,
    help="Continue from a finished expanded model: a path, 'auto' for <output-dir>/model, "
    "or 'none'.",
)
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--max-steps", type=int, default=None, help="Cap total optimizer steps.")
@click.option("--learning-rate", type=float, default=None, help="Peak learning rate.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_cmd(
    config_path: Path,
    layers: str | None,
    model: str | None,
    hidden_size: int | None,
    intermediate_size: int | None,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
    init: str | None,
    base_adapters: str | None,
    init_from_checkpoint: str | None,
    max_samples: int | None,
    max_steps: int | None,
    learning_rate: float | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Append transformer block(s) and fine-tune only them."""
    overrides: dict = {}
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if layers is not None:
        overrides.setdefault("layer_expand", {})["layers"] = parse_layers(layers)
    if base_adapters is not None:
        entries = [part.strip() for part in base_adapters.split(",") if part.strip()]
        if len(entries) == 1 and entries[0].lower() in ("none", "off"):
            entries = []
        overrides.setdefault("layer_expand", {})["base_adapters"] = entries
    for key, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("num_attention_heads", num_attention_heads),
        ("num_key_value_heads", num_key_value_heads),
        ("init", init),
        ("init_from_checkpoint", init_from_checkpoint),
    ):
        if value is not None:
            overrides.setdefault("layer_expand", {})[key] = value
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = max_steps
    if learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = learning_rate
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir

    model_path = run_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Expanded model saved to: {model_path}")


if __name__ == "__main__":
    cli()
