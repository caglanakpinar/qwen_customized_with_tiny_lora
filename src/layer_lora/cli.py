"""Click CLI for the layer-scoped LoRA training pipeline.

    poetry run layer_lora sft --config configs/sft_layer_lora.yaml
"""

from __future__ import annotations

from pathlib import Path

import click

from layer_lora import __version__
from layer_lora.train_sft import run_sft_from_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = PROJECT_ROOT / "configs" / "sft_layer_lora.yaml"


def parse_layers(spec: str) -> list[int]:
    """Parse a layer selection like "7", "0,1,2" or "0-3,11,20-23" into layer indices.

    Ranges are inclusive on both ends, since they name layers rather than slice them --
    "20-23" on a 24-layer model means the last four layers, not three of them.
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
                raise click.BadParameter(f"{chunk!r} is not a layer range like '20-23'.") from None
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
@click.version_option(__version__, prog_name="layer_lora")
def cli() -> None:
    """Layer-scoped LoRA — fine-tune named transformer layers, leaving the rest untouched."""


@cli.command("sft")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (reads its layer_lora: section).",
)
@click.option(
    "--layers",
    default=None,
    help="Layers to adapt, e.g. '7', '0,1,2' or '20-23'. Overrides layer_lora.layers.",
)
@click.option(
    "--target-modules",
    default=None,
    help="Comma-separated projections to adapt, e.g. 'q_proj,k_proj,v_proj'.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--r", type=int, default=None, help="LoRA rank.")
@click.option("--lora-alpha", type=int, default=None, help="LoRA alpha (scaling numerator).")
@click.option(
    "--init-from-checkpoint",
    default=None,
    help="Continue from a saved adapter: a path, or 'auto' for <output-dir>/adapter.",
)
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--max-steps", type=int, default=None, help="Cap total optimizer steps.")
@click.option("--learning-rate", type=float, default=None, help="Peak learning rate.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_cmd(
    config_path: Path,
    layers: str | None,
    target_modules: str | None,
    model: str | None,
    r: int | None,
    lora_alpha: int | None,
    init_from_checkpoint: str | None,
    max_samples: int | None,
    max_steps: int | None,
    learning_rate: float | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run supervised fine-tuning on specific layers with LoRA."""
    overrides: dict = {}
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if layers is not None:
        overrides.setdefault("layer_lora", {})["layers"] = parse_layers(layers)
    if target_modules is not None:
        overrides.setdefault("layer_lora", {})["target_modules"] = [
            part.strip() for part in target_modules.split(",") if part.strip()
        ]
    if r is not None:
        overrides.setdefault("layer_lora", {})["r"] = r
    if lora_alpha is not None:
        overrides.setdefault("layer_lora", {})["lora_alpha"] = lora_alpha
    if init_from_checkpoint is not None:
        overrides.setdefault("layer_lora", {})["init_from_checkpoint"] = init_from_checkpoint
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if max_steps is not None:
        overrides.setdefault("training", {})["max_steps"] = max_steps
    if learning_rate is not None:
        overrides.setdefault("training", {})["learning_rate"] = learning_rate
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir

    adapter_path = run_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Adapter saved to: {adapter_path}")


if __name__ == "__main__":
    cli()
