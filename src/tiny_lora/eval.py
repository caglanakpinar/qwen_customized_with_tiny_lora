"""Compare base-model vs TinyLoRA-checkpoint eval loss/perplexity on the held-out split."""

from __future__ import annotations

import inspect
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import click
from trl import SFTConfig, SFTTrainer

from tiny_lora.config import PipelineConfig, SFTTrainingConfig, build_pipeline_config, load_yaml_config
from tiny_lora.data import prepare_sft_eval_dataset
from tiny_lora.model import (
    load_adapter_tokenizer,
    load_base_model,
    load_peft_adapter,
    load_tokenizer,
    resolve_adapter_base_model,
)


def _length_kwarg() -> str:
    # Same TRL 0.20 rename `train_sft.py` works around: `max_seq_length` -> `max_length`.
    return (
        "max_length"
        if "max_length" in inspect.signature(SFTConfig.__init__).parameters
        else "max_seq_length"
    )


def _eval_loss(model, tokenizer, eval_dataset, train_cfg: SFTTrainingConfig) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="tiny-lora-eval-") as scratch_dir:
        args = SFTConfig(
            output_dir=scratch_dir,
            per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
            report_to="none",
            dataset_text_field="text",
            **{_length_kwarg(): train_cfg.max_seq_length},
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=eval_dataset,  # unused: only `.evaluate()` is called below
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        click.echo(f"Evaluating {model.__class__.__name__} on {len(eval_dataset)} samples ...")
        metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    return {"eval_loss": eval_loss, "perplexity": math.exp(eval_loss)}


def _write_results(evals_dir: Path, payload: dict) -> Path:
    """Write one timestamped JSON record into `evals_dir` and return its path.

    Timestamped rather than a single fixed filename so re-evaluating the same checkpoint -- against
    a different split, sample cap or config -- accumulates a history instead of silently replacing
    the previous number. Sorting the directory by name gives them back in chronological order.
    """
    evals_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = evals_dir / f"eval-{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def run_eval(
    config_path: str | Path,
    adapter_path: str | Path,
    overrides: dict | None = None,
    evals_dir: str | Path | None = None,
    save: bool = True,
) -> dict[str, dict[str, float]]:
    """Evaluate the base model and a TinyLoRA checkpoint/adapter on the configured eval split.

    Unless `save` is False, the results are also written as JSON under `evals_dir`, which defaults
    to an `evals/` folder inside `adapter_path` -- so a checkpoint carries its own scores.
    """
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config: PipelineConfig = build_pipeline_config(raw, SFTTrainingConfig)
    train_cfg: SFTTrainingConfig = config.training  # type: ignore[assignment]
    adapter_path = Path(adapter_path)

    tokenizer = load_tokenizer(
        config.model.model_name_or_path, trust_remote_code=config.model.trust_remote_code
    )
    eval_dataset = prepare_sft_eval_dataset(config.data, tokenizer)
    if eval_dataset is None:
        raise ValueError(
            f"{config_path} has no data.eval_dataset_name set -- an eval split is required "
            "to compare eval loss."
        )

    click.echo(f"Evaluating base model {config.model.model_name_or_path} ...")
    base_model = load_base_model(config.model)
    base_metrics = _eval_loss(base_model, tokenizer, eval_dataset, train_cfg)
    del base_model

    base_model_name = resolve_adapter_base_model(adapter_path)
    adapter_tokenizer = load_adapter_tokenizer(
        adapter_path, base_model_name, config.model.trust_remote_code
    )
    click.echo(f"Evaluating checkpoint {adapter_path} ...")
    adapter_model = load_peft_adapter(
        base_model_name,
        str(adapter_path),
        load_in_4bit=config.model.load_in_4bit,
        trust_remote_code=config.model.trust_remote_code,
    )
    adapter_metrics = _eval_loss(adapter_model, adapter_tokenizer, eval_dataset, train_cfg)

    results = {"base": base_metrics, "checkpoint": adapter_metrics}

    click.echo("\nResults on the held-out eval split (lower is better):")
    click.echo(f"{'':12}{'eval_loss':>12}{'perplexity':>14}")
    for name, metrics in results.items():
        click.echo(f"{name:<12}{metrics['eval_loss']:>12.4f}{metrics['perplexity']:>14.4f}")

    if save:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "adapter": str(adapter_path),
            "base_model": base_model_name,
            "config": str(config_path),
            "eval_dataset": config.data.eval_dataset_name,
            "num_eval_samples": len(eval_dataset),
            "max_seq_length": train_cfg.max_seq_length,
            "results": results,
            # The comparison is the point of the command, so record it rather than making a
            # reader re-derive it from the two rows above.
            "delta": {
                "eval_loss": adapter_metrics["eval_loss"] - base_metrics["eval_loss"],
                "perplexity": adapter_metrics["perplexity"] - base_metrics["perplexity"],
                "perplexity_ratio": adapter_metrics["perplexity"] / base_metrics["perplexity"],
            },
        }
        out_dir = Path(evals_dir) if evals_dir is not None else adapter_path / "evals"
        click.echo(f"\nSaved to {_write_results(out_dir, payload)}")

    return results
