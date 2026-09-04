"""Supervised fine-tuning with layer-scoped LoRA.

Reuses tiny_lora's generic SFT loop (`run_sft_core`) and config plumbing -- only the adapter
construction (`layer_lora.model.load_layer_lora_model`) and the `layer_lora:` yaml section
differ from the TinyLoRA path in `tiny_lora.train_sft`.

Note the two independent ways a run picks up earlier weights, which do different things:
`layer_lora.init_from_checkpoint` starts a *new* run from a finished adapter, while
`run_sft_core` resumes an *interrupted* run from `<output_dir>/checkpoint-N` on its own,
restoring optimizer and scheduler state with it.

They are mutually exclusive, and the explicit one wins: naming a checkpoint turns the automatic
resume off. Otherwise a run continuing from, say, `<output_dir>/checkpoint-12000` in order to
retrain one layer would have the adapter it just loaded overwritten by the last checkpoint the
trainer finds in the same directory, and would pick the old run's step count and LR schedule up
with it -- neither of which is what naming a checkpoint asked for.
"""

from __future__ import annotations

from pathlib import Path

from layer_lora.config import LayerLoraConfig
from layer_lora.model import load_layer_lora_model
from tiny_lora.config import (
    DataConfig,
    ModelConfig,
    SFTTrainingConfig,
    _flatten_data_config,
    _merge_dataclass,
    load_yaml_config,
)
from tiny_lora.model import load_tokenizer
from tiny_lora.train_sft import run_sft_core


def _guard_checkpoint_rotation(layer_cfg: LayerLoraConfig, training_cfg: SFTTrainingConfig) -> None:
    """Refuse to train if checkpoint rotation would delete the checkpoint being continued from.

    Writing into the same `output_dir` the source checkpoint lives in is the normal way to run
    this -- the point is often to land a re-tuned layer next to the run it came from. But
    `save_total_limit` deletes the lowest-numbered `checkpoint-N` directories in `output_dir` as
    new ones are written, and it does not know that one of them is this run's starting point.
    Losing it is unrecoverable, so this runs before the base model loads rather than after the
    first save. ("auto" is exempt: it names `<output_dir>/adapter`, which rotation never
    touches.)
    """
    spec = layer_cfg.init_from_checkpoint
    if spec is None or spec == "auto" or training_cfg.save_total_limit is None:
        return

    output_dir = Path(training_cfg.output_dir).resolve()
    checkpoint = Path(spec).resolve()
    if output_dir not in checkpoint.parents:
        return

    raise ValueError(
        f"{checkpoint} sits inside output_dir ({output_dir}) with "
        f"training.save_total_limit={training_cfg.save_total_limit}, so the trainer would "
        "delete it partway through this run to stay under the limit -- including the weights "
        "this run started from. Set training.save_total_limit to null to keep every "
        "checkpoint, or pass --output-dir to write this run somewhere else."
    )


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)

    model_cfg = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training_cfg = _merge_dataclass(SFTTrainingConfig(), raw.get("training", {}))
    layer_cfg = _merge_dataclass(LayerLoraConfig(), raw.get("layer_lora", {}))

    _guard_checkpoint_rotation(layer_cfg, training_cfg)

    tokenizer = load_tokenizer(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    model, init_checkpoint = load_layer_lora_model(
        model_cfg, layer_cfg, Path(training_cfg.output_dir)
    )
    return run_sft_core(model, tokenizer, data_cfg, training_cfg, resume=init_checkpoint is None)
