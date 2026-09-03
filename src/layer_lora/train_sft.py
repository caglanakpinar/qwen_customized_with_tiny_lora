"""Supervised fine-tuning with layer-scoped LoRA.

Reuses tiny_lora's generic SFT loop (`run_sft_core`) and config plumbing -- only the adapter
construction (`layer_lora.model.load_layer_lora_model`) and the `layer_lora:` yaml section
differ from the TinyLoRA path in `tiny_lora.train_sft`.

Note the two independent ways a run picks up earlier weights, which do different things:
`layer_lora.init_from_checkpoint` starts a *new* run from a finished adapter, while
`run_sft_core` resumes an *interrupted* run from `<output_dir>/checkpoint-N` on its own,
restoring optimizer and scheduler state with it. When an interrupted run is present in
output_dir, that resume takes precedence -- it is the more complete restore of the two.
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


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)

    model_cfg = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training_cfg = _merge_dataclass(SFTTrainingConfig(), raw.get("training", {}))
    layer_cfg = _merge_dataclass(LayerLoraConfig(), raw.get("layer_lora", {}))

    tokenizer = load_tokenizer(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    model = load_layer_lora_model(model_cfg, layer_cfg, Path(training_cfg.output_dir))
    return run_sft_core(model, tokenizer, data_cfg, training_cfg)
