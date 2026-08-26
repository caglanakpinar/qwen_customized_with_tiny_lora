"""Supervised fine-tuning with standard LoRA.

Reuses tiny_lora's generic SFT loop (`run_sft_core`) and config plumbing -- only the adapter
construction (`standard_lora.model.load_lora_model`) and the `standard_lora:` yaml section differ
from the TinyLoRA path in `tiny_lora.train_sft`.
"""

from __future__ import annotations

from pathlib import Path

from standard_lora.config import StandardLoraConfig
from standard_lora.model import load_lora_model
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
    lora_cfg = _merge_dataclass(StandardLoraConfig(), raw.get("standard_lora", {}))

    tokenizer = load_tokenizer(model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code)
    model = load_lora_model(model_cfg, lora_cfg)
    return run_sft_core(model, tokenizer, data_cfg, training_cfg)
