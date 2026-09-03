"""Save/load the lora_base adapter format -- this package's own, not PEFT's.

`adapter_config.json` + `lora_weights.npz` (every `lora_A`/`lora_B`, keyed by layer path).
Cannot be loaded by `tiny_lora.model.load_peft_adapter` / `tiny-lora chat` -- those expect a
PEFT `adapter_model.safetensors` + `adapter_config.json` pair, which this training path never
produces since no `peft` is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import tensorflow as tf

from lora_base.config import LoraBaseConfig, TFModelConfig
from lora_base.layers import LoRADense
from lora_base.qwen_model import QwenForCausalLM

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "lora_weights.npz"


def _lora_layers(model: QwenForCausalLM) -> Iterator[tuple[str, LoRADense]]:
    for layer_idx, layer in enumerate(model.layers_list):
        for module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            module = getattr(layer.self_attn, module_name)
            if isinstance(module, LoRADense):
                yield f"layers.{layer_idx}.self_attn.{module_name}", module
        for module_name in ("gate_proj", "up_proj", "down_proj"):
            module = getattr(layer.mlp, module_name)
            if isinstance(module, LoRADense):
                yield f"layers.{layer_idx}.mlp.{module_name}", module


def save_adapter(
    model: QwenForCausalLM,
    model_cfg: TFModelConfig,
    lora_cfg: LoraBaseConfig,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = {}
    for key, module in _lora_layers(model):
        weights[f"{key}.lora_A"] = module.lora_A.numpy()
        weights[f"{key}.lora_B"] = module.lora_B.numpy()
    np.savez(output_dir / ADAPTER_WEIGHTS_NAME, **weights)

    config = {
        "base_model_name_or_path": model_cfg.model_name_or_path,
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.lora_alpha,
        "lora_dropout": lora_cfg.lora_dropout,
        "target_modules": lora_cfg.target_modules,
        "seed": lora_cfg.seed,
    }
    (output_dir / ADAPTER_CONFIG_NAME).write_text(json.dumps(config, indent=2))


def load_adapter(model: QwenForCausalLM, adapter_dir: str | Path) -> None:
    adapter_dir = Path(adapter_dir)
    weights_path = adapter_dir / ADAPTER_WEIGHTS_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"No {ADAPTER_WEIGHTS_NAME} in {adapter_dir}.")

    weights = np.load(weights_path)
    for key, module in _lora_layers(model):
        module.lora_A.assign(tf.constant(weights[f"{key}.lora_A"]))
        module.lora_B.assign(tf.constant(weights[f"{key}.lora_B"]))
