"""Model loading with standard LoRA adapters."""

from __future__ import annotations

from peft import get_peft_model

from standard_lora.config import StandardLoraConfig
from tiny_lora.config import ModelConfig
from tiny_lora.model import _bitsandbytes_available, load_base_model


def build_lora_config(cfg: StandardLoraConfig):
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_rslora=cfg.use_rslora,
        task_type="CAUSAL_LM",
    )


def load_lora_model(model_cfg: ModelConfig, lora_cfg: StandardLoraConfig):
    base_model = load_base_model(model_cfg)
    if model_cfg.load_in_4bit and _bitsandbytes_available():
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(base_model)

    peft_config = build_lora_config(lora_cfg)
    model = get_peft_model(base_model, peft_config)
    return model
