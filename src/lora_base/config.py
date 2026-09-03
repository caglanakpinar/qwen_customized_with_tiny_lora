"""Configuration dataclasses for the from-scratch TensorFlow LoRA pipeline.

Reuses `tiny_lora.config`'s `_merge_dataclass`/`load_yaml_config`/`DataConfig` rather than
duplicating them -- dataset loading is already framework-agnostic (it hands back a HF
`datasets.Dataset` of plain text), so nothing about it changes when the model/training backend
is TensorFlow instead of PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from tiny_lora.config import DataConfig, _merge_dataclass, load_yaml_config

__all__ = [
    "DataConfig",
    "TFModelConfig",
    "LoraBaseConfig",
    "TFTrainingConfig",
    "LoraBasePipelineConfig",
    "build_lora_base_config",
    "config_to_dict",
    "load_yaml_config",
]


@dataclass
class TFModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Compute dtype for the frozen base weights and the forward pass. LoRA's own A/B matrices
    # are always float32, regardless of this setting, since they are what gradients flow into.
    dtype: str = "bfloat16"
    # Tokenizer-only: AutoTokenizer.from_pretrained still goes through transformers, nothing in
    # the model/training path does.
    trust_remote_code: bool = False


@dataclass
class LoraBaseConfig:
    r: int = 8                    # LoRA rank -- width of the trainable A/B low-rank update
    lora_alpha: int = 16          # update is scaled by lora_alpha / r
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0     # dropout applied to the LoRA path's input during training
    seed: int = 42                # LoRA A-matrix init and dropout


@dataclass
class TFTrainingConfig:
    output_dir: str = "outputs/sft-tf"
    num_train_epochs: int = 1
    max_steps: int = -1           # -1 means "no cap, run the full num_train_epochs"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    weight_decay: float = 0.0
    adam_beta2: float = 0.999
    max_seq_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 200
    save_total_limit: int | None = None
    per_device_eval_batch_size: int = 2
    eval_steps: int = 100


@dataclass
class LoraBasePipelineConfig:
    model: TFModelConfig = field(default_factory=TFModelConfig)
    lora_base: LoraBaseConfig = field(default_factory=LoraBaseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TFTrainingConfig = field(default_factory=TFTrainingConfig)


def build_lora_base_config(raw: dict[str, Any]) -> LoraBasePipelineConfig:
    from tiny_lora.config import _flatten_data_config

    model = _merge_dataclass(TFModelConfig(), raw.get("model", {}))
    lora_base = _merge_dataclass(LoraBaseConfig(), raw.get("lora_base", {}))
    data = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training = _merge_dataclass(TFTrainingConfig(), raw.get("training", {}))
    return LoraBasePipelineConfig(model=model, lora_base=lora_base, data=data, training=training)


def config_to_dict(config: LoraBasePipelineConfig) -> dict[str, Any]:
    return {
        "model": asdict(config.model),
        "lora_base": asdict(config.lora_base),
        "data": asdict(config.data),
        "training": asdict(config.training),
    }
