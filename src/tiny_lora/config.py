"""Configuration loading and dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TinyLoraConfig:
    r: int = 2
    u: int = 32
    weight_tying: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    projection_seed: int = 42
    save_projection: bool = True
    init_weights: str | bool = "uniform"
    init_v_bound: float = 0.02
    tinylora_dropout: float = 0.0


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    load_in_4bit: bool = True
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    # "local" reads dataset_name/eval_dataset_name straight from disk (or the Hub). "gdrive"
    # downloads a zip from Google Drive to gdrive_cache_dir first, then reads it exactly the
    # same way -- dataset_name/eval_dataset_name should point at the paths the zip extracts to.
    reader: str = "local"
    dataset_name: str = "openai/gsm8k"
    dataset_config: str | None = "main"
    split: str = "train"
    max_samples: int | None = None
    prompt_column: str = "question"
    answer_column: str = "answer"
    # Optional held-out split. A Hub repo id or a local .json/.jsonl path, as with
    # `dataset_name`. When set, SFT reports eval loss alongside training loss.
    eval_dataset_name: str | None = None
    # Cap on rows read from eval_dataset_name -- unlike max_samples (training only), this
    # applies to eval regardless of size, since scoring the full split on every eval_steps
    # (or in `tiny-lora eval`, which loads two full models) is often too slow to be worth it.
    max_eval_samples: int | None = None
    # Google Drive file id (or full share URL) of the dataset zip. Required when reader="gdrive".
    gdrive_zip_file_id: str | None = None
    # Where the zip is extracted to. Downloaded once and reused on later runs if already
    # populated, so dataset_name/eval_dataset_name should point inside this directory.
    gdrive_cache_dir: str = "data/synthetic/dataset"


@dataclass
class SFTTrainingConfig:
    output_dir: str = "outputs/sft"
    num_train_epochs: int = 1
    # Caps total optimizer steps regardless of dataset size / num_train_epochs; -1 (the
    # HF Trainer default) means "no cap, run the full num_train_epochs".
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 100
    per_device_eval_batch_size: int = 2
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
    # Early stopping only engages when data.eval_dataset_name is set -- it compares checkpoints
    # by eval_loss, which needs an eval split to compute. When enabled, `save_steps` must be a
    # round multiple of `eval_steps` (Trainer requirement for load_best_model_at_end).
    early_stopping: bool = False
    # Consecutive non-improving evals allowed before training stops. 1 is the strictest
    # possible value: training stops the moment a single eval fails to improve on the best
    # eval_loss seen so far.
    early_stopping_patience: int = 1
    # Minimum decrease in eval_loss to count as an improvement. 0.0 is the strictest possible
    # value: any eval_loss that is not strictly lower than the best seen so far counts as
    # non-improving.
    early_stopping_threshold: float = 0.0


@dataclass
class GRPOTrainingConfig:
    output_dir: str = "outputs/grpo"
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    max_prompt_length: int = 256
    max_completion_length: int = 512
    num_generations: int = 4
    logging_steps: int = 1
    save_steps: int = 100
    bf16: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"


@dataclass
class PipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tinylora: TinyLoraConfig = field(default_factory=TinyLoraConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: SFTTrainingConfig | GRPOTrainingConfig = field(default_factory=SFTTrainingConfig)


def _merge_dataclass(instance: Any, overrides: dict[str, Any]) -> Any:
    valid = {f.name for f in fields(instance)}
    for key, value in overrides.items():
        if key in valid and value is not None:
            setattr(instance, key, value)
    return instance


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _flatten_data_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten the `data.gdrive.*` yaml block into the `gdrive_*` DataConfig fields."""
    flat = dict(raw)
    gdrive = flat.pop("gdrive", {}) or {}
    for key, value in gdrive.items():
        flat[f"gdrive_{key}"] = value
    return flat


def build_pipeline_config(
    raw: dict[str, Any],
    training_cls: type[SFTTrainingConfig] | type[GRPOTrainingConfig],
) -> PipelineConfig:
    model = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    tinylora = _merge_dataclass(TinyLoraConfig(), raw.get("tinylora", {}))
    data = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training = _merge_dataclass(training_cls(), raw.get("training", {}))
    return PipelineConfig(model=model, tinylora=tinylora, data=data, training=training)


def config_to_dict(config: PipelineConfig) -> dict[str, Any]:
    return {
        "model": asdict(config.model),
        "tinylora": asdict(config.tinylora),
        "data": asdict(config.data),
        "training": asdict(config.training),
    }
