"""GRPO reinforcement learning with TinyLoRA."""

from __future__ import annotations

from pathlib import Path

from trl import GRPOConfig, GRPOTrainer

from tiny_lora.config import GRPOTrainingConfig, PipelineConfig, build_pipeline_config, load_yaml_config
from tiny_lora.data import prepare_grpo_dataset
from tiny_lora.model import build_tinylora_config, load_tokenizer
from tiny_lora.rewards import correctness_reward, format_reward


def run_grpo(config: PipelineConfig) -> str:
    train_cfg: GRPOTrainingConfig = config.training  # type: ignore[assignment]

    tokenizer = load_tokenizer(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    dataset = prepare_grpo_dataset(config.data)
    peft_config = build_tinylora_config(config.tinylora)

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        max_prompt_length=train_cfg.max_prompt_length,
        max_completion_length=train_cfg.max_completion_length,
        num_generations=train_cfg.num_generations,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=train_cfg.report_to,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=config.model.model_name_or_path,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=[correctness_reward, format_reward],
    )
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(str(output_dir / "adapter"))
    return str(output_dir / "adapter")


def run_grpo_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config = build_pipeline_config(raw, GRPOTrainingConfig)
    return run_grpo(config)
