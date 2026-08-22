"""Supervised fine-tuning with TinyLoRA."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

from tiny_lora.config import PipelineConfig, SFTTrainingConfig, build_pipeline_config, config_to_dict, load_yaml_config
from tiny_lora.data import prepare_sft_dataset, prepare_sft_eval_dataset
from tiny_lora.model import load_tinylora_model, load_tokenizer


class EmptyMPSCacheCallback(TrainerCallback):
    """Release cached MPS blocks at the end of every training step.

    `on_step_end` fires immediately before the trainer's log/save/evaluate block, which is where
    macOS runs out of memory: the cache still held from the training step, plus the eval forward's
    own logits (batch x seq_len x vocab, which `ForCausalLMLoss` then upcasts to fp32), overshoots
    the MPS allocation ceiling. Handing the cached blocks back first gives eval that headroom.
    """

    def on_step_end(self, args, state, control, **kwargs) -> None:
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def run_sft(config: PipelineConfig) -> str:
    train_cfg: SFTTrainingConfig = config.training  # type: ignore[assignment]

    # Mirrors the `eval_dataset is not None` check below: `prepare_sft_eval_dataset` returns
    # None exactly when `eval_dataset_name` is unset. Checked here, before any model/data
    # loading, so a bad save_steps/eval_steps pairing fails immediately instead of after
    # several minutes of downloading and loading the base model.
    if train_cfg.early_stopping and config.data.eval_dataset_name:
        if train_cfg.save_steps % train_cfg.eval_steps != 0:
            raise ValueError(
                "training.early_stopping requires save_steps to be a round multiple of "
                f"eval_steps, but got save_steps={train_cfg.save_steps}, "
                f"eval_steps={train_cfg.eval_steps}."
            )

    tokenizer = load_tokenizer(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = load_tinylora_model(config.model, config.tinylora)
    dataset = prepare_sft_dataset(config.data, tokenizer)
    eval_dataset = prepare_sft_eval_dataset(config.data, tokenizer)

    model.print_trainable_parameters()

    if train_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # TRL renamed SFTConfig's sequence-length argument from `max_seq_length` to `max_length`
    # in 0.20. Pick whichever this install accepts so the pipeline runs on either side of it.
    length_arg = (
        "max_length"
        if "max_length" in inspect.signature(SFTConfig.__init__).parameters
        else "max_seq_length"
    )

    early_stopping = train_cfg.early_stopping and eval_dataset is not None

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        **{length_arg: train_cfg.max_seq_length},
        logging_steps=train_cfg.logging_steps,
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=train_cfg.report_to,
        dataset_text_field="text",
        # Left at the default: TRL tokenizes `text` itself, and keeping the raw column
        # afterwards hands the collator strings it cannot turn into tensors.
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=train_cfg.eval_steps if eval_dataset is not None else None,
        load_best_model_at_end=early_stopping,
        metric_for_best_model="eval_loss" if early_stopping else None,
        greater_is_better=False if early_stopping else None,
    )

    callbacks: list[TrainerCallback] = [EmptyMPSCacheCallback()]
    if early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=train_cfg.early_stopping_patience,
                early_stopping_threshold=train_cfg.early_stopping_threshold,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    last_checkpoint = get_last_checkpoint(str(output_dir))
    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return str(output_dir / "adapter")


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config = build_pipeline_config(raw, SFTTrainingConfig)
    return run_sft(config)
