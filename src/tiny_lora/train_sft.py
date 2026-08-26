"""Supervised fine-tuning with TinyLoRA."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from trl import SFTConfig, SFTTrainer

from tiny_lora.config import (
    DataConfig,
    PipelineConfig,
    SFTTrainingConfig,
    build_pipeline_config,
    load_yaml_config,
)
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


# The weight files a resumable checkpoint may carry. Adapter runs write the `adapter_*` pair rather
# than a full model, and sharded saves write an index instead of a single blob, so any one of these
# is enough for the trainer to restore from.
_CHECKPOINT_WEIGHT_FILES = (
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    ADAPTER_WEIGHTS_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
)


def resolve_resume_checkpoint(output_dir: Path) -> str | None:
    """Return the last checkpoint in `output_dir` to resume from, or None to train from scratch.

    `get_last_checkpoint` matches on the `checkpoint-N` directory name alone, so a directory left
    behind by an interrupted or partially-deleted run still counts as the latest checkpoint -- and
    the trainer then refuses it with "Can't find a valid checkpoint at ...". Confirm the weights are
    actually there and start from the beginning when they are not.
    """
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint is None:
        return None

    checkpoint_dir = Path(last_checkpoint)
    if any((checkpoint_dir / name).is_file() for name in _CHECKPOINT_WEIGHT_FILES):
        return last_checkpoint

    print(f"Ignoring {last_checkpoint}: no weights in it. Training from the beginning.")
    return None


def run_sft_core(model, tokenizer, data_cfg: DataConfig, train_cfg: SFTTrainingConfig) -> str:
    """Adapter-agnostic SFT loop: build the dataset, Trainer, and run training.

    `model` must already have its adapter attached (TinyLoRA, standard LoRA, ...) -- this
    function only knows about the generic PEFT-model/Trainer plumbing, not any particular
    adapter type. `run_sft` below is the TinyLoRA entry point; `standard_lora.train_sft` calls
    this directly with a LoRA-wrapped model instead.
    """
    # Mirrors the `eval_dataset is not None` check below: `prepare_sft_eval_dataset` returns
    # None exactly when `eval_dataset_name` is unset. Checked here, before any model/data
    # loading, so a bad save_steps/eval_steps pairing fails immediately instead of after
    # several minutes of downloading and loading the base model.
    if train_cfg.early_stopping and data_cfg.eval_dataset_name:
        if train_cfg.save_steps % train_cfg.eval_steps != 0:
            raise ValueError(
                "training.early_stopping requires save_steps to be a round multiple of "
                f"eval_steps, but got save_steps={train_cfg.save_steps}, "
                f"eval_steps={train_cfg.eval_steps}."
            )

    dataset = prepare_sft_dataset(data_cfg, tokenizer)
    eval_dataset = prepare_sft_eval_dataset(data_cfg, tokenizer)

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
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_steps=train_cfg.warmup_steps,
        max_grad_norm=train_cfg.max_grad_norm,
        weight_decay=train_cfg.weight_decay,
        adam_beta2=train_cfg.adam_beta2,
        **{length_arg: train_cfg.max_seq_length},
        logging_steps=train_cfg.logging_steps,
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
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
    trainer.train(resume_from_checkpoint=resolve_resume_checkpoint(output_dir))
    trainer.save_model(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    return str(output_dir / "adapter")


def run_sft(config: PipelineConfig) -> str:
    """TinyLoRA SFT entry point: attach a TinyLoRA adapter, then run the shared SFT loop."""
    tokenizer = load_tokenizer(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    model = load_tinylora_model(config.model, config.tinylora)
    return run_sft_core(model, tokenizer, config.data, config.training)


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config = build_pipeline_config(raw, SFTTrainingConfig)
    return run_sft(config)
