"""GRPO reinforcement learning with TinyLoRA."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from transformers import TrainerCallback

# `trl.trainer.grpo_trainer` does `if is_vllm_available(): from vllm import LLM, SamplingParams`
# at module import time (confirmed directly against the trl==0.14.0 wheel from PyPI -- a
# universal py3-none-any build, identical on every platform, so this isn't a Linux/Colab-only
# copy). `is_vllm_available()` is just `_vllm_available`, a module-level bool trl.import_utils
# computes once via `importlib.util.find_spec("vllm") is not None` the first time it's imported
# in a process. Seen on a Colab box: that check reports vllm present even though `import vllm`
# itself then raises "No module named 'vllm'" moments later in the very same process -- crashing
# this whole import before training starts. We never call GRPOConfig(use_vllm=...) anywhere in
# this codebase, so trl's vllm-backed generation path is never exercised regardless of whether
# vllm is actually installed; forcing the flag off here (before trl's own submodules import,
# so nothing has consumed the stale True yet) sidesteps whatever makes that detection unreliable
# on some machines, without patching trl itself.
import trl.import_utils

trl.import_utils._vllm_available = False
from trl import GRPOConfig, GRPOTrainer

from tiny_lora.config import (
    DataConfig,
    GRPOTrainingConfig,
    PipelineConfig,
    build_pipeline_config,
    load_yaml_config,
)
from tiny_lora.data import prepare_grpo_dataset
from tiny_lora.model import build_tinylora_config, load_tokenizer
from tiny_lora.rewards import (
    calculation_accuracy_reward,
    correctness_reward,
    format_reward,
    length_reward,
    reasoning_step_reward,
    repetition_penalty_reward,
)
from tiny_lora.train_sft import resolve_resume_checkpoint

DEFAULT_REWARD_FUNCS: list[Callable] = [
    correctness_reward,
    format_reward,
    length_reward,
    reasoning_step_reward,
    calculation_accuracy_reward,
    repetition_penalty_reward,
]


def run_grpo_core(
    model,
    tokenizer,
    data_cfg: DataConfig,
    train_cfg: GRPOTrainingConfig,
    reward_funcs: list[Callable],
    resume: bool = True,
    final_dir_name: str = "adapter",
    extra_callbacks: list[TrainerCallback] | None = None,
    peft_config=None,
) -> str:
    """Adapter-agnostic GRPO loop: build the dataset, Trainer, and run training.

    Mirrors `run_sft_core`'s split between the generic Trainer plumbing and however the model got
    its trainable parameters. `model` may be a model name/path string -- GRPOTrainer loads it and
    applies `peft_config` itself, the plain TinyLoRA case `run_grpo` below drives -- or an
    already-built module whose trainable/frozen parameters are already set, in which case
    `peft_config` is left None; `layer_grow.train_grpo` passes a grown stack this way, the same
    division of labour `layer_grow.train_sft` has with `run_sft_core`.

    `extra_callbacks`/`final_dir_name` mean the same as in `run_sft_core`. `resume=False` starts a
    fresh run even when `output_dir` already holds `checkpoint-N` directories -- same reasoning:
    the weights in `model` came from somewhere the trainer does not know about.
    """
    dataset = prepare_grpo_dataset(data_cfg)

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = GRPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.num_train_epochs,
        max_steps=train_cfg.max_steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        max_prompt_length=train_cfg.max_prompt_length,
        max_completion_length=train_cfg.max_completion_length,
        num_generations=train_cfg.num_generations,
        logging_steps=train_cfg.logging_steps,
        save_steps=train_cfg.save_steps,
        save_total_limit=train_cfg.save_total_limit,
        bf16=train_cfg.bf16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=train_cfg.report_to,
        remove_unused_columns=False,
    )

    # trl==0.14.0's GRPOTrainer.__init__ does `model.warnings_issued["estimate_tokens"] = True`
    # unconditionally, expecting `PreTrainedModel.__init__` to have already set that dict --
    # true in the transformers 4.x trl==0.14.0 was written against, but transformers 5.15.0
    # (confirmed: `grep -r warnings_issued` over its whole source turns up nothing) dropped the
    # attribute entirely along with whatever warning-suppression system used it. Same shape of
    # incompatibility as the vllm one above: trl reaching for something transformers no longer
    # has. Setting it here, rather than patching trl, keeps the fix local to us.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
        callbacks=extra_callbacks or None,
    )

    # PEFT models (the plain TinyLoRA case) report this themselves; a plain transformer (a grown
    # stack, which sets requires_grad directly rather than through PEFT) has no such method --
    # same fallback run_sft_core uses.
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )

    resume_from_checkpoint = (
        resolve_resume_checkpoint(output_dir, train_cfg.gdrive_zip_file_id, train_cfg.gdrive_cache_dir)
        if resume
        else None
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / final_dir_name
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return str(final_dir)


def run_grpo(config: PipelineConfig) -> str:
    """TinyLoRA GRPO entry point: attach a TinyLoRA adapter, then run the shared GRPO loop."""
    train_cfg: GRPOTrainingConfig = config.training  # type: ignore[assignment]

    tokenizer = load_tokenizer(
        config.model.model_name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    peft_config = build_tinylora_config(config.tinylora)

    return run_grpo_core(
        config.model.model_name_or_path,
        tokenizer,
        config.data,
        train_cfg,
        reward_funcs=DEFAULT_REWARD_FUNCS,
        peft_config=peft_config,
    )


def run_grpo_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config = build_pipeline_config(raw, GRPOTrainingConfig)
    return run_grpo(config)
