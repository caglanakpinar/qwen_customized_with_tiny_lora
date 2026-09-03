"""Supervised fine-tuning with from-scratch LoRA on TensorFlow -- no trl SFTTrainer/GRPOTrainer.

Everything a `Trainer` would normally do -- the loss, the LR schedule, the gradient-accumulation
step, checkpointing -- is written out explicitly below instead of delegated to one.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import tensorflow as tf

from lora_base.adapter_io import ADAPTER_WEIGHTS_NAME, load_adapter, save_adapter
from lora_base.config import (
    DataConfig,
    LoraBaseConfig,
    LoraBasePipelineConfig,
    TFModelConfig,
    TFTrainingConfig,
    build_lora_base_config,
    load_yaml_config,
)
from lora_base.data_tf import IGNORE_INDEX, prepare_tf_sft_datasets
from lora_base.qwen_model import QwenForCausalLM, build_qwen_lora_model, print_trainable_parameters

_PROGRESS_FILE = "trainer_progress.json"


def causal_lm_loss(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    """Next-token cross-entropy, masked at `IGNORE_INDEX` -- a from-scratch `ForCausalLMLoss`."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = tf.not_equal(shift_labels, IGNORE_INDEX)
    safe_labels = tf.where(mask, shift_labels, tf.zeros_like(shift_labels))
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=safe_labels, logits=shift_logits)
    losses = tf.where(mask, losses, tf.zeros_like(losses))
    denom = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.float32)), 1.0)
    return tf.reduce_sum(losses) / denom


class WarmupLinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Linear warmup to `peak_lr` over `warmup_steps`, then linear decay to 0 by `total_steps`."""

    def __init__(self, peak_lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = max(warmup_steps, 0)
        self.total_steps = max(total_steps, 1)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        total_steps = tf.cast(self.total_steps, tf.float32)
        warmup_lr = self.peak_lr * (step / tf.maximum(warmup_steps, 1.0))
        decay_progress = (step - warmup_steps) / tf.maximum(total_steps - warmup_steps, 1.0)
        decay_lr = self.peak_lr * tf.maximum(0.0, 1.0 - decay_progress)
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self) -> dict:
        return {"peak_lr": self.peak_lr, "warmup_steps": self.warmup_steps, "total_steps": self.total_steps}


def _run_eval(model: QwenForCausalLM, eval_dataset: tf.data.Dataset) -> float:
    total_loss, count = 0.0, 0
    for batch, labels in eval_dataset:
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"], training=False)
        total_loss += float(causal_lm_loss(logits, labels))
        count += 1
    return total_loss / max(count, 1)


def _checkpoints(output_dir: Path) -> list[tuple[int, Path]]:
    found = []
    for path in output_dir.glob("checkpoint-*"):
        if not (path / ADAPTER_WEIGHTS_NAME).exists():
            continue
        try:
            found.append((int(path.name.split("-")[-1]), path))
        except ValueError:
            continue
    return sorted(found, key=lambda item: item[0])


def _resolve_resume_checkpoint(model: QwenForCausalLM, output_dir: Path) -> tuple[int, int]:
    """Load the highest `checkpoint-N` with weights present; returns `(global_step, epoch)`.

    Resumes LoRA weights only -- a documented simplification vs. the HF Trainer's full
    optimizer-state resume (the optimizer, including its LR schedule step, restarts fresh).
    """
    if not output_dir.is_dir():
        return 0, 0
    checkpoints = _checkpoints(output_dir)
    if not checkpoints:
        return 0, 0

    step, path = checkpoints[-1]
    load_adapter(model, path)
    progress_path = path / _PROGRESS_FILE
    epoch = json.loads(progress_path.read_text())["epoch"] if progress_path.exists() else 0
    print(f"Resuming from {path} (global_step={step})")
    return step, epoch


def _prune_checkpoints(output_dir: Path, save_total_limit: int | None) -> None:
    if save_total_limit is None:
        return
    checkpoints = _checkpoints(output_dir)
    excess = len(checkpoints) - save_total_limit
    for _, path in checkpoints[: max(excess, 0)]:
        shutil.rmtree(path)


def run_sft_core(
    model: QwenForCausalLM,
    tokenizer,
    model_cfg: TFModelConfig,
    lora_cfg: LoraBaseConfig,
    data_cfg: DataConfig,
    train_cfg: TFTrainingConfig,
) -> str:
    train_tf, eval_tf, num_train_examples = prepare_tf_sft_datasets(
        data_cfg,
        tokenizer,
        train_cfg.max_seq_length,
        train_cfg.per_device_train_batch_size,
        train_cfg.per_device_eval_batch_size,
    )
    print_trainable_parameters(model)

    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    micro_batches_per_epoch = math.ceil(num_train_examples / train_cfg.per_device_train_batch_size)
    opt_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / train_cfg.gradient_accumulation_steps))
    total_steps = (
        train_cfg.max_steps if train_cfg.max_steps > 0 else opt_steps_per_epoch * train_cfg.num_train_epochs
    )

    lr_schedule = WarmupLinearDecay(train_cfg.learning_rate, train_cfg.warmup_steps, total_steps)
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule, beta_2=train_cfg.adam_beta2, weight_decay=train_cfg.weight_decay
    )

    trainable_vars = model.trainable_variables
    accum_grads = [tf.Variable(tf.zeros_like(v), trainable=False) for v in trainable_vars]
    accumulation_steps = tf.constant(train_cfg.gradient_accumulation_steps, dtype=tf.float32)

    @tf.function
    def forward_backward(input_ids, attention_mask, labels):
        with tf.GradientTape() as tape:
            logits = model(input_ids, attention_mask=attention_mask, training=True)
            loss = causal_lm_loss(logits, labels)
            scaled_loss = loss / accumulation_steps
        grads = tape.gradient(scaled_loss, trainable_vars)
        for acc, g in zip(accum_grads, grads):
            if g is not None:
                acc.assign_add(g)
        return loss

    @tf.function
    def optimizer_step():
        grads = [acc.read_value() for acc in accum_grads]
        grads, _ = tf.clip_by_global_norm(grads, train_cfg.max_grad_norm)
        optimizer.apply_gradients(zip(grads, trainable_vars))
        for acc in accum_grads:
            acc.assign(tf.zeros_like(acc))

    global_step, start_epoch = _resolve_resume_checkpoint(model, output_dir)
    micro_step = 0
    running_loss, running_count = 0.0, 0

    for epoch in range(start_epoch, train_cfg.num_train_epochs):
        for batch, labels in train_tf:
            loss = forward_backward(batch["input_ids"], batch["attention_mask"], labels)
            running_loss += float(loss)
            running_count += 1
            micro_step += 1

            if micro_step % train_cfg.gradient_accumulation_steps != 0:
                continue
            optimizer_step()
            global_step += 1

            if global_step % train_cfg.logging_steps == 0:
                lr = float(lr_schedule(global_step))
                print(f"step {global_step}/{total_steps}  loss={running_loss / running_count:.4f}  lr={lr:.2e}")
                running_loss, running_count = 0.0, 0

            if eval_tf is not None and global_step % train_cfg.eval_steps == 0:
                print(f"step {global_step}  eval_loss={_run_eval(model, eval_tf):.4f}")

            if global_step % train_cfg.save_steps == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                save_adapter(model, model_cfg, lora_cfg, checkpoint_dir)
                tokenizer.save_pretrained(str(checkpoint_dir))
                (checkpoint_dir / _PROGRESS_FILE).write_text(json.dumps({"epoch": epoch, "global_step": global_step}))
                _prune_checkpoints(output_dir, train_cfg.save_total_limit)

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    adapter_dir = output_dir / "adapter"
    save_adapter(model, model_cfg, lora_cfg, adapter_dir)
    tokenizer.save_pretrained(str(adapter_dir))
    return str(adapter_dir)


def run_sft(config: LoraBasePipelineConfig) -> str:
    from tiny_lora.model import load_tokenizer

    tokenizer = load_tokenizer(config.model.model_name_or_path, trust_remote_code=config.model.trust_remote_code)
    model = build_qwen_lora_model(config.model, config.lora_base)
    return run_sft_core(model, tokenizer, config.model, config.lora_base, config.data, config.training)


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config = build_lora_base_config(raw)
    return run_sft(config)
