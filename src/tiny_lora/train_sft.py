"""Supervised fine-tuning with TinyLoRA."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
from transformers import EarlyStoppingCallback, TrainerCallback
from transformers.trainer import TRAINER_STATE_NAME
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
from tiny_lora.gdrive import download_and_extract_zip, flatten_single_wrapper_dir
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


def _last_valid_checkpoint(output_dir: Path) -> str | None:
    """The checkpoint-with-weights-and-state check `resolve_resume_checkpoint` runs, factored out
    so it can be re-run against a freshly-downloaded outputs cache without re-triggering the
    download.

    `get_last_checkpoint` matches on the `checkpoint-N` directory name alone, so a directory left
    behind by an interrupted or partially-deleted run still counts as the latest checkpoint.
    Two things are confirmed before it is handed to the trainer:

    * weights are actually there -- otherwise the trainer refuses it outright with "Can't find a
      valid checkpoint at ...";
    * `trainer_state.json` is there too -- `Trainer.train(resume_from_checkpoint=...)` reads it
      unconditionally to restore step count, optimizer and LR-scheduler state, and raises
      `FileNotFoundError` deep inside `.train()` if it is missing. A checkpoint zipped up by hand
      (rather than by the Trainer itself) can easily drop it if only the weight files were
      selected -- re-zip the whole `checkpoint-N` directory as the Trainer wrote it to fix that.

    Reports nothing usable, rather than raising, when either is missing -- consistent with every
    other "can't resume from this" case here, which falls back to the next source instead of
    failing the run.
    """
    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint is None:
        return None

    checkpoint_dir = Path(last_checkpoint)
    if not any((checkpoint_dir / name).is_file() for name in _CHECKPOINT_WEIGHT_FILES):
        print(f"Ignoring {last_checkpoint}: no weights in it.")
        return None

    if not (checkpoint_dir / TRAINER_STATE_NAME).is_file():
        print(
            f"Ignoring {last_checkpoint}: no {TRAINER_STATE_NAME} in it, so the Trainer can't "
            "restore step/optimizer/scheduler state from it. If this came from a hand-made zip, "
            "re-zip the whole checkpoint-N directory rather than a subset of its files."
        )
        return None

    return last_checkpoint


def resolve_resume_checkpoint(
    output_dir: Path,
    gdrive_zip_file_id: str | None = None,
    gdrive_cache_dir: str | Path | None = None,
) -> str | None:
    """Return the checkpoint to resume from, or None to train from scratch.

    Tries `output_dir` first. If nothing usable is there and `gdrive_zip_file_id` is set,
    downloads and extracts that zip into `gdrive_cache_dir` (defaulting to `output_dir` itself,
    so the extracted `checkpoint-N` directories land exactly where the Trainer looks for them)
    and checks again -- e.g. a fresh machine picking up a run that was checkpointed elsewhere.
    Training starts from scratch only when neither source has a usable checkpoint. The download
    is skipped if `gdrive_cache_dir` already has a usable checkpoint of its own, so this only
    pays the download cost once per cache dir.

    The skip check looks for a checkpoint specifically, not just "cache_dir has files in it":
    when `gdrive_cache_dir` defaults to `output_dir`, that directory already holds
    pre-training artifacts by the time this runs -- `layer_expand`'s sidecar, for one -- so
    "any file present" would skip the download before a checkpoint ever had a chance to land.
    """
    last_checkpoint = _last_valid_checkpoint(output_dir)
    if last_checkpoint is not None:
        return last_checkpoint

    if not gdrive_zip_file_id:
        print(f"No checkpoint in {output_dir}. Training from the beginning.")
        return None

    cache_dir = Path(gdrive_cache_dir) if gdrive_cache_dir else output_dir
    last_checkpoint = _last_valid_checkpoint(cache_dir)
    if last_checkpoint is not None:
        print(f"{cache_dir} already has a checkpoint; skipping the Google Drive download.")
        return last_checkpoint

    print(
        f"No checkpoint in {output_dir}; downloading outputs from Google Drive "
        f"({gdrive_zip_file_id}) into {cache_dir}."
    )
    download_and_extract_zip(cache_dir, gdrive_zip_file_id, "_gdrive_outputs.zip")
    flatten_single_wrapper_dir(cache_dir)

    last_checkpoint = _last_valid_checkpoint(cache_dir)
    if last_checkpoint is None:
        print(
            f"No usable checkpoint in the downloaded outputs archive either ({cache_dir}). "
            "Training from the beginning."
        )
    return last_checkpoint


def run_sft_core(
    model,
    tokenizer,
    data_cfg: DataConfig,
    train_cfg: SFTTrainingConfig,
    resume: bool = True,
    final_dir_name: str = "adapter",
    extra_callbacks: list[TrainerCallback] | None = None,
) -> str:
    """Adapter-agnostic SFT loop: build the dataset, Trainer, and run training.

    `model` must already have its adapter attached (TinyLoRA, standard LoRA, ...) -- this
    function only knows about the generic PEFT-model/Trainer plumbing, not any particular
    adapter type. `run_sft` below is the TinyLoRA entry point; `standard_lora.train_sft` calls
    this directly with a LoRA-wrapped model instead.

    `extra_callbacks` are appended to the trainer's own. `layer_expand` uses this to write its
    shape sidecar into each `checkpoint-N` as it is saved, since a non-uniform stack cannot be
    rebuilt from config.json alone and a checkpoint without it is unloadable.

    `final_dir_name` names the subdirectory of `output_dir` the finished weights are written
    to. It defaults to "adapter" because every caller here trains one, but `layer_expand` saves
    a whole model rather than an adapter and passes "model" instead.

    `resume=False` starts a fresh run even when `output_dir` already holds `checkpoint-N`
    directories. Callers pass it when the weights in `model` came from somewhere the trainer
    does not know about -- `layer_lora`'s `init_from_checkpoint`, say -- since resuming would
    otherwise overwrite them with whatever the last checkpoint in `output_dir` happens to hold,
    and restore that run's step count and LR schedule along with it.
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

    # PEFT models report this themselves; a plain transformer (layer_expand's frozen stack plus
    # its new blocks) has no such method, so count it the same way here.
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable:,} || all params: {total:,} || "
            f"trainable%: {100 * trainable / total:.4f}"
        )

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
        neftune_noise_alpha=train_cfg.neftune_noise_alpha,
        label_smoothing_factor=train_cfg.label_smoothing_factor,
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
    callbacks.extend(extra_callbacks or [])
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
    resume_from_checkpoint = (
        resolve_resume_checkpoint(
            output_dir, train_cfg.gdrive_zip_file_id, train_cfg.gdrive_cache_dir
        )
        if resume
        else None
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_dir = output_dir / final_dir_name
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return str(final_dir)


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
