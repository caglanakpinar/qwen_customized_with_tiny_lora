"""Supervised fine-tuning of appended transformer blocks.

Reuses tiny_lora's SFT loop (`run_sft_core`) and config plumbing, as the LoRA modules do. What
differs is the model handed to it: a plain expanded transformer with everything frozen except
the new blocks, rather than a PEFT-wrapped base -- and consequently a `model/` directory at the
end instead of an `adapter/` one.

Three separate ways a run picks up earlier weights, which do different things:

* `layer_expand.base_adapters` merges the finished adapters into the *frozen* base, in order,
  before the new block is added. This is the "start from my tuned model" setting, and it is a
  chain: the TinyLoRA run covers all 24 layers, the layer_lora run refines layer 21 on top.
* `layer_expand.init_from_checkpoint` starts a new run from a *finished* expanded model.
* `run_sft_core` resumes an *interrupted* run from `<output_dir>/checkpoint-N` by itself,
  restoring optimizer and scheduler state with it.

The last two are mutually exclusive and the explicit one wins, for the same reason it does in
layer_lora: resuming would overwrite the weights that were just loaded and adopt the old run's
step count and LR schedule along with them.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from transformers import TrainerCallback

from layer_expand.config import DISABLED, LayerExpandConfig
from layer_expand.model import (
    FINAL_DIR_NAME,
    SIDECAR_NAME,
    load_layer_expand_model,
    read_sidecar,
    stamp_config,
)
from tiny_lora.config import (
    DataConfig,
    ModelConfig,
    SFTTrainingConfig,
    _flatten_data_config,
    _merge_dataclass,
    load_yaml_config,
)
from tiny_lora.model import load_tokenizer
from tiny_lora.train_sft import run_sft_core


class ExpandedCheckpointCallback(TrainerCallback):
    """Make every `checkpoint-N` a self-contained, loadable expanded model.

    The Trainer writes weights and config.json and nothing else, which is not enough for a
    non-uniform stack: the new blocks' shapes live only in the sidecar, so a checkpoint without
    one cannot be rebuilt. Two things happen on each save:

    * the sidecar is copied in, so `--adapter outputs/sft-layer-expand/checkpoint-500` works for
      chat and eval exactly as the final `model/` does;
    * a wrapped stack's config.json is retyped, so a stock `from_pretrained` pointed at the
      checkpoint raises instead of silently reinitialising the new block.

    Done per save rather than once at the end because a run that is interrupted -- or simply
    still going -- is exactly when its checkpoints get loaded by hand.
    """

    def __init__(self, sidecar_path: Path, wrapped: bool):
        self.sidecar_path = sidecar_path
        self.wrapped = wrapped

    def on_save(self, args, state, control, **kwargs) -> None:
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            return
        if self.sidecar_path.is_file():
            shutil.copy2(self.sidecar_path, checkpoint / SIDECAR_NAME)
        stamp_config(checkpoint, self.wrapped)


def _guard_checkpoint_rotation(
    expand_cfg: LayerExpandConfig, training_cfg: SFTTrainingConfig
) -> None:
    """Refuse to start if checkpoint rotation would delete the checkpoint being continued from.

    `save_total_limit` deletes the lowest-numbered `checkpoint-N` directories in `output_dir` as
    new ones are written, and it does not know that one of them is this run's starting point.
    Losing it is unrecoverable, so this runs before the base model loads rather than after the
    first save. ("auto" is exempt: it names `<output_dir>/model`, which rotation never touches.)
    """
    spec = expand_cfg.init_from_checkpoint
    if (
        spec is None
        or spec.strip().lower() in DISABLED | {"auto"}
        or training_cfg.save_total_limit is None
    ):
        return

    output_dir = Path(training_cfg.output_dir).resolve()
    checkpoint = Path(spec).resolve()
    if output_dir not in checkpoint.parents:
        return

    raise ValueError(
        f"{checkpoint} sits inside output_dir ({output_dir}) with "
        f"training.save_total_limit={training_cfg.save_total_limit}, so the trainer would "
        "delete it partway through this run to stay under the limit -- including the weights "
        "this run started from. Set training.save_total_limit to null to keep every "
        "checkpoint, or pass --output-dir to write this run somewhere else."
    )


def _warn_about_gradient_checkpointing(training_cfg: SFTTrainingConfig) -> None:
    """Gradient checkpointing buys almost nothing here; say so rather than let it cost silently.

    With every layer before the new block frozen, those layers store no activations to begin
    with -- their outputs carry `requires_grad=False`, so autograd never records them.
    Checkpointing re-runs their forward pass anyway, paying the recompute for memory that was
    never allocated. The only real saving is the new block's own activations, ~0.6GB at batch 8
    x 1536, against a logits tensor several times that size which checkpointing does not touch.
    """
    if training_cfg.gradient_checkpointing:
        warnings.warn(
            "training.gradient_checkpointing is on. Layer expansion freezes everything before "
            "the new block, so those layers already store no activations -- checkpointing "
            "re-runs their forward pass to save memory that was never allocated. Turn it off "
            "unless you have measured a win; lowering per_device_train_batch_size is the "
            "effective lever here, since the logits dominate peak memory.",
            stacklevel=2,
        )


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)

    model_cfg = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training_cfg = _merge_dataclass(SFTTrainingConfig(), raw.get("training", {}))
    expand_cfg = _merge_dataclass(LayerExpandConfig(), raw.get("layer_expand", {}))

    _guard_checkpoint_rotation(expand_cfg, training_cfg)
    _warn_about_gradient_checkpointing(training_cfg)

    output_dir = Path(training_cfg.output_dir)
    tokenizer = load_tokenizer(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    model, init_checkpoint = load_layer_expand_model(model_cfg, expand_cfg, output_dir)

    sidecar = output_dir / SIDECAR_NAME
    wrapped = bool((read_sidecar(output_dir) or {}).get("wrapped"))

    final_dir = run_sft_core(
        model,
        tokenizer,
        data_cfg,
        training_cfg,
        resume=init_checkpoint is None,
        final_dir_name=FINAL_DIR_NAME,
        extra_callbacks=[ExpandedCheckpointCallback(sidecar, wrapped)],
    )

    # The sidecar has to travel with the weights, not just sit in output_dir: a non-uniform
    # stack cannot be rebuilt from config.json alone, so a `model/` directory copied or pushed
    # to the Hub without it is unloadable.
    if sidecar.is_file():
        shutil.copy2(sidecar, Path(final_dir) / SIDECAR_NAME)
    stamp_config(Path(final_dir), wrapped)
    return final_dir
