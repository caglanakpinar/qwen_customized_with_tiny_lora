"""Supervised fine-tuning of a newly-grown block, plus every block grown before it.

Reuses tiny_lora's SFT loop (`run_sft_core`) and config plumbing, exactly as `layer_expand` does.
What differs from `layer_expand` is which layers end up trainable: `load_layer_grow_model` freezes
only the original base, leaving every block any round has ever added -- including this run's new
one -- open to further training. See `layer_grow/model.py` for why.

Two ways a run picks up earlier weights, mirroring layer_expand's three (there is no
`base_adapters` here -- whatever adapters the first round merged are already baked into
`layer_grow.previous_checkpoint`'s weights):

* `layer_grow.previous_checkpoint` names the finished checkpoint this round grows from. Always
  read; this is what makes a run a *growth* run at all.
* `layer_grow.init_from_checkpoint` starts *this round* from a previous, unfinished-or-finished
  attempt at it. Mutually exclusive with `run_sft_core` resuming an *interrupted* run from
  `<output_dir>/checkpoint-N` by itself -- the explicit setting wins, same reasoning as
  layer_expand's own docstring.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

from transformers import TrainerCallback

from layer_grow.config import LayerGrowConfig
from layer_grow.model import (
    FINAL_DIR_NAME,
    SIDECAR_NAME,
    load_layer_grow_model,
    read_growth_spec,
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


class GrownCheckpointCallback(TrainerCallback):
    """Make every `checkpoint-N` a self-contained, loadable grown model -- same reasoning as
    layer_expand's `ExpandedCheckpointCallback`: the Trainer writes weights and config.json and
    nothing else, which is not enough to rebuild a multi-round, non-uniform stack."""

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


def _guard_checkpoint_rotation(grow_cfg: LayerGrowConfig, training_cfg: SFTTrainingConfig) -> None:
    """Refuse to start if checkpoint rotation would delete the checkpoint being continued from.

    Same guard as layer_expand's own: `save_total_limit` does not know that
    `init_from_checkpoint` names one of the checkpoints it might rotate away.
    """
    spec = grow_cfg.init_from_checkpoint
    if spec is None or spec.strip().lower() in ("none", "off", "") or spec.strip().lower() == "auto":
        return
    if training_cfg.save_total_limit is None:
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
    """Same non-savings layer_expand warns about: every layer before the trainable tail is
    frozen, so those layers store no activations to begin with -- checkpointing pays the
    recompute for memory that was never allocated."""
    if training_cfg.gradient_checkpointing:
        warnings.warn(
            "training.gradient_checkpointing is on. Layer growth freezes only the base, so "
            "those layers already store no activations -- checkpointing re-runs their forward "
            "pass to save memory that was never allocated. Turn it off unless you have measured "
            "a win; lowering per_device_train_batch_size is the effective lever here.",
            stacklevel=2,
        )


def run_sft_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)

    model_cfg = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training_cfg = _merge_dataclass(
        SFTTrainingConfig(), _flatten_data_config(raw.get("training", {}))
    )
    # _flatten_data_config also unwraps a `layer_grow.gdrive.*` block into gdrive_zip_file_id/
    # gdrive_cache_dir fields, the same way it does for data.gdrive/training.gdrive -- see
    # LayerGrowConfig's own fields for what those drive.
    grow_cfg = _merge_dataclass(LayerGrowConfig(), _flatten_data_config(raw.get("layer_grow", {})))

    _guard_checkpoint_rotation(grow_cfg, training_cfg)
    _warn_about_gradient_checkpointing(training_cfg)

    output_dir = Path(training_cfg.output_dir)
    tokenizer = load_tokenizer(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    model, init_checkpoint = load_layer_grow_model(model_cfg, grow_cfg, output_dir)

    sidecar = output_dir / SIDECAR_NAME
    wrapped = bool((read_growth_spec(output_dir) or {}).get("rounds", [{}])[-1].get("wrapped"))

    final_dir = run_sft_core(
        model,
        tokenizer,
        data_cfg,
        training_cfg,
        resume=init_checkpoint is None,
        final_dir_name=FINAL_DIR_NAME,
        extra_callbacks=[GrownCheckpointCallback(sidecar, wrapped)],
    )

    if sidecar.is_file():
        shutil.copy2(sidecar, Path(final_dir) / SIDECAR_NAME)
    stamp_config(Path(final_dir), wrapped)
    return final_dir
