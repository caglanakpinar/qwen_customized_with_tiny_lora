"""GRPO reinforcement learning for a newly-grown block, plus every block grown before it.

Reuses tiny_lora's GRPO loop (`run_grpo_core`) and config plumbing, exactly as `layer_grow.
train_sft` reuses `run_sft_core`. What differs from plain TinyLoRA GRPO
(`tiny_lora.train_grpo.run_grpo`) is the model: `load_layer_grow_model` builds the grown stack
directly -- previous rounds' blocks loaded, this round's new block spliced in, only the original
base frozen -- and that is handed to `run_grpo_core` as-is, with `peft_config=None`. There is no
PEFT wrapper here, the same way `layer_grow.train_sft` hands `run_sft_core` an already-built
module instead of building one through TinyLoRA.

Which reward functions score completions is picked by `layer_grow.reward_set` (yaml) or
`--reward-set` (CLI, overrides yaml) -- see `layer_grow/rewards.py` for what each name means and
why they aren't merged into one list.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from layer_grow.config import LayerGrowConfig
from layer_grow.model import (
    FINAL_DIR_NAME,
    SIDECAR_NAME,
    load_layer_grow_model,
    read_growth_spec,
    stamp_config,
)
from layer_grow.rewards import DEFAULT_REWARD_SET, SYSTEM_PROMPTS, resolve_reward_funcs
from layer_grow.train_sft import (
    GrownCheckpointCallback,
    _guard_checkpoint_rotation,
    _warn_about_gradient_checkpointing,
)
from tiny_lora.config import (
    DataConfig,
    GRPOTrainingConfig,
    ModelConfig,
    _flatten_data_config,
    _merge_dataclass,
    load_yaml_config,
)
from tiny_lora.model import load_tokenizer
from tiny_lora.train_grpo import run_grpo_core


def run_grpo_from_yaml(config_path: str | Path, overrides: dict | None = None) -> str:
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)

    model_cfg = _merge_dataclass(ModelConfig(), raw.get("model", {}))
    data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get("data", {})))
    training_cfg = _merge_dataclass(
        GRPOTrainingConfig(), _flatten_data_config(raw.get("training", {}))
    )
    # `layer_grow.gdrive.*` unwraps into gdrive_zip_file_id/gdrive_cache_dir the same way
    # data.gdrive/training.gdrive do elsewhere; `reward_set` isn't a LayerGrowConfig field, so
    # `_merge_dataclass` silently drops it here and it's read straight off `raw` below instead --
    # the same reason CLI overrides write it into the `layer_grow` dict alongside `layers`/
    # `hidden_size` rather than needing separate override plumbing.
    grow_cfg = _merge_dataclass(LayerGrowConfig(), _flatten_data_config(raw.get("layer_grow", {})))
    reward_set = raw.get("layer_grow", {}).get("reward_set", DEFAULT_REWARD_SET)
    reward_funcs = resolve_reward_funcs(reward_set)
    # The system prompt follows the reward set, so `--reward-set` switches both together; an
    # explicit `data.system_prompt` (null included, to turn it off) wins over the set's default.
    if "system_prompt" not in raw.get("data", {}):
        data_cfg.system_prompt = SYSTEM_PROMPTS[reward_set]

    _guard_checkpoint_rotation(grow_cfg, training_cfg)
    _warn_about_gradient_checkpointing(training_cfg)

    output_dir = Path(training_cfg.output_dir)
    tokenizer = load_tokenizer(
        model_cfg.model_name_or_path, trust_remote_code=model_cfg.trust_remote_code
    )
    model, init_checkpoint = load_layer_grow_model(model_cfg, grow_cfg, output_dir)

    sidecar = output_dir / SIDECAR_NAME
    wrapped = bool((read_growth_spec(output_dir) or {}).get("rounds", [{}])[-1].get("wrapped"))

    final_dir = run_grpo_core(
        model,
        tokenizer,
        data_cfg,
        training_cfg,
        reward_funcs=reward_funcs,
        resume=init_checkpoint is None,
        final_dir_name=FINAL_DIR_NAME,
        extra_callbacks=[GrownCheckpointCallback(sidecar, wrapped)],
    )

    if sidecar.is_file():
        shutil.copy2(sidecar, Path(final_dir) / SIDECAR_NAME)
    stamp_config(Path(final_dir), wrapped)
    return final_dir
