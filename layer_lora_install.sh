#!/usr/bin/env bash
#
# Install the repo and fine-tune a single transformer layer with layer-scoped LoRA.
#
#     curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/layer_lora_install.sh | bash
#
# or, from a copy of this file:
#
#     bash layer_lora_install.sh
#
# The layer_lora counterpart to install.sh: same clone/install/dataset preamble, but the run at
# the end is `layer_lora sft` against configs/sft_layer_lora.yaml with layer 23 alone selected.
# Every layer other than 23 keeps its base weights and never receives a gradient.
#
# Environment overrides:
#     REPO_DIR          where to clone to                          (default: ./qwen_customized_with_tiny_lora,
#                                                                    skipped entirely when run from inside a checkout)
#     CONFIG            training config to run                     (default: configs/sft_layer_lora.yaml)
#     LAYERS            layers to fine-tune, e.g. 23 or 20-23      (default: 23)
#     TARGET_MODULES    projections to adapt within those layers   (default: unset -- the config's list)
#     INIT_CHECKPOINT   continue from a saved adapter instead of   (default: unset -- fresh adapter on LAYERS)
#                       starting fresh; a path, or "auto"
#     OUTPUT_DIR        where checkpoints are written              (default: unset -- the config's output_dir)
#     MAX_STEPS         cap on optimizer steps                     (default: unset -- the config's max_steps)
#     LEARNING_RATE     peak LR                                    (default: unset -- the config's learning_rate)
#     MAX_SAMPLES       cap on training rows read                  (default: unset -- the config's max_samples)
#     SKIP_TRAIN        set to 1 to install only                   (default: unset -- training runs)
#
# Note on INIT_CHECKPOINT: continuing from a checkpoint that lives *inside* OUTPUT_DIR is
# supported, but the config's save_total_limit must be null first -- otherwise checkpoint
# rotation would delete the weights the run started from, and layer_lora refuses to start.

set -euo pipefail

REPO_URL="https://github.com/caglanakpinar/qwen_customized_with_tiny_lora.git"
REPO_DIR="${REPO_DIR:-qwen_customized_with_tiny_lora}"
CONFIG="${CONFIG:-configs/sft_layer_lora.yaml}"
LAYERS="${LAYERS:-23}"

# Running this from inside an existing checkout must not clone a second copy underneath it --
# that is how a nested qwen_customized_with_tiny_lora/ ends up shadowing the real one. Detect the
# checkout by its own pyproject and stay put.
if [ -f pyproject.toml ] && grep -q '^name = "tiny-lora"' pyproject.toml; then
  echo "==> Already inside the tiny-lora checkout ($(pwd)); not cloning"
elif [ -d "$REPO_DIR/.git" ]; then
  echo "==> $REPO_DIR already cloned; pulling latest"
  git -C "$REPO_DIR" pull --ff-only
  cd "$REPO_DIR"
else
  echo "==> Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

if [ ! -f "$CONFIG" ]; then
  echo "==> No such config: $CONFIG (looked in $(pwd))" >&2
  exit 1
fi

# Poetry drives every install below. `command -v` keeps a second run from reinstalling it.
if command -v poetry >/dev/null 2>&1; then
  echo "==> poetry already installed: $(poetry --version)"
else
  echo "==> Installing poetry"
  pip install poetry
fi

# -E gdrive pulls in gdown, which data.reader: "gdrive" needs to fetch the dataset zip. It is a
# superset of a plain `poetry install`, so one call covers both. The layer_lora package is
# declared in pyproject's packages list, so this also puts the `layer_lora` entry point on PATH.
echo "==> Installing dependencies (with the gdrive extra)"
poetry install -E gdrive

# Assemble the training command now so SKIP_TRAIN can print exactly what it skipped.
args=(sft --config "$CONFIG" --layers "$LAYERS")
[ -n "${TARGET_MODULES:-}" ] && args+=(--target-modules "$TARGET_MODULES")
[ -n "${INIT_CHECKPOINT:-}" ] && args+=(--init-from-checkpoint "$INIT_CHECKPOINT")
[ -n "${OUTPUT_DIR:-}" ] && args+=(--output-dir "$OUTPUT_DIR")
[ -n "${MAX_STEPS:-}" ] && args+=(--max-steps "$MAX_STEPS")
[ -n "${LEARNING_RATE:-}" ] && args+=(--learning-rate "$LEARNING_RATE")
[ -n "${MAX_SAMPLES:-}" ] && args+=(--max-samples "$MAX_SAMPLES")
# --no-quant skips bitsandbytes 4-bit loading, which is Linux/CUDA only.
args+=(--no-quant)

if [ "${SKIP_TRAIN:-}" = "1" ]; then
  echo "==> SKIP_TRAIN=1 set; stopping before training"
  echo "    Run training yourself with:"
  echo "      cd $(pwd) && poetry run layer_lora ${args[*]}"
  exit 0
fi

# `data/` is gitignored, so a fresh clone never has it. `layer_lora sft` would fetch it itself on
# a "gdrive" config -- load_raw_dataset calls the same ensure_gdrive_dataset() below -- but doing
# it here first means a bad zip_file_id or a missing dataset fails in seconds, before the model
# and tokenizer have loaded, rather than minutes into the training step.
echo "==> Preparing dataset for $CONFIG"
poetry run python -c "
import sys
from tiny_lora.config import DataConfig, _flatten_data_config, _merge_dataclass, load_yaml_config
from tiny_lora.data import ensure_gdrive_dataset

raw = load_yaml_config('$CONFIG')
data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get('data', {})))
if data_cfg.reader != 'gdrive':
    print(f'    reader is {data_cfg.reader!r}; nothing to fetch')
    sys.exit(0)

cache_dir = ensure_gdrive_dataset(data_cfg.gdrive_cache_dir, data_cfg.gdrive_zip_file_id)
shards = sorted(cache_dir.glob('sft_train-*.jsonl'))
if not shards:
    sys.exit(
        f'no sft_train-*.jsonl in {cache_dir} after extraction -- check data.gdrive.zip_file_id '
        f'in $CONFIG points at a dataset zip, not something else'
    )
print(f'    {len(shards)} shard(s) ready in {cache_dir}')
"

echo "==> Fine-tuning layer(s) $LAYERS with $CONFIG"
echo "    every other layer stays at the weights it already has"
poetry run layer_lora "${args[@]}"
