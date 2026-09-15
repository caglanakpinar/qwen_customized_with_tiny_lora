#!/usr/bin/env bash
#
# Install the repo and fine-tune a newly appended transformer block on top of the latest
# layer_lora run.
#
#     curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/layer_expand_install.sh | bash
#
# or, from a copy of this file:
#
#     bash layer_expand_install.sh
#
# The layer_expand counterpart to layer_lora_install.sh: same clone/install/dataset preamble,
# but the run at the end is `layer_expand sft` against configs/sft_layer_expand.yaml, which
# adds a 1536-wide layer 24 to the 24-layer base and trains that block alone. Unlike the LoRA
# scripts this is not an adapter run -- the new block's 46.2M parameters all train, and the
# result is a whole 25-layer model rather than an adapter to merge later.
#
# Because that block is wider than the base model's, the result loads only through layer_expand
# (which rebuilds it from the layer_expand.json sidecar) -- not through a stock
# AutoModelForCausalLM, and not via GGUF export. `tiny-lora chat|serve|chat-api|eval --adapter`,
# and chat.sh/web.sh/eval.sh via ADAPTER, all detect the sidecar and handle it automatically.
# For a base-width block that stays fully standard, set HIDDEN_SIZE, INTERMEDIATE_SIZE and
# NUM_HEADS to the base model's own 896 / 4864 / 14.
#
# By default the frozen base is the TinyLoRA run merged with the layer_lora run, in that order
# (outputs/sft-ds-assistant then outputs/sft-layer-lora). Merging only the latter would leave
# every layer but 21 at the stock Qwen2.5-0.5B weights. A previous expansion is continued if one
# finished (INIT_CHECKPOINT=auto, read from the config's output_dir).
#
# The block's shape comes from the config, but can be overridden here, e.g.:
#     HIDDEN_SIZE=2048 NUM_HEADS=32 INTERMEDIATE_SIZE=11136 bash layer_expand_install.sh
# Adding base-width blocks instead of widening one keeps the result a stock Qwen2:
#     HIDDEN_SIZE=896 NUM_HEADS=14 INTERMEDIATE_SIZE=4864 LAYERS=24-25 bash layer_expand_install.sh
#
# Environment overrides:
#     REPO_DIR          where to clone to                          (default: ./qwen_customized_with_tiny_lora,
#                                                                    skipped entirely when run from inside a checkout)
#     CONFIG            training config to run                     (default: configs/sft_layer_expand.yaml)
#     LAYERS            positions for the new block(s), e.g. 24    (default: 24)
#                       or 24-25 for two
#     HIDDEN_SIZE       width of the new block                     (default: unset -- the base model's 896)
#     INTERMEDIATE_SIZE MLP width of the new block                 (default: unset -- the base model's 4864)
#     NUM_HEADS         query heads; hidden_size must be heads*64  (default: unset -- the base model's 14)
#     NUM_KV_HEADS      KV heads in the new block                  (default: unset -- the base model's 2)
#     INIT              identity | random                          (default: unset -- the config's "identity")
#     BASE_ADAPTERS     comma-separated adapters merged into the   (default: unset -- the config's chain,
#                       frozen base, in order; "none" for the        outputs/sft-ds-assistant then
#                       stock base model                             outputs/sft-layer-lora)
#     INIT_CHECKPOINT   finished expanded model to continue from:  (default: unset -- the config's "auto")
#                       a path, "auto", or "none"
#     OUTPUT_DIR        where checkpoints are written              (default: unset -- the config's output_dir)
#     MAX_STEPS         cap on optimizer steps                     (default: unset -- the config's max_steps)
#     LEARNING_RATE     peak LR                                    (default: unset -- the config's learning_rate)
#     MAX_SAMPLES       cap on training rows read                  (default: unset -- the config's max_samples)
#     SKIP_TRAIN        set to 1 to install only                   (default: unset -- training runs)
#
# Note on INIT_CHECKPOINT: continuing from a checkpoint that lives *inside* OUTPUT_DIR is
# supported, but the config's save_total_limit must be null first -- otherwise checkpoint
# rotation would delete the weights the run started from, and layer_expand refuses to start.
# "auto" is exempt: it names <output_dir>/model, which rotation never touches.

set -euo pipefail

REPO_URL="https://github.com/caglanakpinar/qwen_customized_with_tiny_lora.git"
REPO_DIR="${REPO_DIR:-qwen_customized_with_tiny_lora}"
CONFIG="${CONFIG:-configs/sft_layer_expand.yaml}"
LAYERS="${LAYERS:-24}"

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
# superset of a plain `poetry install`, so one call covers both. The layer_expand package is
# declared in pyproject's packages list, so this also puts the `layer_expand` entry point on PATH.
echo "==> Installing dependencies (with the gdrive extra)"
poetry install -E gdrive

# Assemble the training command now so SKIP_TRAIN can print exactly what it skipped.
args=(sft --config "$CONFIG" --layers "$LAYERS")
[ -n "${HIDDEN_SIZE:-}" ] && args+=(--hidden-size "$HIDDEN_SIZE")
[ -n "${INTERMEDIATE_SIZE:-}" ] && args+=(--intermediate-size "$INTERMEDIATE_SIZE")
[ -n "${NUM_HEADS:-}" ] && args+=(--num-attention-heads "$NUM_HEADS")
[ -n "${NUM_KV_HEADS:-}" ] && args+=(--num-key-value-heads "$NUM_KV_HEADS")
[ -n "${INIT:-}" ] && args+=(--init "$INIT")
[ -n "${BASE_ADAPTERS:-}" ] && args+=(--base-adapters "$BASE_ADAPTERS")
[ -n "${INIT_CHECKPOINT:-}" ] && args+=(--init-from-checkpoint "$INIT_CHECKPOINT")
[ -n "${OUTPUT_DIR:-}" ] && args+=(--output-dir "$OUTPUT_DIR")
[ -n "${MAX_STEPS:-}" ] && args+=(--max-steps "$MAX_STEPS")
[ -n "${LEARNING_RATE:-}" ] && args+=(--learning-rate "$LEARNING_RATE")
[ -n "${MAX_SAMPLES:-}" ] && args+=(--max-samples "$MAX_SAMPLES")
# --no-quant skips bitsandbytes 4-bit loading, which is Linux/CUDA only. It also keeps the base
# weights in bf16, which merging the layer_lora adapter into them needs.
args+=(--no-quant)

if [ "${SKIP_TRAIN:-}" = "1" ]; then
  echo "==> SKIP_TRAIN=1 set; stopping before training"
  echo "    Run training yourself with:"
  echo "      cd $(pwd) && poetry run layer_expand ${args[*]}"
  exit 0
fi

# `data/` is gitignored, so a fresh clone never has it. `layer_expand sft` would fetch it itself
# on a "gdrive" config -- load_raw_dataset calls the same ensure_gdrive_dataset() below -- but
# doing it here first means a bad zip_file_id or a missing dataset fails in seconds, before the
# model and tokenizer have loaded, rather than minutes into the training step.
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

echo "==> Adding block(s) at position(s) $LAYERS and fine-tuning them with $CONFIG"
echo "    every existing layer stays frozen at the weights it already has"
poetry run layer_expand "${args[@]}"
