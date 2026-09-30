#!/usr/bin/env bash
#
# Install the repo and grow an already-expanded stack by one more transformer block, training it
# *together with* every block grown before it -- by supervised fine-tuning (the default) or, with
# MODE=grpo, by reinforcement learning instead.
#
#     curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/layer_grow_install.sh | bash
#
# or, from a copy of this file:
#
#     bash layer_grow_install.sh
#     MODE=grpo bash layer_grow_install.sh
#
# The layer_grow counterpart to layer_expand_install.sh: same clone/install/dataset preamble, but
# the run at the end is `layer_grow sft` against configs/sft_layer.yaml (MODE=sft, the default) or
# `layer_grow grpo` against configs/grpo_layer.yaml (MODE=grpo) -- either way starting from a
# *finished* layer_expand/layer_grow checkpoint (PREVIOUS_CHECKPOINT) rather than the stock base
# model, since there is no "grow the stock base" case, unlike layer_expand's base_adapters, which
# can be empty. Unlike layer_expand (which freezes everything except the block it just added),
# layer_grow leaves every block any earlier round added trainable too, alongside the new one --
# only the original base stays frozen. See layer_grow/model.py's own docstring for why.
#
# Because at least one grown block is wider than the base model's, the result loads only through
# layer_grow (which rebuilds it from the layer_grow.json sidecar, falling back to a plain
# layer_expand.json if this checkpoint has been through only one round so far) -- not through a
# stock AutoModelForCausalLM, and not via GGUF export. `tiny-lora chat|serve|chat-api|eval
# --adapter`, and chat.sh/web.sh/eval.sh via ADAPTER, all detect the sidecar and handle it
# automatically.
#
# The new block's shape defaults to null on all four dimensions, which layer_grow reads as "same
# as whatever PREVIOUS_CHECKPOINT's own last block already is" -- i.e. keep widening at the same
# rate rather than changing tack. Override any of HIDDEN_SIZE/INTERMEDIATE_SIZE/NUM_HEADS/
# NUM_KV_HEADS to build a differently-shaped block for this round instead, e.g.:
#     HIDDEN_SIZE=4608 NUM_HEADS=72 INTERMEDIATE_SIZE=24960 bash layer_grow_install.sh
#
# Environment overrides:
#     MODE                  sft | grpo -- which subcommand and       (default: sft)
#                           default config (configs/{MODE}_layer.yaml) to run
#     REWARD_SET            math | diagnosis -- reward functions     (default: unset -- the config's
#                           to score completions with (MODE=grpo       layer_grow.reward_set, "math")
#                           only; see layer_grow/rewards.py)
#     REPO_DIR             where to clone to                       (default: ./qwen_customized_with_tiny_lora,
#                                                                     skipped entirely when run from inside a checkout)
#     CONFIG                training config to run                  (default: configs/${MODE}_layer.yaml)
#     PREVIOUS_CHECKPOINT   finished layer_expand/layer_grow        (default: unset -- the config's
#                           checkpoint (or its run dir) to grow       previous_checkpoint)
#                           from -- required one way or another
#     LAYERS                position(s) for the new block(s), e.g.  (default: unset -- the config's layers,
#                           26, or 26-27 for two                      or auto-appended if that is null too)
#     HIDDEN_SIZE           width of this round's new block          (default: unset -- inherits the
#                                                                       previous round's own width)
#     INTERMEDIATE_SIZE     MLP width of this round's new block      (default: unset -- inherits likewise)
#     NUM_HEADS             query heads; hidden_size must be         (default: unset -- inherits likewise)
#                           heads*64
#     NUM_KV_HEADS          KV heads in this round's new block       (default: unset -- inherits likewise)
#     INIT                  identity | random                       (default: unset -- the config's "identity")
#     INIT_CHECKPOINT       continue *this round* from a finished    (default: unset -- the config's "auto")
#                           attempt at it: a path, "auto", or "none"
#     OUTPUT_DIR            where checkpoints are written            (default: unset -- the config's output_dir)
#     MAX_STEPS             cap on optimizer steps                   (default: unset -- the config's max_steps)
#     LEARNING_RATE         peak LR                                  (default: unset -- the config's learning_rate)
#     MAX_SAMPLES           cap on training rows read                (default: unset -- the config's max_samples)
#     SKIP_TRAIN            set to 1 to install only                 (default: unset -- training runs)
#
# Note on INIT_CHECKPOINT: continuing from a checkpoint that lives *inside* OUTPUT_DIR is
# supported, but the config's save_total_limit must be null first -- otherwise checkpoint
# rotation would delete the weights the run started from, and layer_grow refuses to start.
# "auto" is exempt: it names <output_dir>/model, which rotation never touches. This is separate
# from PREVIOUS_CHECKPOINT, which names the *earlier round* to grow from and is always read.

set -euo pipefail

REPO_URL="https://github.com/caglanakpinar/qwen_customized_with_tiny_lora.git"
REPO_DIR="${REPO_DIR:-qwen_customized_with_tiny_lora}"
MODE="${MODE:-sft}"
if [ "$MODE" != "sft" ] && [ "$MODE" != "grpo" ]; then
  echo "==> Unknown MODE: $MODE (expected 'sft' or 'grpo')" >&2
  exit 1
fi
CONFIG="${CONFIG:-configs/${MODE}_layer.yaml}"

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

# -E gdrive pulls in gdown, which data.reader: "gdrive" (and layer_grow.gdrive.zip_file_id, if
# set) needs to fetch a dataset or checkpoint zip. It is a superset of a plain `poetry install`,
# so one call covers both. The layer_grow package is declared in pyproject's packages list, so
# this also puts the `layer_grow` entry point on PATH.
echo "==> Installing dependencies (with the gdrive extra)"
poetry install -E gdrive

# ---------------------------------------------------------------------------
# Clear a stray "vllm": trl's GRPOTrainer module only does `from vllm import LLM,
# SamplingParams` when transformers' is_vllm_available() (importlib.util.find_spec('vllm')
# is not None) says yes -- our own GRPOConfig never sets use_vllm, so we never want vllm
# and never installed it via pyproject.toml. Seen on a Colab box: find_spec('vllm')
# resolves to *something* (no vllm anywhere in this run's own pip install log, so it is
# either a leftover from an earlier session on that box, or a stray empty namespace-package
# directory some other install left on sys.path) even though `import vllm` itself then
# raises "No module named 'vllm'" -- which crashes `from trl import GRPOTrainer` before
# training ever starts, regardless of MODE. Runs every time, not just after a failure: it is
# cheap, a no-op when nothing stray is present, and this is the only point before the
# training command where it is safe to remove a directory that might be sitting under this
# venv's site-packages.
# ---------------------------------------------------------------------------
echo "==> Checking for a stray vllm (trl's GRPOTrainer import guard only wants it absent)"
poetry run python3 -c "
import importlib.util
import shutil
import subprocess
import sys


def check():
    return importlib.util.find_spec('vllm')


spec = check()
if spec is None:
    print('    no vllm spec found -- nothing to clear')
else:
    print(f'    find_spec(\"vllm\"): origin={spec.origin} locations={spec.submodule_search_locations}')
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'vllm'], capture_output=True)
    spec = check()
    if spec is not None:
        # Survived a pip uninstall -- not a package pip tracks via dist-info/RECORD, so this
        # is the stray-namespace-directory case: remove the directory(ies) directly.
        for loc in spec.submodule_search_locations or []:
            print(f'    removing stray namespace dir: {loc}')
            shutil.rmtree(loc, ignore_errors=True)
        spec = check()
    print(f'    vllm spec after cleanup: {spec}')
"

# Assemble the training command now so SKIP_TRAIN can print exactly what it skipped.
args=("$MODE" --config "$CONFIG")
[ -n "${PREVIOUS_CHECKPOINT:-}" ] && args+=(--previous-checkpoint "$PREVIOUS_CHECKPOINT")
[ -n "${LAYERS:-}" ] && args+=(--layers "$LAYERS")
[ -n "${HIDDEN_SIZE:-}" ] && args+=(--hidden-size "$HIDDEN_SIZE")
[ -n "${INTERMEDIATE_SIZE:-}" ] && args+=(--intermediate-size "$INTERMEDIATE_SIZE")
[ -n "${NUM_HEADS:-}" ] && args+=(--num-attention-heads "$NUM_HEADS")
[ -n "${NUM_KV_HEADS:-}" ] && args+=(--num-key-value-heads "$NUM_KV_HEADS")
[ -n "${INIT:-}" ] && args+=(--init "$INIT")
[ -n "${INIT_CHECKPOINT:-}" ] && args+=(--init-from-checkpoint "$INIT_CHECKPOINT")
# --reward-set only exists on the grpo subcommand; guarded by MODE so an sft run never sees an
# option it doesn't have.
[ -n "${REWARD_SET:-}" ] && [ "$MODE" = "grpo" ] && args+=(--reward-set "$REWARD_SET")
[ -n "${OUTPUT_DIR:-}" ] && args+=(--output-dir "$OUTPUT_DIR")
[ -n "${MAX_STEPS:-}" ] && args+=(--max-steps "$MAX_STEPS")
[ -n "${LEARNING_RATE:-}" ] && args+=(--learning-rate "$LEARNING_RATE")
[ -n "${MAX_SAMPLES:-}" ] && args+=(--max-samples "$MAX_SAMPLES")
# --no-quant skips bitsandbytes 4-bit loading, which is Linux/CUDA only. It also keeps the
# previous round's weights in bf16, which loading them onto a freshly built stack needs.
args+=(--no-quant)

if [ "${SKIP_TRAIN:-}" = "1" ]; then
  echo "==> SKIP_TRAIN=1 set; stopping before training"
  echo "    Run training yourself with:"
  echo "      cd $(pwd) && poetry run layer_grow ${args[*]}"
  exit 0
fi

# `data/` is gitignored, so a fresh clone never has it. `layer_grow sft` would fetch it itself on
# a "gdrive" config -- load_raw_dataset calls the same ensure_gdrive_dataset() below -- but doing
# it here first means a bad zip_file_id or a missing dataset fails in seconds, before the model
# and tokenizer have loaded, rather than minutes into the training step. This does not fetch
# PREVIOUS_CHECKPOINT itself if it needs layer_grow.gdrive -- that download is resolved lazily,
# inside `layer_grow sft`, once it actually needs the checkpoint.
echo "==> Preparing dataset for $CONFIG"
poetry run python -c "
import sys
from pathlib import Path

from tiny_lora.config import DataConfig, _flatten_data_config, _merge_dataclass, load_yaml_config
from tiny_lora.data import ensure_gdrive_dataset

raw = load_yaml_config('$CONFIG')
data_cfg = _merge_dataclass(DataConfig(), _flatten_data_config(raw.get('data', {})))
if data_cfg.reader != 'gdrive':
    print(f'    reader is {data_cfg.reader!r}; nothing to fetch')
    sys.exit(0)

cache_dir = ensure_gdrive_dataset(
    data_cfg.gdrive_cache_dir, data_cfg.gdrive_zip_file_id, data_cfg.gdrive_mount_path
)
# The pattern's own basename, not a hardcoded 'sft_train-*.jsonl' -- MODE=grpo's dataset_name is
# 'grpo_math_train-*.jsonl', 'grpo_diagnosis_train-*.jsonl', or a literal filename, none of which
# that SFT-specific glob would ever match.
pattern = Path(data_cfg.dataset_name).name
shards = sorted(cache_dir.glob(pattern))
if not shards:
    sys.exit(
        f'no files matching {pattern!r} in {cache_dir} after extraction -- check '
        f'data.gdrive.zip_file_id/mount_path in $CONFIG points at a dataset zip, not something else'
    )
print(f'    {len(shards)} shard(s) ready in {cache_dir}')
"

echo "==> Growing the stack with $CONFIG"
if [ "$MODE" = "grpo" ]; then
  echo "    the original base stays frozen; every block any round has grown -- old and new -- is reinforcement-learned"
else
  echo "    the original base stays frozen; every block any round has grown -- old and new -- trains"
fi
poetry run layer_grow "${args[@]}"
