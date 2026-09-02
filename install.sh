#!/usr/bin/env bash
#
# Clone the repo, install its dependencies, and kick off an SFT run.
#
#     curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/install.sh | bash
#
# or, from a copy of this file:
#
#     bash install.sh
#
# Environment overrides:
#     REPO_DIR   where to clone to               (default: ./qwen_customized_with_tiny_lora)
#     CONFIG     training config to run          (default: configs/sft_ds_assistant_27B.yaml)
#     SKIP_TRAIN set to 1 to install only        (default: unset -- training runs)

set -euo pipefail

REPO_URL="https://github.com/caglanakpinar/qwen_customized_with_tiny_lora.git"
REPO_DIR="${REPO_DIR:-qwen_customized_with_tiny_lora}"
CONFIG="${CONFIG:-configs/sft_ds_assistant.yaml}"

# Clone only when the target isn't already a checkout, so re-running the script updates in place
# instead of failing on a non-empty directory.
if [ -d "$REPO_DIR/.git" ]; then
  echo "==> $REPO_DIR already cloned; pulling latest"
  git -C "$REPO_DIR" pull --ff-only
else
  echo "==> Cloning $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# Poetry drives every install below. `command -v` keeps a second run from reinstalling it.
if command -v poetry >/dev/null 2>&1; then
  echo "==> poetry already installed: $(poetry --version)"
else
  echo "==> Installing poetry"
  pip install poetry
fi

# -E gdrive pulls in gdown, which data.reader: "gdrive" needs to fetch the dataset zip. It is a
# superset of a plain `poetry install`, so one call covers both.
echo "==> Installing dependencies (with the gdrive extra)"
poetry install -E gdrive

if [ "${SKIP_TRAIN:-}" = "1" ]; then
  echo "==> SKIP_TRAIN=1 set; stopping before training"
  echo "    Run training yourself with:"
  echo "      cd $REPO_DIR && poetry run tiny-lora sft --config $CONFIG --no-quant"
  exit 0
fi

# `data/` is gitignored, so a fresh clone never has it. `tiny-lora sft` would fetch it itself on
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

# --no-quant skips bitsandbytes 4-bit loading, which is Linux/CUDA only.
echo "==> Starting SFT run with $CONFIG"
poetry run tiny-lora sft --config "$CONFIG" --no-quant
