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
#     CONFIG     training config to run          (default: configs/sft_ds_assistant.yaml)
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

# --no-quant skips bitsandbytes 4-bit loading, which is Linux/CUDA only.
echo "==> Starting SFT run with $CONFIG"
poetry run tiny-lora sft --config "$CONFIG" --no-quant
