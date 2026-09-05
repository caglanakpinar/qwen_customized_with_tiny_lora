#!/usr/bin/env bash
#
# Compare base-model vs checkpoint eval loss/perplexity on the configured eval split.
#
#     bash eval.sh
#
# Environment overrides:
#     CONFIG            training config supplying the eval split   (default: configs/sft_ds_assistant.yaml)
#     CHECKPOINT        checkpoint number to evaluate               (default: unset -- use the newest
#                                                                     checkpoint-N/ under outputs/sft-ds-assistant/;
#                                                                     ignored if ADAPTER is set)
#     ADAPTER           adapter or checkpoint dir to evaluate       (default: outputs/sft-ds-assistant/adapter,
#                                                                     falling back to the checkpoint picked via
#                                                                     CHECKPOINT, or the newest checkpoint-N/
#                                                                     under the same dir if the adapter isn't
#                                                                     written yet)
#     MAX_EVAL_SAMPLES  cap the eval split to this many rows        (default: unset -- use the config's cap)
#     EVALS_DIR         where to write the JSON results             (default: unset -- an evals/ folder inside --adapter)
#     NO_SAVE           set to 1 to print without writing to disk   (default: unset)
#     SKIP_INSTALL      set to 1 to skip the poetry install step    (default: unset -- deps are installed)

set -euo pipefail

CONFIG="${CONFIG:-configs/sft_ds_assistant.yaml}"

# --adapter is outputs/sft-ds-assistant/adapter/ (written once SFT completes) or, while a run is
# still in progress or was interrupted early, a checkpoint-N/ under the same dir -- CHECKPOINT
# picks a specific one, falling back to the newest if unset.
DEFAULT_ADAPTER="outputs/sft-ds-assistant/adapter"
if [ ! -d "$DEFAULT_ADAPTER" ]; then
  if [ -n "${CHECKPOINT:-}" ]; then
    DEFAULT_ADAPTER="outputs/sft-ds-assistant/checkpoint-${CHECKPOINT}"
    if [ ! -d "$DEFAULT_ADAPTER" ]; then
      echo "==> No checkpoint-${CHECKPOINT} found under outputs/sft-ds-assistant/" >&2
      exit 1
    fi
  else
    DEFAULT_ADAPTER="$(ls -dt outputs/sft-ds-assistant/checkpoint-*/ 2>/dev/null | head -1)"
  fi
fi

ADAPTER="${ADAPTER:-$DEFAULT_ADAPTER}"

if [ -z "$ADAPTER" ]; then
  echo "==> No adapter found under outputs/sft-ds-assistant/; set ADAPTER=<dir> and re-run" >&2
  exit 1
fi

# Dependencies come after the adapter check on purpose: a missing adapter should fail in a
# second, not after the install. Mirrors install.sh from here -- `command -v` keeps a second run
# from reinstalling poetry, and -E gdrive pulls in gdown, which the configs' data.reader:
# "gdrive" needs to fetch the eval split.
if [ "${SKIP_INSTALL:-}" = "1" ]; then
  echo "==> SKIP_INSTALL=1 set; using whatever is already installed"
else
  if command -v poetry >/dev/null 2>&1; then
    echo "==> poetry already installed: $(poetry --version)"
  else
    echo "==> Installing poetry"
    pip install poetry
  fi

  echo "==> Installing dependencies (with the gdrive extra)"
  poetry install -E gdrive
fi

args=(eval --config "$CONFIG" --adapter "$ADAPTER")
[ -n "${MAX_EVAL_SAMPLES:-}" ] && args+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
[ -n "${EVALS_DIR:-}" ] && args+=(--evals-dir "$EVALS_DIR")
[ "${NO_SAVE:-}" = "1" ] && args+=(--no-save)

echo "==> Evaluating $ADAPTER against $CONFIG"
poetry run tiny-lora "${args[@]}"
