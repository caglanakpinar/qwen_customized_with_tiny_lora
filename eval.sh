#!/usr/bin/env bash
#
# Compare base-model vs checkpoint eval loss/perplexity on the configured eval split.
#
#     bash eval.sh                      # newest checkpoint (or the final adapter) of the run
#     CHECKPOINT=5000 bash eval.sh      # evaluate outputs/<run>/checkpoint-5000 specifically
#
# Environment overrides:
#     CONFIG            training config supplying the eval split   (default: configs/sft_ds_assistant.yaml)
#     CHECKPOINT        checkpoint number to evaluate               (default: unset -- use the final
#                                                                     adapter/ if present, else the newest
#                                                                     checkpoint-N/ under the run's
#                                                                     output_dir; when set it wins over
#                                                                     both)
#     ADAPTER           adapter or checkpoint dir to evaluate       (default: unset -- resolved from
#                                                                     CHECKPOINT as above; when set it
#                                                                     wins over CHECKPOINT)
#     OUTPUT_DIR        run dir holding adapter/ and checkpoint-N/  (default: training.output_dir read
#                                                                     out of $CONFIG)
#     MAX_EVAL_SAMPLES  cap the eval split to this many rows        (default: unset -- use the config's cap)
#     GENERATION_SAMPLES  eval examples to also score by generating (default: unset -- the CLI's 30;
#                                                                     0 skips generation metrics)
#     MAX_NEW_TOKENS    tokens per generated reply                  (default: unset -- the CLI's 256)
#     EVALS_DIR         where to write the JSON results             (default: unset -- an evals/ folder inside --adapter)
#     NO_SAVE           set to 1 to print without writing to disk   (default: unset)
#     SKIP_INSTALL      set to 1 to skip the poetry install step    (default: unset -- deps are installed)

set -euo pipefail

CONFIG="${CONFIG:-configs/sft_ds_assistant.yaml}"

if [ ! -f "$CONFIG" ]; then
  echo "==> No such config: $CONFIG" >&2
  exit 1
fi

# The run dir is whatever $CONFIG trained into, so CHECKPOINT resolves against the same run the
# eval split comes from instead of a hard-coded path -- configs/sft_layer_lora.yaml writes to
# outputs/sft-layer-lora, not outputs/sft-ds-assistant.
_yaml_value() {  # _yaml_value <key>: first "key: value" in $CONFIG, quotes and trailing comment stripped
  sed -n "s/^[[:space:]]*$1:[[:space:]]*//p" "$CONFIG" | head -1 |
    sed -e 's/[[:space:]]*#.*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'$/\1/"
}

OUTPUT_DIR="${OUTPUT_DIR:-$(_yaml_value output_dir)}"
if [ -z "$OUTPUT_DIR" ]; then
  echo "==> $CONFIG has no training.output_dir; set OUTPUT_DIR=<dir> or ADAPTER=<dir> and re-run" >&2
  exit 1
fi

EVAL_DATASET="$(_yaml_value eval_dataset_name)"
if [ -z "$EVAL_DATASET" ]; then
  echo "==> $CONFIG has no data.eval_dataset_name -- eval needs a held-out split" >&2
  exit 1
fi

# Resolution order, most explicit first: ADAPTER names a directory outright; CHECKPOINT names one
# by step number and must resolve to that exact checkpoint (it is NOT a fallback -- an explicitly
# requested checkpoint that is missing is an error, never a silent slide onto adapter/); with
# neither set, prefer the final adapter/ written once SFT completes and fall back to the newest
# checkpoint-N/ while a run is still in progress or was interrupted early.
if [ -n "${ADAPTER:-}" ]; then
  RESOLVED_FROM="ADAPTER"
elif [ -n "${CHECKPOINT:-}" ]; then
  ADAPTER="$OUTPUT_DIR/checkpoint-${CHECKPOINT}"
  RESOLVED_FROM="CHECKPOINT=${CHECKPOINT}"
  if [ ! -d "$ADAPTER" ]; then
    echo "==> No checkpoint-${CHECKPOINT} found under $OUTPUT_DIR/" >&2
    echo "==> Available: $(ls -d "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
  fi
elif [ -d "$OUTPUT_DIR/adapter" ]; then
  ADAPTER="$OUTPUT_DIR/adapter"
  RESOLVED_FROM="final adapter"
else
  ADAPTER="$(ls -dt "$OUTPUT_DIR"/checkpoint-*/ 2>/dev/null | head -1)"
  ADAPTER="${ADAPTER%/}"
  RESOLVED_FROM="newest checkpoint"
fi

if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
  echo "==> No adapter or checkpoint found under $OUTPUT_DIR/; set ADAPTER=<dir> and re-run" >&2
  exit 1
fi

# The model is built as base + these weights (`PeftModel.from_pretrained`), and the base model id
# is read out of this file -- so a directory without it cannot be loaded at all. Checking here
# fails in a second rather than after the install and a base-model download.
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
  echo "==> $ADAPTER has no adapter_config.json -- not a saved adapter/checkpoint dir" >&2
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
[ -n "${GENERATION_SAMPLES:-}" ] && args+=(--generation-samples "$GENERATION_SAMPLES")
[ -n "${MAX_NEW_TOKENS:-}" ] && args+=(--max-new-tokens "$MAX_NEW_TOKENS")
[ -n "${EVALS_DIR:-}" ] && args+=(--evals-dir "$EVALS_DIR")
[ "${NO_SAVE:-}" = "1" ] && args+=(--no-save)

# Print what was resolved before the (slow) run starts, so a mistyped CHECKPOINT is visible in the
# first line of output rather than only in the saved JSON afterwards.
echo "==> Config:     $CONFIG"
echo "==> Eval split: $EVAL_DATASET"
echo "==> Checkpoint: $ADAPTER  (from $RESOLVED_FROM)"
echo "==> Evaluating the base model and the model initialized from $ADAPTER"
poetry run tiny-lora "${args[@]}"
