#!/usr/bin/env bash
#
# Fine-tune every extra layer of the published layer_grow model, one layer at a time, on several
# datasets in turn -- a sequential run of `layer_grow sft`, pasting each result back into the
# downloaded model before the next one starts.
#
#     bash layer_grow_multiple_run.sh
#     curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/layer_grow_multiple_run.sh | bash
#
# What it does:
#
#   1. Downloads Caglana/qwen0.5b-tinylora-ds-assistant -- every layer, ~5 GB -- into
#      $WORK_DIR/model, once. A re-run keeps whatever is in there (that is the point: it holds the
#      work done so far), it never downloads over it.
#   2. Reads that model's layers and each layer's dimensions from the model itself (layer_grow.json
#      and the weights' own tensor shapes). Qwen2.5-0.5B has BASE_LAYERS=24 layers and the model has
#      30, so the extra layers are 25-30 (layer_grow.layers counts from 0: indexes 24-29). They are
#      not the same size -- for the published model, 25 is 1536d, 26-28 are 3072d, 29-30 are 6144d.
#   3. Loops, layers outside, datasets inside:
#
#          for layer in 25 26 27 28 29 30:
#              for dataset in ds-assistant-v6 ... ds-assistant-v10:
#                  a. write a training config for this layer x dataset -- see below
#                  b. layer_grow sft --config <it> --no-quant, training ONLY this layer
#                     (layer_grow.train_only_listed_layers; a refine round, so no new block)
#                  c. paste the result back into $WORK_DIR/model, so the next iteration starts
#                     from it; then delete that run's checkpoints. Each paste adds one refine round
#                     to the model's layer_grow.json, the history of what was trained when.
#
#      so each layer is trained on every dataset in order, and by the end all six layers have been
#      trained on all of them. The datasets are fetched and checked up front, before any training.
#
# The config in 3a is configs/sft_layer.yaml with this iteration's values patched in, written to
# $WORK_DIR/configs/layer<index>__<dataset>.yaml: the local model as layer_grow.previous_checkpoint
# (so the Drive fallback for it is dropped), layers: [<index>], that layer's dimensions as read from
# the model, and the dataset's zip id with a cache directory of its own. configs/sft_layer.yaml
# itself is never written to and layer_grow_install.sh is not changed (it is only run once, in its
# install-only mode, below): running that on its own behaves exactly as it always did, and `git pull`
# on a checkout never meets a modified config.
#
# Resuming: finished (layer, dataset) pairs are recorded in $WORK_DIR/progress.txt and skipped, so a
# stopped or crashed run is continued by running the same command again. An iteration that was cut
# off partway resumes from its own last checkpoint. To start over, delete $WORK_DIR.
#
# Before the first run, fill in the zip_file_id of each dataset in DATASETS below -- a bare id or a
# pasted share link both work. A dataset with no id is skipped with a warning; with none at all the
# script stops.
#
# Disk: each dataset is ~10 GB (data_generate_v5.sh rolls 10 GB per folder) and all five are kept, so
# each layer's pass reuses them instead of downloading again; plus the model (~5 GB) and the run in
# progress -- a checkpoint holds the whole model plus the trained layer's optimizer state, roughly
# 8 GB for the 6144d layers, and up to save_total_limit of them are kept (OVERRIDES can lower it).
#
# ds-assistant-v6 ... v10 are data_generate_v5.sh's output, whose records run up to ~6k tokens;
# its header says to set training.max_seq_length >= 4096, and configs/sft_layer.yaml has 1536. The
# script warns when it is below MIN_SEQ_LENGTH. Sequences that long need a much smaller per-device
# batch to fit in memory (the effective batch size stays the same), e.g.:
#     OVERRIDES="training.max_seq_length=4096 training.per_device_train_batch_size=2 training.gradient_accumulation_steps=16" \
#         bash layer_grow_multiple_run.sh
#
# Environment overrides:
#     HF_REPO_ID        the model to fine-tune                          (default: Caglana/qwen0.5b-tinylora-ds-assistant)
#     BASE_LAYERS       layers of the base Qwen2.5-0.5B-Instruct         (default: 24; checked against the model)
#     WORK_DIR          the model, per-run outputs, configs, logs        (default: outputs/layer-grow-multi)
#     DATA_DIR          one cache directory per dataset                  (default: data/layer-grow-multi)
#     BASE_CONFIG       the config every iteration is derived from       (default: configs/sft_layer.yaml)
#     LAYER_NUMBERS     train only these extra layers, 1-based, e.g.     (default: unset -- all of them)
#                       "26 27" (the way layers are counted above)
#     ONLY_DATASETS     train only these datasets, by name               (default: unset -- all with an id)
#     DATASET_LIST      "name|zip_id[|mount_path];name|..." instead of   (default: unset -- the DATASETS array)
#                       editing DATASETS below
#     MAX_STEPS         cap on optimizer steps per iteration             (default: unset -- the config's max_steps)
#     LEARNING_RATE     peak LR                                          (default: unset -- the config's learning_rate)
#     MAX_SAMPLES       cap on training rows read                        (default: unset -- the config's max_samples)
#     OVERRIDES         more config values, space separated:             (default: unset)
#                       "training.max_seq_length=4096 training.save_total_limit=2"
#     SINGLE_LAYER      1 trains only the layer being worked on; 0       (default: 1)
#                       also keeps every earlier grown block trainable,
#                       which is what layer_grow_install.sh does
#     VERIFY_PASTE      1 checks, before replacing the model, that       (default: 1)
#                       nothing but the trained layer changed
#     KEEP_RUNS         1 keeps each run's checkpoints after pasting     (default: 0 -- deleted, they are huge)
#     SKIP_INSTALL      1 skips installing poetry and the dependencies   (default: unset -- layer_grow_install.sh does it)
#     DRY_RUN           1 prints the plan and writes the configs, but    (default: unset)
#                       downloads no weights or datasets and trains nothing
#     REPO_DIR          where to clone to                                (default: ./qwen_customized_with_tiny_lora,
#                                                                          skipped when run from inside a checkout)

set -euo pipefail

REPO_URL="https://github.com/caglanakpinar/qwen_customized_with_tiny_lora.git"
REPO_DIR="${REPO_DIR:-qwen_customized_with_tiny_lora}"

# ---------------------------------------------------------------------------
# What to train: the model, its base layer count, and the datasets.
# ---------------------------------------------------------------------------
HF_REPO_ID="${HF_REPO_ID:-Caglana/qwen0.5b-tinylora-ds-assistant}"
BASE_LAYERS="${BASE_LAYERS:-24}"

# One dataset per line, in training order:   name | Google Drive zip file id | mount path (optional)
# Paste the zip file id after the first "|". The optional third field is the same zip's path
# relative to "My Drive" -- on Colab, with Drive mounted, it is read from there instead of through
# gdown, which Google throttles hard for files this big ("Too many users have viewed ...").
DATASETS=(
  "ds-assistant-v6|"
  "ds-assistant-v7|"
  "ds-assistant-v8|"
  "ds-assistant-v9|"
  "ds-assistant-v10|"
)
# Below this training.max_seq_length the settings printout warns -- see the header.
MIN_SEQ_LENGTH=4096

WORK_DIR="${WORK_DIR:-outputs/layer-grow-multi}"
DATA_DIR="${DATA_DIR:-data/layer-grow-multi}"
BASE_CONFIG="${BASE_CONFIG:-configs/sft_layer.yaml}"
SINGLE_LAYER="${SINGLE_LAYER:-1}"
VERIFY_PASTE="${VERIFY_PASTE:-1}"
KEEP_RUNS="${KEEP_RUNS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Run from the checkout this file lives in, wherever it was started from -- otherwise the install
# script below would not recognise the checkout and clone a second copy next to it. Piped through
# `curl | bash` there is no file, and the clone branch below takes over.
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
  cd "$(dirname "${BASH_SOURCE[0]}")"
fi

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

if [ ! -f "$BASE_CONFIG" ]; then
  echo "==> No such config: $BASE_CONFIG (looked in $(pwd))" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# The datasets: those with an id, in order. Checked first -- before anything is installed or
# downloaded -- because a missing id is the likeliest mistake, and the cheapest to report.
# ---------------------------------------------------------------------------
trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

if [ -n "${DATASET_LIST:-}" ]; then
  entries="$(printf '%s\n' "$DATASET_LIST" | tr ';' '\n')"
else
  entries="$(printf '%s\n' "${DATASETS[@]}")"
fi

NAMES=()
IDS=()
MOUNTS=()
WITHOUT_ID=""
while IFS= read -r entry; do
  [ -n "$(trim "$entry")" ] || continue
  ds_name="" ds_id="" ds_mount=""
  IFS='|' read -r ds_name ds_id ds_mount <<<"$entry"
  ds_name="$(trim "$ds_name")"
  ds_id="$(trim "$ds_id")"
  ds_mount="$(trim "$ds_mount")"
  if [ -n "${ONLY_DATASETS:-}" ] && ! printf ' %s ' "${ONLY_DATASETS//,/ }" | grep -qF " $ds_name "; then
    continue
  fi
  if [ -z "$ds_id" ] && [ -z "$ds_mount" ]; then
    WITHOUT_ID="$WITHOUT_ID $ds_name"
    continue
  fi
  NAMES+=("$ds_name")
  IDS+=("$ds_id")
  MOUNTS+=("$ds_mount")
done <<<"$entries"

if [ -n "$WITHOUT_ID" ]; then
  echo "==> WARNING: no zip file id for:$WITHOUT_ID -- skipped. Fill them in under DATASETS in layer_grow_multiple_run.sh." >&2
fi
if [ "${#NAMES[@]}" -eq 0 ]; then
  echo "==> No dataset to train on: none has a zip_file_id. Edit DATASETS in layer_grow_multiple_run.sh" >&2
  echo "    (or pass DATASET_LIST=\"ds-assistant-v6|<zip id>;ds-assistant-v7|<zip id>\")." >&2
  exit 1
fi
echo "==> Datasets (${#NAMES[@]}), in order: ${NAMES[*]}"

# Unbuffered, so `tee` shows training output as it happens rather than in chunks.
export PYTHONUNBUFFERED=1

if [ "$DRY_RUN" = "1" ]; then
  echo "==> DRY_RUN=1: planning only -- no install, no weights or datasets downloaded, nothing trained"
elif [ "${SKIP_INSTALL:-}" = "1" ]; then
  echo "==> SKIP_INSTALL=1 set; assuming poetry and the dependencies are already installed"
else
  # layer_grow_install.sh's own install-only mode (SKIP_TRAIN=1): clone check, poetry, `poetry
  # install -E gdrive`, and the stray-vllm cleanup. Its "stopping before training" lines are expected.
  echo "==> Installing dependencies, via layer_grow_install.sh's install-only mode"
  SKIP_TRAIN=1 MODE=sft CONFIG=configs/sft_layer.yaml bash layer_grow_install.sh
fi

PY="poetry run python"
HELPER="scripts/layer_grow_multi.py"

# ---------------------------------------------------------------------------
# 1. The model: download it once, then work on the local copy.
# ---------------------------------------------------------------------------
MODEL_DIR="$WORK_DIR/model"
PROGRESS_FILE="$WORK_DIR/progress.txt"
mkdir -p "$WORK_DIR/configs" "$WORK_DIR/logs" "$WORK_DIR/runs"

META_ONLY=""
[ "$DRY_RUN" = "1" ] && META_ONLY="--metadata-only"

echo "==> Model: $HF_REPO_ID"
$PY "$HELPER" download --repo-id "$HF_REPO_ID" --model-dir "$MODEL_DIR" \
  --marker "$WORK_DIR/model.download.json" $META_ONLY

# ---------------------------------------------------------------------------
# 2. The layers: whatever the model has beyond the base, with their dimensions.
# ---------------------------------------------------------------------------
echo "==> Layers"
# The helper prints its table to stderr, and two lines to stdout: the model's layer count, then the
# 0-based indices to train (layer_grow.layers counts from 0; the table and LAYER_NUMBERS from 1).
LAYERS_OUT="$($PY "$HELPER" layers --model-dir "$MODEL_DIR" --base-layers "$BASE_LAYERS" \
  --numbers "${LAYER_NUMBERS:-}" | tail -n 2)"
MODEL_LAYERS="$(printf '%s\n' "$LAYERS_OUT" | sed -n '1p')"
LAYER_INDICES="$(printf '%s\n' "$LAYERS_OUT" | sed -n '2p')"
LAYER_COUNT=0
for _ in $LAYER_INDICES; do LAYER_COUNT=$((LAYER_COUNT + 1)); done
if [ "$LAYER_COUNT" -eq 0 ] || [ -z "$MODEL_LAYERS" ]; then
  echo "==> The helper listed no layers to train (got: $LAYERS_OUT)" >&2
  exit 1
fi

# Settings that apply to every iteration, from the named knobs and OVERRIDES.
OVERRIDE_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then OVERRIDE_ARGS+=(--max-steps "$MAX_STEPS"); fi
if [ -n "${LEARNING_RATE:-}" ]; then OVERRIDE_ARGS+=(--learning-rate "$LEARNING_RATE"); fi
if [ -n "${MAX_SAMPLES:-}" ]; then OVERRIDE_ARGS+=(--max-samples "$MAX_SAMPLES"); fi
set -f  # OVERRIDES is split on spaces; a "*" in a value must not be expanded against the directory
for kv in ${OVERRIDES:-}; do OVERRIDE_ARGS+=(--set "$kv"); done
set +f

$PY "$HELPER" settings --base-config "$BASE_CONFIG" --expect-min-seq-length "$MIN_SEQ_LENGTH" \
  ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}

if [ "$SINGLE_LAYER" = "1" ]; then
  echo "==> Each iteration trains only its own layer (SINGLE_LAYER=0 for layer_grow's every-grown-block mode)"
else
  echo "==> SINGLE_LAYER=0: each iteration trains its layer together with every grown block"
fi

# Fetch and validate every dataset now: a wrong zip id or a zip of the wrong thing should cost
# minutes, not the hours of training that come before the first iteration to use it.
if [ "$DRY_RUN" != "1" ]; then
  echo "==> Fetching and checking the datasets"
  for ((i = 0; i < ${#NAMES[@]}; i++)); do
    # --opt=value, not --opt value: a Drive file id may start with "-", which argparse would take
    # for a flag.
    $PY "$HELPER" prefetch --name "${NAMES[$i]}" --cache-dir "$DATA_DIR/${NAMES[$i]}" \
      --zip-id="${IDS[$i]}" --mount-path="${MOUNTS[$i]}" --base-config "$BASE_CONFIG"
  done
fi

# ---------------------------------------------------------------------------
# 3. The loops.
# ---------------------------------------------------------------------------
TOTAL=$((LAYER_COUNT * ${#NAMES[@]}))
STEP=0
TRAINED=0
RESUMED=0
echo "==> $TOTAL iteration(s): $LAYER_COUNT layer(s) x ${#NAMES[@]} dataset(s)"

for idx in $LAYER_INDICES; do
  for ((i = 0; i < ${#NAMES[@]}; i++)); do
    name="${NAMES[$i]}"
    STEP=$((STEP + 1))
    tag="layer${idx}__${name}"
    key="layer${idx}/${name}"
    cfg="$WORK_DIR/configs/${tag}.yaml"
    run_dir="$WORK_DIR/runs/${tag}"
    log="$WORK_DIR/logs/${tag}.log"

    if [ -f "$PROGRESS_FILE" ] && grep -qxF "$key" "$PROGRESS_FILE"; then
      echo "==> [$STEP/$TOTAL] layer $((idx + 1)) (index $idx) x $name: already done, skipping"
      RESUMED=$((RESUMED + 1))
      continue
    fi

    echo
    echo "==> [$STEP/$TOTAL] layer $((idx + 1)) of $MODEL_LAYERS (index $idx) x $name  ($(date '+%F %T'))"

    # --joint when SINGLE_LAYER=0; the default is this layer only.
    mode_arg="--single-layer"
    [ "$SINGLE_LAYER" = "1" ] || mode_arg="--joint"
    $PY "$HELPER" config --base-config "$BASE_CONFIG" --model-dir "$MODEL_DIR" \
      --base-layers "$BASE_LAYERS" --layer-index "$idx" --dataset "$name" \
      --cache-dir "$DATA_DIR/$name" --zip-id="${IDS[$i]}" --mount-path="${MOUNTS[$i]}" \
      --output-dir "$run_dir" --out "$cfg" "$mode_arg" \
      ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}

    if [ "$DRY_RUN" = "1" ]; then
      echo "    would run: poetry run layer_grow sft --config $cfg --no-quant"
      echo "    then paste $run_dir/model into $MODEL_DIR and record '$key' in $PROGRESS_FILE"
      continue
    fi

    # A finished run left behind (stopped between training and pasting) must not train again --
    # layer_grow writes the sidecar into model/ as the very last step of a run.
    if [ -f "$run_dir/model/layer_grow.json" ]; then
      echo "    found a finished run in $run_dir; pasting it without training again"
    else
      # --no-quant: bitsandbytes 4-bit is Linux/CUDA only, and the grown weights need bf16 anyway
      # (the same reason layer_grow_install.sh passes it). Appended to the log: a resumed iteration
      # continues the same file.
      if ! poetry run layer_grow sft --config "$cfg" --no-quant 2>&1 | tee -a "$log"; then
        echo "==> FAILED: layer $((idx + 1)) x $name. Log: $log" >&2
        echo "    $run_dir is kept; run the same command again to resume from its last checkpoint." >&2
        exit 1
      fi
    fi

    paste_args=(--model-dir "$MODEL_DIR" --run-dir "$run_dir" --layer-index "$idx"
      --progress-file "$PROGRESS_FILE" --progress-key "$key" "$mode_arg")
    [ "$VERIFY_PASTE" = "1" ] || paste_args+=(--no-verify)
    $PY "$HELPER" paste "${paste_args[@]}"
    TRAINED=$((TRAINED + 1))

    case "$run_dir" in
      */runs/layer*)
        if [ "$KEEP_RUNS" = "1" ]; then
          echo "    KEEP_RUNS=1: keeping $run_dir"
        else
          rm -rf "$run_dir"
        fi
        ;;
    esac
  done
done

echo
if [ "$DRY_RUN" = "1" ]; then
  echo "==> DRY_RUN=1: planned $TOTAL iteration(s); configs are in $WORK_DIR/configs. Nothing was trained."
  exit 0
fi
echo "==> Done: $TRAINED iteration(s) trained now, $RESUMED already done from before ($TOTAL planned)."
echo "    Fine-tuned model: $MODEL_DIR   (progress: $PROGRESS_FILE, logs: $WORK_DIR/logs)"
echo "    Nothing was uploaded. To publish it: CHECKPOINT_DIR=$MODEL_DIR bash upload_to_hf.sh"
