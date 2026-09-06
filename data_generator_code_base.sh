#!/usr/bin/env bash
#
# Build the Kaggle-grounded code corpus: real open datasets, real schemas, and the source code of
# a solution as the answer. Five topics per problem -- problem framing, feature engineering, model
# selection, hyperparameter tuning, evaluation.
#
#     bash data_generator_code_base.sh                  # check, then write the whole catalogue
#     LIST=1 bash data_generator_code_base.sh           # show the 24 problems and exit
#     CHECK_ONLY=1 bash data_generator_code_base.sh     # parse every code block, write nothing
#     DOWNLOAD=1 bash data_generator_code_base.sh       # also fetch the datasets themselves
#
# Environment overrides:
#     OUT_DIR         where the dataset is written        (default: /Volumes/PS2000W/ds_assistant)
#     FORMATS         comma-separated: sft, grpo          (default: sft)
#     TARGET_MB       stop after roughly this many MB     (default: 0 -- write everything)
#     EVAL_RECORDS    held-out eval record count          (default: 500)
#     SHARD_MB        shard size in MB                    (default: 256)
#     SEED            random seed                         (default: 42)
#     FRESH           1 to rebuild from scratch           (default: unset -- appends)
#     CHECK           0 to skip the code-block check      (default: 1)
#     CHECK_ONLY      1 to check and exit                 (default: unset)
#     LIST            1 to print the catalogue and exit   (default: unset)
#     DOWNLOAD        1 to fetch the Kaggle files too     (default: unset)
#     DOWNLOAD_DIR    where those files land              (default: data/kaggle)
#     DOWNLOAD_KEYS   space-separated slugs to limit to   (default: unset -- all 24, ~40 GB)
#
# The corpus itself needs no Kaggle credentials: the catalogue describes the datasets, it does not
# read them. DOWNLOAD=1 is for when you want to *run* the generated code, and that path needs the
# kaggle CLI, ~/.kaggle/kaggle.json, and each competition's rules accepted on the site.

set -euo pipefail

MODULE="data.synthetic.data_generator_code_base"
OUT_DIR="${OUT_DIR:-/Volumes/PS2000W/ds_assistant}"
FORMATS="${FORMATS:-sft}"
TARGET_MB="${TARGET_MB:-0}"
EVAL_RECORDS="${EVAL_RECORDS:-500}"
SHARD_MB="${SHARD_MB:-256}"
SEED="${SEED:-42}"
CHECK="${CHECK:-1}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-data/kaggle}"

if [ "${LIST:-}" = "1" ]; then
  poetry run python -m "$MODULE" --list
  exit 0
fi

# ---------------------------------------------------------------------------
# Optional: pull the datasets so the generated code has something to run against.
# Failures here are warnings, not errors -- a competition whose rules you have not accepted should
# not stop the other 23 from downloading.
# ---------------------------------------------------------------------------
if [ "${DOWNLOAD:-}" = "1" ]; then
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "==> DOWNLOAD=1 needs the kaggle CLI: pip install kaggle, then put an API token in" >&2
    echo "    ~/.kaggle/kaggle.json (Kaggle -> Settings -> Create New Token)" >&2
    exit 1
  fi
  mkdir -p "$DOWNLOAD_DIR"
  echo "==> Fetching Kaggle files into $DOWNLOAD_DIR (large: the full catalogue is tens of GB)"
  while read -r kind slug _file; do
    if [ -n "${DOWNLOAD_KEYS:-}" ] && [[ " $DOWNLOAD_KEYS " != *" $slug "* ]]; then
      continue
    fi
    target="$DOWNLOAD_DIR/${slug//\//__}"
    if [ -d "$target" ] && [ -n "$(ls -A "$target" 2>/dev/null)" ]; then
      echo "    $slug -- already there, skipping"
      continue
    fi
    mkdir -p "$target"
    echo "    $slug"
    set +e
    if [ "$kind" = "competition" ]; then
      kaggle competitions download -c "$slug" -p "$target" -q
    else
      kaggle datasets download -d "$slug" -p "$target" -q --unzip
    fi
    status=$?
    set -e
    if [ $status -ne 0 ]; then
      echo "    !! $slug failed (rules not accepted, or no access) -- carrying on" >&2
      continue
    fi
    for zip in "$target"/*.zip; do
      [ -e "$zip" ] || continue
      unzip -oq "$zip" -d "$target" && rm -f "$zip"
    done
  done < <(poetry run python -m "$MODULE" --slugs)
fi

# ---------------------------------------------------------------------------
# The corpus. The check runs first by default: every answer in this set is a code block, and a
# template that renders a syntax error renders it a few hundred times.
# ---------------------------------------------------------------------------
if [ "${CHECK_ONLY:-}" = "1" ]; then
  poetry run python -m "$MODULE" --check-only
  exit 0
fi

args=(-m "$MODULE"
  --out-dir "$OUT_DIR"
  --formats "$FORMATS"
  --target-mb "$TARGET_MB"
  --eval-records "$EVAL_RECORDS"
  --shard-mb "$SHARD_MB"
  --seed "$SEED")
[ "$CHECK" = "1" ] && args+=(--check)
[ "${FRESH:-}" = "1" ] && args+=(--fresh)

echo "==> Building the Kaggle code corpus into $OUT_DIR (formats=$FORMATS, target=${TARGET_MB}MB)"
poetry run python "${args[@]}"

echo
echo "==> Point a training config at ${OUT_DIR}/${FORMATS%%,*}_train-*.jsonl, or copy the shards"
echo "    into data/synthetic/dataset/ to train on this and the synthetic set together."
