#!/usr/bin/env bash
#
# Assemble the synthetic training set (data/synthetic/) for TinyLoRA fine-tuning.
#
#     bash data_generate.sh
#
# Environment overrides:
#     OUT_DIR          where the dataset is written                 (default: data/synthetic/dataset)
#     TARGET_GB        bytes to target, in GB                       (default: 5.0)
#     FORMATS          comma-separated: sft, grpo                   (default: sft)
#     READER           local or gdrive                               (default: local)
#     GDRIVE_FILE_ID   Drive file id/URL of the dataset zip          (default: unset -- required for READER=gdrive)
#     FRESH            set to 1 to rebuild from scratch, not append  (default: unset -- appends to an existing set)
#     ZIP              set to 1 to zip OUT_DIR when done             (default: unset -- implied by READER=gdrive)
#     NO_STORES        set to 1 to skip the Chroma/FAISS build       (default: unset)
#     STORES_ONLY      set to 1 to rebuild only Chroma/FAISS         (default: unset -- leaves the JSONL alone)
#     EVAL_RECORDS     held-out eval record count                   (default: 4000)
#     SEED             random seed                                   (default: 42)

set -euo pipefail

TARGET_GB="${TARGET_GB:-5.0}"
FORMATS="${FORMATS:-sft}"
READER="${READER:-local}"

if [ "$READER" = "gdrive" ] && [ -z "${GDRIVE_FILE_ID:-}" ]; then
  echo "==> READER=gdrive requires GDRIVE_FILE_ID to be set" >&2
  exit 1
fi

args=(-m data.synthetic.build --target-gb "$TARGET_GB" --formats "$FORMATS" --reader "$READER")
[ -n "${OUT_DIR:-}" ] && args+=(--out-dir "$OUT_DIR")
[ -n "${GDRIVE_FILE_ID:-}" ] && args+=(--gdrive-file-id "$GDRIVE_FILE_ID")
[ -n "${EVAL_RECORDS:-}" ] && args+=(--eval-records "$EVAL_RECORDS")
[ -n "${SEED:-}" ] && args+=(--seed "$SEED")
[ "${FRESH:-}" = "1" ] && args+=(--fresh)
[ "${ZIP:-}" = "1" ] && args+=(--zip)
[ "${NO_STORES:-}" = "1" ] && args+=(--no-stores)
[ "${STORES_ONLY:-}" = "1" ] && args+=(--stores-only)

echo "==> Building synthetic dataset (target ${TARGET_GB}GB, formats=$FORMATS, reader=$READER)"
poetry run python "${args[@]}"
