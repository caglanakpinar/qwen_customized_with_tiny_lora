#!/usr/bin/env bash
#
# Push a checkpoint to the Hugging Face Hub as a new version of the adapter repo.
#
#     bash upload_to_hf.sh
#
# Walks through: clearing any stale HF_TOKEN/login, logging in with a token you paste,
# collecting release notes for this version, then uploading the checkpoint + a regenerated
# model card in one commit (optionally tagged).
#
# The card is rebuilt from the checkpoint itself -- method (TinyLoRA vs layer-scoped LoRA),
# trainable parameter count, hyperparameters, and the base-vs-adapter benchmark in
# outputs/eval_results.json -- so pushing a different checkpoint regenerates a matching card.
#
# Preview the card without uploading anything (no token needed):
#
#     DRY_RUN=1 CHECKPOINT_DIR=outputs/sft-layer-lora/checkpoint-3200 bash upload_to_hf.sh
#
# Push the layer-LoRA checkpoint as a tagged new version:
#
#     CHECKPOINT_DIR=outputs/sft-layer-lora/checkpoint-3200 TAG=v3-layer-lora bash upload_to_hf.sh
#
# Environment overrides:
#     REPO_ID          target Hub repo                              (default: Caglana/qwen0.5b-tinylora-ds-assistant)
#     CHECKPOINT_DIR   local checkpoint to push                     (default: outputs/sft-layer-lora/checkpoint-3200)
#     TAG              tag/revision name for this commit            (default: unset -- no tag created)
#     EVAL_RESULTS     benchmark file to read scores from           (default: outputs/eval_results.json)
#     DRY_RUN          1 to print the model card and upload nothing (default: unset)

set -euo pipefail

REPO_ID="${REPO_ID:-Caglana/qwen0.5b-tinylora-ds-assistant}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/sft-layer-lora/checkpoint-3200}"
EVAL_RESULTS="${EVAL_RESULTS:-outputs/eval_results.json}"

# Built as a plain string, not a bash array -- macOS ships bash 3.2, where expanding an empty
# array under `set -u` throws "unbound variable". None of these values contain spaces, so
# word-splitting them back apart on invocation is safe.
EXTRA_ARGS=""
if [ -n "${TAG:-}" ]; then
  EXTRA_ARGS="--tag $TAG"
fi

if [ "${DRY_RUN:-}" = "1" ]; then
  echo "==> DRY RUN: rendering the model card for $CHECKPOINT_DIR, uploading nothing"
  poetry run python scripts/push_to_hub.py \
    --repo-id "$REPO_ID" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --eval-results "$EVAL_RESULTS" \
    --dry-run
  exit 0
fi

echo "==> Clearing any cached/expired Hugging Face credentials"
unset HF_TOKEN || true
poetry run hf auth logout >/dev/null 2>&1 || true

echo
echo "==> Paste a Hugging Face WRITE access token (create one at https://huggingface.co/settings/tokens)"
read -r -s -p "HF token: " HF_TOKEN
echo
# Strip any stray whitespace/newlines a terminal paste can leave on a hidden read.
HF_TOKEN="$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')"
if [ -z "$HF_TOKEN" ]; then
  echo "==> No token entered, aborting." >&2
  exit 1
fi
export HF_TOKEN

echo "==> Verifying token"
poetry run python -c "
from huggingface_hub import HfApi
info = HfApi().whoami()
print('==> Logged in as:', info['name'])
"

echo
echo "==> Paste release notes for this version (what's new / what changed)."
echo "    Press Enter on an empty line when done; leave blank to use the default notes."
NOTES=""
while IFS= read -r line; do
  [ -z "$line" ] && break
  NOTES="${NOTES}${line}"$'\n'
done

if [ -n "$NOTES" ]; then
  echo "==> Uploading $CHECKPOINT_DIR to $REPO_ID with your release notes"
  poetry run python scripts/push_to_hub.py \
    --repo-id "$REPO_ID" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --eval-results "$EVAL_RESULTS" \
    --release-notes "$NOTES" \
    $EXTRA_ARGS
else
  echo "==> Uploading $CHECKPOINT_DIR to $REPO_ID with the default release notes"
  poetry run python scripts/push_to_hub.py \
    --repo-id "$REPO_ID" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --eval-results "$EVAL_RESULTS" \
    $EXTRA_ARGS
fi
