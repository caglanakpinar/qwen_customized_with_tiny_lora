#!/usr/bin/env bash
#
# Serve the browser chat UI (web/) against a trained TinyLoRA adapter.
#
#     bash web.sh
#
# Environment overrides:
#     ADAPTER    adapter or checkpoint dir to load   (default: outputs/sft-ds-assistant/adapter,
#                                                       falling back to the newest checkpoint-N/
#                                                       under the same dir if the adapter isn't
#                                                       written yet)
#     MODEL      base model override                 (default: unset -- read from the adapter)
#     DB_PATH    knowledge base dir for RAG           (default: data/data_science_dbs)
#     NO_QUANT   set to 1 to disable 4-bit quant      (default: 1 -- required on macOS)
#     SYSTEM     system prompt to prepend             (default: unset)
#     HOST       host interface to bind to            (default: 127.0.0.1)
#     PORT       port to serve the chat UI on         (default: 8000)

set -euo pipefail

# --adapter is outputs/sft-ds-assistant/adapter/ (written once SFT completes) or, while a run is
# still in progress or was interrupted early, the newest checkpoint-N/ under the same dir.
DEFAULT_ADAPTER="outputs/sft-ds-assistant/adapter"
if [ ! -d "$DEFAULT_ADAPTER" ]; then
  DEFAULT_ADAPTER="$(ls -dt outputs/sft-ds-assistant/checkpoint-*/ 2>/dev/null | head -1)"
fi

ADAPTER="${ADAPTER:-$DEFAULT_ADAPTER}"
DB_PATH="${DB_PATH:-data/data_science_dbs}"
NO_QUANT="${NO_QUANT:-1}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

if [ -z "$ADAPTER" ]; then
  echo "==> No adapter found under outputs/sft-ds-assistant/; set ADAPTER=<dir> and re-run" >&2
  exit 1
fi

args=(serve --adapter "$ADAPTER" --host "$HOST" --port "$PORT")
[ -n "${MODEL:-}" ] && args+=(--model "$MODEL")
[ "$NO_QUANT" = "1" ] && args+=(--no-quant)
[ -n "${SYSTEM:-}" ] && args+=(--system "$SYSTEM")
[ -d "$DB_PATH" ] && args+=(--db-path "$DB_PATH")

echo "==> Serving chat UI on http://$HOST:$PORT with adapter $ADAPTER"
poetry run tiny-lora "${args[@]}"
