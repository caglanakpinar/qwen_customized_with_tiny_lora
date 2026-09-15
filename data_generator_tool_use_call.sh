#!/usr/bin/env bash
#
# Build the tool-use corpus: conversations in which the assistant calls data tools -- list and
# describe the registered datasets, profile a column, run read-only SQL or Python, cross-validate a
# baseline, search the knowledge base -- reads what they return, and answers from it. Eleven shapes:
# single, parallel and chained calls, looking a key up first, recovering from a tool error, refusing
# to invent a dataset, answering without a tool when none is needed, asking for a missing argument,
# declining what no tool can do, and answering a follow-up from the last result.
#
# Plus the edge-case shapes, which are the hard ones: a call that *succeeds* and returns something
# broken. A profile reporting dtype "object" for a column of money is a successful result and a
# wrong answer, and nothing in the payload says "problem". Those shapes are dirty_data (a sentinel,
# a string-typed number, a dtype that will not support the question asked) and misnamed_column (the
# user names a column the dataset does not have, close to one it does).
#
# Written in Qwen2.5's native <tool_call> / <tool_response> format as plain {role, content}
# messages, so it trains on its own or in the same glob as the other two corpora.
#
#     bash data_generator_tool_use_call.sh                  # check, then write the whole corpus
#     LIST=1 bash data_generator_tool_use_call.sh           # show the tools and per-shape counts
#     SHOW=1 bash data_generator_tool_use_call.sh           # print one conversation per shape
#     CHECK_ONLY=1 bash data_generator_tool_use_call.sh     # validate everything, write nothing
#
# Environment overrides:
#     OUT_DIR         where the dataset is written              (default: data/synthetic/dataset_tool_use)
#     FORMATS         comma-separated: sft, grpo                (default: sft)
#     TARGET_MB       stop after roughly this many MB           (default: 0 -- write everything)
#     VARIANTS        draws per conversation template           (default: 4)
#     EVAL_RECORDS    held-out eval record count                (default: 500)
#     SHARD_MB        shard size in MB                          (default: 256)
#     SEED            random seed                               (default: 42)
#     TOKENIZER       tokenizer for the render/length check     (default: Qwen/Qwen2.5-0.5B-Instruct)
#     MAX_TOKENS      longest record the check accepts          (default: 1536 -- training.max_seq_length)
#     FRESH           1 to rebuild from scratch                 (default: unset -- appends)
#     CHECK           0 to skip the check                       (default: 1)
#     CHECK_ONLY      1 to check and exit                       (default: unset)
#     LIST            1 to print the tools and counts and exit  (default: unset)
#     SHOW            N to print N conversations per shape      (default: unset)
#     NO_KB           1 to skip the knowledge-base shapes       (default: unset -- reads the Chroma stores)
#
# VARIANTS is the volume knob, not TARGET_MB. The corpus is finite -- 3,019 conversation templates
# per variant (2,259 with --no-kb), roughly 7.5 MB -- so TARGET_MB only caps a run early; it cannot
# produce more than the catalogue holds. VARIANTS=16 is ~120 MB, VARIANTS=32 ~240 MB, VARIANTS=256
# ~1.9 GB. Past roughly 64, records repeat with nothing new but random values, lead-in phrasing and
# tool orderings, so prefer widening the generator to padding it.
#
# The lead-in bank was widened to 43 entries (data/synthetic/edge_cases.py EDGE_TEXT), which is what
# makes the higher variant counts less degenerate than they used to be: with ten lead-ins, anything
# past variant ten was re-rolling numbers under a phrasing it had already used.
#
# The check validates every call against its tool's JSON schema. When the tokenizer loads, it also
# renders a sample through the real chat template both ways -- the flat messages the trainer sees,
# and structured tool_calls with tools= as a serving client sends them -- and fails on any
# difference. Sequence length is bounded at generation time instead: a record whose rendered text
# exceeds the character budget is dropped and counted, because the trainer truncates from the right
# and would cut off the final answer. The check still measures real token counts and fails if any
# record slips past. On very large runs, CHECK=0 skips the tokenizer pass; the budget still holds.

set -euo pipefail

# ---------------------------------------------------------------------------
# Precondition: the generator package has to be present.
#
# `data/` is gitignored (line 1 of .gitignore), so it lives only in a working tree -- it is in no
# commit and on no remote. It did not survive the repository re-arrange, and `git checkout` cannot
# bring it back. Without this check the run prints "Building..." and then dies several lines later
# on a bare `ModuleNotFoundError: No module named 'data'`, which reads like a broken virtualenv
# rather than a missing directory.
# ---------------------------------------------------------------------------
if [ ! -f "data/synthetic/data_generator_tool_use_call.py" ]; then
  echo "==> missing data/synthetic/data_generator_tool_use_call.py" >&2
  echo "    This script drives the generator package under data/synthetic/, which is not in this" >&2
  echo "    working tree. data/ is gitignored, so it is in no commit and on no remote and cannot" >&2
  echo "    be restored with git -- it has to be copied back from another working tree or" >&2
  echo "    regenerated before any of the data_generator_*.sh scripts will run." >&2
  exit 1
fi

MODULE="data.synthetic.data_generator_tool_use_call"
OUT_DIR="${OUT_DIR:-data/synthetic/dataset_tool_use}"
FORMATS="${FORMATS:-sft}"
TARGET_MB="${TARGET_MB:-0}"
VARIANTS="${VARIANTS:-4}"
EVAL_RECORDS="${EVAL_RECORDS:-500}"
SHARD_MB="${SHARD_MB:-256}"
SEED="${SEED:-42}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_TOKENS="${MAX_TOKENS:-1536}"
CHECK="${CHECK:-1}"

common=(--variants "$VARIANTS" --seed "$SEED")
[ "${NO_KB:-}" = "1" ] && common+=(--no-kb)

if [ "${LIST:-}" = "1" ]; then
  poetry run python -m "$MODULE" --list "${common[@]}"
  exit 0
fi

if [ -n "${SHOW:-}" ]; then
  poetry run python -m "$MODULE" --show "$SHOW" "${common[@]}"
  exit 0
fi

check_args=(--tokenizer "$TOKENIZER" --max-tokens "$MAX_TOKENS")

if [ "${CHECK_ONLY:-}" = "1" ]; then
  poetry run python -m "$MODULE" --check-only "${common[@]}" "${check_args[@]}"
  exit 0
fi

args=(-m "$MODULE"
  --out-dir "$OUT_DIR"
  --formats "$FORMATS"
  --target-mb "$TARGET_MB"
  --eval-records "$EVAL_RECORDS"
  --shard-mb "$SHARD_MB"
  "${common[@]}")
[ "$CHECK" = "1" ] && args+=(--check "${check_args[@]}")
[ "${FRESH:-}" = "1" ] && args+=(--fresh)

echo "==> Building the tool-use corpus into $OUT_DIR (formats=$FORMATS, variants=$VARIANTS, target=${TARGET_MB}MB)"
poetry run python "${args[@]}"

echo
echo "==> Point a training config at ${OUT_DIR}/${FORMATS%%,*}_train-*.jsonl, or copy the shards"
echo "    into data/synthetic/dataset/ to train on this and the synthetic set together."
echo "    At inference, build the prompt with apply_chat_template(messages, tools=TOOLS) and the"
echo "    same system prompt, then parse the <tool_call> blocks the model emits."
