#!/usr/bin/env bash
#
# Build the whole training set: all three corpora, one third of the byte budget each, into one
# directory with one training glob and one held-out eval file.
#
#     bash data_generator_general.sh                       # 21 GB into /Volumes/PS2000W/ds-assistant
#     TARGET_GB=6 bash data_generator_general.sh           # a smaller run, same three-way split
#     FRESH=1 bash data_generator_general.sh               # rebuild from scratch, do not extend
#     STAGES=tool_use bash data_generator_general.sh       # add just that corpus to what is there
#     DRY_RUN=1 bash data_generator_general.sh             # print the plan, build nothing
#
# The three builders run in turn and all write into $OUT_DIR itself:
#
#     $OUT_DIR/sft_train-00000-of-000NN.jsonl ...   every corpus, one shard series
#     $OUT_DIR/sft_eval.jsonl                       every corpus's hold-out, one file
#
# ---------------------------------------------------------------------------
# How three builders share one directory
# ---------------------------------------------------------------------------
#
# They already know how. Each one checks for `sft_train-*.jsonl` in its output directory and, if it
# finds shards and was not given --fresh, *extends* rather than replaces:
#
#   - `ShardWriter` numbers its new shards after the ones already there, then renames the whole
#     series to the `-of-000NN` convention so the old and the new agree on the total. Without that
#     rename both would match the training glob and the trainer would read a doubled dataset.
#   - the hold-out is written with mode="a", so each corpus's eval records are appended to the one
#     `sft_eval.jsonl` rather than overwriting it. That is the union, for free.
#   - each builder first reads the existing shards back to recover their dedup keys, so it emits
#     nothing the previous corpora already wrote -- and, more importantly, does not regenerate a
#     record that is sitting in the eval file and file it as training data.
#
# Two consequences worth knowing about:
#
#   FRESH applies to the first stage only. It means "rebuild the set from scratch", so it has to
#   reach the first builder that runs and must not reach the second -- passing it to all three
#   would have each one delete the previous one's output. The script handles this; it is the whole
#   reason the stages are not just three independent invocations.
#
#   The dedup read-back is not free. Stage 2 reads back stage 1's shards and stage 3 reads back
#   both, so a 21 GB run parses roughly 21 GB of JSONL beyond generating it. It buys cross-corpus
#   deduplication and a guaranteed-clean train/eval boundary. Set NO_CROSS_DEDUP=1 to build each
#   corpus into its own subdirectory instead and skip it -- faster, but then the three sets are
#   only internally deduplicated and there is one eval file per corpus rather than a union.
#
# Environment overrides:
#     OUT_DIR         where everything is written          (default: /Volumes/PS2000W/ds-assistant)
#     TARGET_GB       total bytes to aim for, in GB        (default: 21 -- 7 GB per corpus)
#     SHARES          three comma-separated weights        (default: 1,1,1 -- equal thirds)
#     STAGES          comma-separated subset to build      (default: synthetic,code_base,tool_use)
#     FORMATS         comma-separated: sft, grpo           (default: sft)
#     EVAL_RECORDS    held-out records *per corpus*        (default: 4000 -- so 12000 in the union)
#     SHARD_MB        shard size in MB                     (default: 512)
#     SEED            random seed                          (default: 777)
#     FRESH           1 to rebuild from scratch            (default: unset -- extends)
#     NO_STORES       1 to skip the Chroma/FAISS build     (default: 1 -- set 0 to build them)
#     NO_KB           1 to skip the knowledge-base shapes  (default: unset)
#     CHECK           0 to skip the per-corpus checks      (default: 1)
#     MAX_VARIANTS    cap on the tool-use volume knob      (default: 256 -- see the note below)
#     NO_CROSS_DEDUP  1 for a subdirectory per corpus      (default: unset -- one shared directory)
#     DRY_RUN         1 to print the plan and stop         (default: unset)
#
# ---------------------------------------------------------------------------
# On the three-way split, honestly
# ---------------------------------------------------------------------------
#
# The three corpora do not hold the same amount of material, and an equal split of the bytes is
# not an equal split of the content:
#
#   synthetic   effectively unbounded. 16 domain families x 6 techniques x 24 broken-column
#               variants is 44.8M distinct scenarios, each carrying 122 tasks and 39 phrasings.
#               A 7 GB share uses a rounding error of the catalogue.
#
#   code_base   bounded by 24 real Kaggle datasets, widened by technique seeds and 63 phrasings.
#               Measured at ~420 MB/min and confirmed past 7 GB without plateauing, so it reaches
#               its share -- but the dedup rate climbs as it goes, because the tasks that do not
#               reference the estimator render identically for every technique variant. A large
#               fraction of the tail is phrasing variation over a catalogue already walked.
#               CHECK=1 parses every generated code block first: about 9 minutes, worth it.
#
#   tool_use    the tightest. ~3,019 conversation templates per variant, ~7.5 MB. Reaching 7 GB
#               needs roughly 900 variants, and a variant only re-rolls the offered tools, the
#               lead-in phrasing and the simulated numbers. Past ~64 variants you are buying
#               repetition, not coverage.
#
# So MAX_VARIANTS defaults to 256 and the tool-use stage stops there rather than at its byte share.
# The run prints what each corpus actually contributed. If you want the literal equal thirds
# regardless, set MAX_VARIANTS=2000; if you would rather the budget went where the material is, set
# SHARES=3,2,1. Neither is wrong -- but the script should not quietly pretend 7 GB of tool-use data
# is 7 GB of distinct tool-use data.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Precondition: the generator package has to be present.
#
# Checked here rather than left to the three sub-scripts, so a 21 GB plan is not printed -- and an
# hour of the first stage not spent -- before the run discovers it cannot generate anything.
# `data/` is gitignored (line 1 of .gitignore): it is in no commit and on no remote, so a missing
# one cannot be restored with git and has to be copied back from another working tree.
# ---------------------------------------------------------------------------
if [ ! -d "data/synthetic" ]; then
  echo "==> missing data/synthetic/ -- the generator package this script drives" >&2
  echo "    data/ is gitignored, so it is in no commit and on no remote. git cannot restore it;" >&2
  echo "    copy it back from another working tree, then re-run." >&2
  exit 1
fi

OUT_DIR="${OUT_DIR:-/Volumes/PS2000W/ds-assistant}"
TARGET_GB="${TARGET_GB:-21}"
SHARES="${SHARES:-1,1,1}"
STAGES="${STAGES:-synthetic,code_base,tool_use}"
FORMATS="${FORMATS:-sft}"
EVAL_RECORDS="${EVAL_RECORDS:-4000}"
SHARD_MB="${SHARD_MB:-512}"
SEED="${SEED:-777}"
NO_STORES="${NO_STORES:-1}"
CHECK="${CHECK:-1}"
MAX_VARIANTS="${MAX_VARIANTS:-256}"

PRIMARY="${FORMATS%%,*}"

# Measured on this catalogue: 4 variants of the tool-use corpus wrote 30.1 MB.
BYTES_PER_VARIANT_MB="7.5"

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

read -r SHARE_SYN SHARE_CB SHARE_TU <<<"$(echo "$SHARES" | tr ',' ' ')"
if [ -z "${SHARE_TU:-}" ]; then
  echo "==> SHARES must be three comma-separated numbers, e.g. SHARES=1,1,1" >&2
  exit 1
fi

plan="$(python3 - "$TARGET_GB" "$SHARE_SYN" "$SHARE_CB" "$SHARE_TU" "$MAX_VARIANTS" "$BYTES_PER_VARIANT_MB" <<'PY'
import math
import sys

target_gb, s_syn, s_cb, s_tu, max_variants, mb_per_variant = (
    float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]),
    int(sys.argv[5]), float(sys.argv[6]),
)
total = s_syn + s_cb + s_tu
if total <= 0:
    raise SystemExit("SHARES must sum to more than zero")

syn_gb = target_gb * s_syn / total
cb_mb = target_gb * s_cb / total * 1000
tu_mb = target_gb * s_tu / total * 1000

# The tool-use corpus is driven by --variants, not by a byte target: the byte target only caps a
# run early, it cannot make the generator produce more than the catalogue holds.
wanted = math.ceil(tu_mb / mb_per_variant)
variants = min(wanted, max_variants)
print(f"{syn_gb:.3f} {cb_mb:.0f} {tu_mb:.0f} {variants} {wanted}")
PY
)"
read -r SYN_GB CB_MB TU_MB TU_VARIANTS TU_WANTED <<<"$plan"

wants() {
  [[ ",${STAGES}," == *",$1,"* ]]
}

# Where each stage writes. One shared directory is the point of this script; NO_CROSS_DEDUP=1
# falls back to a subdirectory per corpus, which skips the read-back at the cost of the union.
stage_dir() {
  if [ "${NO_CROSS_DEDUP:-}" = "1" ]; then echo "$OUT_DIR/$1"; else echo "$OUT_DIR"; fi
}

echo "=============================================================================="
echo "  Target      ${TARGET_GB} GB total, shares ${SHARES} (synthetic,code_base,tool_use)"
echo "  Output      ${OUT_DIR}"
if [ "${NO_CROSS_DEDUP:-}" = "1" ]; then
  echo "  Layout      one subdirectory per corpus (no union, no cross-dedup)"
else
  echo "  Layout      one ${PRIMARY}_train-*.jsonl series + one ${PRIMARY}_eval.jsonl"
fi
echo "  Stages      ${STAGES}"
echo "  Formats     ${FORMATS}${FRESH:+   (FRESH: rebuilding from scratch)}"
echo "------------------------------------------------------------------------------"
wants synthetic && printf "  %-12s %10s GB\n" "synthetic" "$SYN_GB"
wants code_base && printf "  %-12s %10s MB\n" "code_base" "$CB_MB"
wants tool_use  && printf "  %-12s %10s MB   --variants %s\n" "tool_use" "$TU_MB" "$TU_VARIANTS"

if wants tool_use && [ "$TU_WANTED" -gt "$TU_VARIANTS" ]; then
  echo
  echo "  !! tool_use would need ${TU_WANTED} variants for a full ${TU_MB} MB share; capped at"
  echo "     MAX_VARIANTS=${MAX_VARIANTS}. Past ~64 variants a record differs from an earlier one"
  echo "     only in its simulated numbers, its lead-in and the order of the offered tools."
  echo "     Raise MAX_VARIANTS to overrule, or use SHARES to send the budget elsewhere."
fi

# FRESH wipes the shared directory, so a partial-STAGES rebuild would take the other corpora with
# it. Worth a loud warning rather than a silent deletion of several hours of work.
if [ -n "${FRESH:-}" ] && [ "${NO_CROSS_DEDUP:-}" != "1" ] && [ "$STAGES" != "synthetic,code_base,tool_use" ]; then
  echo
  echo "  !! FRESH=1 with STAGES=${STAGES}: the first stage deletes everything already in"
  echo "     ${OUT_DIR}, including the corpora this run is not rebuilding. Drop FRESH to extend"
  echo "     what is there instead."
fi
echo "=============================================================================="
echo

if [ "${DRY_RUN:-}" = "1" ]; then
  echo "==> DRY_RUN=1: nothing built."
  exit 0
fi

mkdir -p "$OUT_DIR"

# FRESH reaches the first stage that actually runs, and nothing after it. Every later stage must
# see the shards the earlier ones wrote, or it would delete them and the "union" would end up
# being whichever corpus happened to run last.
#
# Two traps here, both of which produce the same silent failure -- only the last corpus survives,
# because each stage deleted the one before it.
#
#   1. `take_fresh` assigns to FRESH_ARG rather than echoing it. Called as $(take_fresh) inside the
#      env line it would run in a *subshell*, where clearing fresh_pending has no effect on the
#      parent, so every stage would get FRESH=1.
#   2. The user invokes this as `FRESH=1 bash data_generator_general.sh`, which puts FRESH=1 in
#      *this* script's environment -- and `env` passes the whole inherited environment on to the
#      child. So the later stages have to actively `-u FRESH` to remove it; simply not adding it
#      is not enough. And `-u` is an *option*: env stops parsing options at the first
#      NAME=VALUE, so it has to precede every assignment on the line, not follow them.
fresh_pending="${FRESH:-}"
FRESH_ARG=""
take_fresh() {
  if [ -n "$fresh_pending" ]; then
    FRESH_ARG="FRESH=1"
    fresh_pending=""
  else
    FRESH_ARG="-u FRESH"
  fi
}

# Knobs belonging to the sub-scripts that would short-circuit or redirect a stage if they happened
# to be exported in the calling shell. This script sets everything it means to set explicitly, so
# anything else inherited is an accident.
ENV_SCRUB=(-u LIST -u SHOW -u CHECK_ONLY -u STORES_ONLY -u DOWNLOAD -u ZIP -u TARGET_MB -u TARGET_GB -u VARIANTS)

started_at=$(date +%s)

# ---------------------------------------------------------------------------
# 1. The domain-scenario corpus
# ---------------------------------------------------------------------------
if wants synthetic; then
  echo "=============================================================================="
  echo "==> synthetic  ->  $(stage_dir synthetic)   (${SYN_GB} GB)"
  echo "=============================================================================="
  take_fresh
  env "${ENV_SCRUB[@]}" $FRESH_ARG OUT_DIR="$(stage_dir synthetic)" \
      TARGET_GB="$SYN_GB" \
      FORMATS="$FORMATS" \
      EVAL_RECORDS="$EVAL_RECORDS" \
      SHARD_MB="$SHARD_MB" \
      SEED="$SEED" \
      NO_STORES="$NO_STORES" \
      bash data_generate.sh
  echo
fi

# ---------------------------------------------------------------------------
# 2. The Kaggle-grounded code corpus
# ---------------------------------------------------------------------------
if wants code_base; then
  echo "=============================================================================="
  echo "==> code_base  ->  $(stage_dir code_base)   (${CB_MB} MB)"
  echo "=============================================================================="
  take_fresh
  env "${ENV_SCRUB[@]}" $FRESH_ARG OUT_DIR="$(stage_dir code_base)" \
      TARGET_MB="$CB_MB" \
      FORMATS="$FORMATS" \
      EVAL_RECORDS="$EVAL_RECORDS" \
      SHARD_MB="$SHARD_MB" \
      SEED="$SEED" \
      CHECK="$CHECK" \
      bash data_generator_code_base.sh
  echo
fi

# ---------------------------------------------------------------------------
# 3. The tool-calling corpus
#
# TARGET_MB is passed as well as VARIANTS so that a run which turns out denser than the 7.5 MB
# estimate still stops at its share rather than overshooting it.
# ---------------------------------------------------------------------------
if wants tool_use; then
  echo "=============================================================================="
  echo "==> tool_use   ->  $(stage_dir tool_use)    (${TU_MB} MB, ${TU_VARIANTS} variants)"
  echo "=============================================================================="
  take_fresh
  env "${ENV_SCRUB[@]}" $FRESH_ARG OUT_DIR="$(stage_dir tool_use)" \
      TARGET_MB="$TU_MB" \
      VARIANTS="$TU_VARIANTS" \
      FORMATS="$FORMATS" \
      EVAL_RECORDS="$EVAL_RECORDS" \
      SHARD_MB="$SHARD_MB" \
      SEED="$SEED" \
      CHECK="$CHECK" \
      ${NO_KB:+NO_KB=1} \
      bash data_generator_tool_use_call.sh
  echo
fi

# ---------------------------------------------------------------------------
# Summary
#
# With everything in one shard series the corpora can no longer be told apart by directory, so the
# breakdown is read back from each record's `meta.source` -- which is what it is there for.
# ---------------------------------------------------------------------------

elapsed=$(( $(date +%s) - started_at ))

echo "=============================================================================="
echo "  Done in $((elapsed / 60))m $((elapsed % 60))s"
echo "=============================================================================="

python3 - "$OUT_DIR" "$PRIMARY" <<'PY'
import glob
import json
import os
import sys

out_dir, primary = sys.argv[1], sys.argv[2]

# The corpus each `meta.source` belongs to. The synthetic builder emits three of them.
CORPUS = {
    "identity": "synthetic",
    "knowledge_base": "synthetic",
    "code_task": "synthetic",
    "kaggle_code_task": "code_base",
    "tool_use": "tool_use",
}
ORDER = ["synthetic", "code_base", "tool_use"]


def tally(paths):
    rows = total_bytes = 0
    per_corpus: dict[str, int] = {}
    per_source: dict[str, int] = {}
    for path in paths:
        total_bytes += os.path.getsize(path)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                source = json.loads(line).get("meta", {}).get("source", "?")
                per_source[source] = per_source.get(source, 0) + 1
                corpus = CORPUS.get(source, "other")
                per_corpus[corpus] = per_corpus.get(corpus, 0) + 1
    return rows, total_bytes, per_corpus, per_source


# Both layouts: one shared directory, or a subdirectory per corpus under NO_CROSS_DEDUP=1.
train = sorted(glob.glob(f"{out_dir}/{primary}_train-*.jsonl")) or sorted(
    glob.glob(f"{out_dir}/*/{primary}_train-*.jsonl")
)
evals = sorted(glob.glob(f"{out_dir}/{primary}_eval.jsonl")) or sorted(
    glob.glob(f"{out_dir}/*/{primary}_eval.jsonl")
)
if not train:
    print(f"  nothing found under {out_dir}")
    raise SystemExit(0)

rows, train_bytes, corpora, sources = tally(train)
eval_rows, eval_bytes, eval_corpora, _ = tally(evals)

print(f"  {'corpus':<12} {'train rows':>14} {'share':>8} {'eval rows':>12}")
for corpus in ORDER + sorted(set(corpora) - set(ORDER)):
    if corpus not in corpora:
        continue
    count = corpora[corpus]
    print(f"  {corpus:<12} {count:>14,} {count / rows:>7.1%} {eval_corpora.get(corpus, 0):>12,}")
print("  " + "-" * 50)
print(f"  {'total':<12} {rows:>14,} {'':>8} {eval_rows:>12,}")
print()
print(f"  train   {len(train)} shard(s), {train_bytes / 1e9:.2f} GB")
print(f"  eval    {len(evals)} file(s), {eval_bytes / 1e6:.1f} MB, {eval_rows:,} records")
print()
print("  by source tag:")
for source, count in sorted(sources.items(), key=lambda kv: -kv[1]):
    print(f"    {source:<20} {count:>12,}")
PY

echo
if [ "${NO_CROSS_DEDUP:-}" = "1" ]; then
  echo "==> NO_CROSS_DEDUP=1: one subdirectory per corpus, so the globs need the wildcard:"
  echo
  echo "    data:"
  echo "      reader: \"local\""
  echo "      dataset_name: \"${OUT_DIR}/*/${PRIMARY}_train-*.jsonl\""
  echo "      eval_dataset_name: \"${OUT_DIR}/*/${PRIMARY}_eval.jsonl\""
else
  echo "==> Point a training config at these two paths:"
  echo
  echo "    data:"
  echo "      reader: \"local\""
  echo "      dataset_name: \"${OUT_DIR}/${PRIMARY}_train-*.jsonl\""
  echo "      eval_dataset_name: \"${OUT_DIR}/${PRIMARY}_eval.jsonl\""
fi
echo
echo "    Shuffle on read -- the shard series is corpus-major, and max_samples takes the first"
echo "    N rows of the glob, which would otherwise be all synthetic and no tool calls."
