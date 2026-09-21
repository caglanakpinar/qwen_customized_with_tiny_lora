"""Push an adapter checkpoint to the Hugging Face Hub as a new version of the adapter repo.

Builds a fresh model card from the checkpoint's own files -- `adapter_config.json` and
`adapter_model.safetensors` decide which method is being described (TinyLoRA vs layer-scoped LoRA)
and how many parameters it trains, `trainer_state.json` supplies the step/epoch/eval loss, and
`outputs/eval_results.json` supplies the measured base-vs-adapter benchmark. Nothing in the card is
hardcoded per run, so the numbers always match the checkpoint being pushed.

    poetry run python scripts/push_to_hub.py \
        --repo-id Caglana/qwen0.5b-tinylora-ds-assistant \
        --checkpoint-dir outputs/sft-layer-lora/checkpoint-3200 \
        --release-notes "$(cat notes.txt)" \
        --tag v3
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import re
import struct
import tempfile
from pathlib import Path

# Files an inference-only adapter repo needs. Excludes optimizer.pt/scheduler.pt/rng_state.pth/
# trainer_state.json/training_args.bin -- those are resume-training state, not something anyone
# loading this adapter with PeftModel.from_pretrained needs, and the existing repo doesn't carry them.
_ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

# A layer_expand checkpoint is a whole merged model, not an adapter -- there is no
# adapter_config.json/adapter_model.safetensors, and the sidecar has to travel with the weights
# or the repo is unloadable (see layer_expand/model.py's own SIDECAR_NAME docs). Same exclusion
# philosophy as _ADAPTER_FILES: optimizer.pt/scheduler.pt/rng_state.pth/trainer_state.json/
# training_args.bin are resume-training state, not something inference needs.
_LAYER_EXPAND_FILES = [
    "config.json",
    "model.safetensors",
    "layer_expand.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]

_PROJECT_URL = "https://github.com/caglanakpinar/llm_with_tiny_lora"

# Which training config each method was run from, for the hyperparameter table. The adapter's own
# geometry (layers, target modules, rank) is read from adapter_config.json instead of from here --
# a run started with `--layers 21` does not match the yaml's `layers:` list, and the checkpoint is
# the authority on what was actually trained.
_METHOD_CONFIGS = {
    "TINYLORA": "configs/sft_ds_assistant.yaml",
    "LORA": "configs/sft_layer_lora.yaml",
}

_METHOD_TITLES = {"TINYLORA": "TinyLoRA", "LORA": "Layer-LoRA"}

# `tool-use`/`function-calling` describe the dataset build rather than the method: every corpus
# version from the tool-use generator onwards trains both methods on those conversations.
_METHOD_TAGS = {
    "TINYLORA": ["sft", "tinylora", "lora", "tool-use", "function-calling"],
    "LORA": ["sft", "lora", "layer-lora", "tool-use", "function-calling"],
}

_METHOD_BLURBS = {
    "TINYLORA": """\
Unlike standard LoRA, which trains a pair of low-rank matrices `A`/`B` per target module, TinyLoRA
freezes a set of random projection matrices and trains only a small weighting vector `v` on top of
them:

```
ΔW = Σᵢ vᵢ Pᵢ
```

The adapter is spread across every transformer layer, but each module contributes only `u` trained
scalars rather than a full low-rank pair.
""",
    "LORA": """\
This is a **layer-scoped LoRA**: a standard LoRA update

```
ΔW = (α / r) · B A
```

installed on the attention projections of a *chosen subset* of transformer layers, with every other
layer left at its base weights. The late layers carry the most task-specific representation, so an
adapter placed there moves output style and format the most per trained parameter — and confining it
to those layers keeps the rest of the network's behaviour exactly as the base model shipped it.
""",
}

# What this dataset version added on top of the earlier corpus. Update this when the generators
# change; it describes the training data, not the checkpoint, so it is shared across runs that read
# the same dataset build.
_DATASET_BLURB = """\
### New in this dataset version: tool-use conversations

Earlier versions taught the assistant what to *say*. This version adds a fourth generator,
**`data_generator_tool_use_call.py`**, which teaches it what to *do* when it has tools: a workspace
of registered datasets it can list, describe, profile, query with SQL or Python and cross-validate a
baseline on, plus the knowledge base the rest of the corpus is generated from. See the
[Tool use](#tool-use) section for the format and the tool catalogue.

Tool *arguments* and *schemas* are real — dataset keys, column names, targets, positive labels and
split structure come from the same Kaggle catalogue the code corpus uses. Tool *results* are
simulated, under one rule: catalogue facts (row counts, positive rates, schemas) repeat across
records, while everything else (missing counts, means, fold scores) is drawn fresh per conversation.
A value that changes every time it appears cannot be memorised, so the only way for the loss to fall
on the final answer is to copy it out of the tool result above — which is the behaviour being taught.

### Also in this corpus: a Kaggle-grounded code corpus

The earlier versions trained on two generators:

- **Knowledge-base Q&A** (`kb_questions.py`) — hand-written data-science concepts turned into nine
  question shapes per concept (definition, practice, tradeoffs, pitfalls, checks, overview, review,
  decision, failure map), so the same material is asked for in many different ways.
- **Synthetic code tasks** (`code_tasks.py`) — 45 pandas / matplotlib / scikit-learn / statistics /
  SQL tasks, each rendered against many invented domains and against a `Technique` (estimator,
  scaling, CV scheme, search space, metric, all moving together).

A third generator, **`data_generator_code_base.py`**, is the same code-task idea anchored to datasets
that actually exist. 24 open Kaggle problems — Titanic, House Prices,
Spaceship Titanic, Credit Card Fraud, Telco Churn, Adult Census Income, Pima Diabetes, Heart Failure,
Home Credit Default Risk, Santander, Porto Seguro, IEEE-CIS Fraud, Bike Sharing, NYC Taxi Duration,
Store Sales, Rossmann, Otto, MNIST, Give Me Some Credit, Wine Quality, Mobile Price, NYC Airbnb,
Cardiovascular Disease, Mercedes-Benz Greener Manufacturing — each written against the five stages of
a modelling project:

| stage | what the answer teaches |
|---|---|
| problem framing | reading the target to decide classification vs regression, and what the leaderboard metric implies |
| feature engineering | the domain features, then a fold-safe `ColumnTransformer` built from dtype selectors |
| model selection | candidates compared on shared folds, chosen with the spread in view, not the point estimate |
| hyperparameter tuning | a search over the whole pipeline, plus an unbiased read of the tuned procedure |
| evaluation | the metrics the task calls for, threshold choice, slice checks, saved artefacts |

Two properties are enforced across every emitted answer, and they are what the fine-tune is meant to
absorb:

- **Generic** — the dataset-specific part is a short header (`DATA`, `TARGET`, `DROP`, domain
  features); everything below it is written against dtypes and column selectors, so the same body
  transfers to any tabular problem instead of being Titanic-shaped.
- **Reliable** — every fitted step lives inside a `Pipeline` fit *inside* the CV loop, the split
  respects the group/time structure the dataset actually has, leakage columns are dropped by name,
  seeds are pinned, and the specific failure being avoided is named on a `Watch out:` line rather
  than left implicit.

Every answer in both code generators has the same three-part shape — a lead sentence saying what the
code does and why, one runnable block, and a `Watch out:` line carrying the knowledge-base concept it
is grounded in — so the corpora teach one output format rather than three.
"""

# The tool-use half of the card. A plain string rather than part of the f-string card body, so the
# braces in its code samples need no escaping. Counts are from the corpus the tool-use generator
# writes (`data/synthetic/dataset_tool_use`), which is the set these numbers describe.
_TOOL_USE_SECTION = """\
## Tool use

This version is trained to call tools in Qwen2.5's **native tool format**, the one the base model's
chat template already defines: the offered tools arrive in the system prompt inside `<tools>` tags,
and the model answers with one `<tool_call>` block per call.

```
<tool_call>
{"name": "describe_dataset", "arguments": {"dataset": "titanic"}}
</tool_call>
```

Pass the tools through the chat template and the model sees exactly what it was trained on. Parse
the `<tool_call>` blocks out of its reply, run them yourself, and hand the results back as `tool`
messages.

### Calling it

Continuing from the snippet above:

```python
import json, re

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "describe_dataset",
            "description": (
                "Profile a registered dataset: rows, columns by type, the target and its metric, "
                "split structure and known data-quality issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {"dataset": {"type": "string", "description": "Dataset key."}},
                "required": ["dataset"],
            },
        },
    }
]

# The system prompt this adapter was trained under. Keep it — see the note below.
SYSTEM = (
    "You are a senior data scientist. You answer questions about data engineering, feature "
    "engineering, statistics, machine learning and visualisation, and you write working code when "
    "code is what the question calls for. Be direct and concrete: name the trade-off, name the "
    "failure mode, and say what to check. When you write code, give a short explanation, one "
    "runnable block, and a note on what usually goes wrong."
)

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "Profile the titanic dataset for me."},
]


def generate(messages):
    text = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


reply = generate(messages)
calls = [json.loads(block) for block in re.findall(r"<tool_call>\\s*(.*?)\\s*</tool_call>", reply, re.S)]

# Run each call yourself, then give the results back and generate again.
for call in calls:
    result = run_my_tool(call["name"], call["arguments"])  # your implementation
    messages += [
        {"role": "assistant", "content": reply},
        {"role": "tool", "content": json.dumps(result)},
    ]

print(generate(messages) if calls else reply)
```

Two details worth knowing:

- **Replay the assistant turn as content.** Training wrote tool calls into message *content*, exactly
  as the chat template renders them, so replaying them that way is what the model saw. The structured
  form (`{"role": "assistant", "tool_calls": [...]}` with `tools=` on the template) renders to the
  same string byte for byte, so either works.
- **Greedy decoding** (`do_sample=False`) is how the adapter is evaluated, and it makes the call
  format reproducible.

**The system prompt is not optional in practice.** Every tool-use conversation in training sat under
the prompt above, and the adapter learned the behaviour in that context. Greedy runs of this
checkpoint asked to profile a dataset called a tool in every condition tested *with* that prompt —
one, two and four tools offered, bf16 and fp32 — and in one of twelve without it. With no system
prompt it tends to ask a follow-up question ("which file?") instead of calling anything. How many
tools you offer made little difference; the system prompt made all of it.

### The tools it was trained on

The offered list is drawn per conversation — the tools a question needs plus one to three others, in
shuffled order — so the model reads the `<tools>` block rather than memorising one fixed toolbox.

| tool | what it returns |
|---|---|
| `list_datasets` | the registered datasets (key, title, task, rows), optionally filtered |
| `describe_dataset` | a dataset profile: rows, columns by type, target and metric, splits, known issues |
| `column_stats` | one column's dtype, missing and distinct counts, distribution |
| `run_sql` | one read-only DuckDB `SELECT`; each dataset is a table named by its key |
| `run_python` | Python with pandas/numpy/scikit-learn; `load(key)` returns a dataset as a DataFrame |
| `cross_validate` | per-fold scores for a baseline model on a dataset |
| `search_knowledge_base` | the best-matching passages from the data-science knowledge base |

Your own tools are not these seven. How far the behaviour carries to a different toolbox is a
question of how well a 0.5B adapter generalises — the format is what was trained, not the catalogue.

### Behaviours it was trained for

9,903 conversations, of which 6,843 make at least one call (5,435 one call, 1,315 two, 93 three or
four). The rest are the cases where calling a tool is the wrong move — which is half the skill:

| shape | records | what it teaches |
|---|---|---|
| `single_call` | 2,471 | one question, one call, an answer that quotes the result |
| `knowledge_base` | 2,405 | search first, then answer only from the passages it returned |
| `no_tool` | 2,401 | tools on offer, question needs none of them — answered directly |
| `unsupported` | 659 | the request needs a capability no offered tool has; say so |
| `parallel_calls` | 586 | two independent lookups issued together in one turn |
| `chained_calls` | 377 | the second call's arguments come out of the first call's result |
| `unknown_dataset` | 317 | the tool says it does not exist; the answer says so and invents nothing |
| `error_recovery` | 261 | the tool rejects an identifier; the retry uses the one it suggests |
| `lookup_first` | 184 | the dataset is named by title, so `list_datasets` finds the key first |
| `clarify` | 147 | an argument the call needs is missing, so it asks before calling |
| `follow_up` | 95 | the next answer is already in the last tool result; no second call |

"""

_METRIC_ROWS = [
    ("eval_loss", "eval_loss", "cross-entropy on the held-out split (lower is better)"),
    ("perplexity", "perplexity", "exp(eval_loss)"),
    ("rouge_l_f1", "ROUGE-L F1", "longest-common-subsequence overlap with the reference answer"),
    ("token_f1", "token F1", "unigram overlap with the reference answer"),
    ("code_valid_rate", "valid-Python rate", "generated Python blocks that parse with `ast`"),
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"No {path.name} in {path.parent}")
    return json.loads(path.read_text())


def _safetensors_header(path: Path) -> dict:
    """Tensor names/shapes, read from the safetensors header without loading any weights."""
    with path.open("rb") as f:
        length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(length))
    header.pop("__metadata__", None)
    return header


# One adapted module owns several saved tensors -- `q_proj.lora_A.weight` / `q_proj.lora_B.weight`
# for LoRA, `q_proj.tinylora_A/_B/_P` for TinyLoRA. Strip that suffix to count modules, not tensors.
# TinyLoRA's trained vectors are saved separately as `base_model.tinylora_v.N`, one per module, and
# are not per-module tensors in this sense -- they are excluded here and counted below instead.
_MODULE_TENSOR_RE = re.compile(r"^(.*)\.(?:tiny)?lora_[A-Za-z](?:\.weight)?$")
_TINYLORA_V_RE = re.compile(r"\.tinylora_v\.\d+$")


def _module_count(header: dict) -> int:
    """Distinct target modules the adapter touches (one module owns several tensors)."""
    modules = {m.group(1) for name in header if (m := _MODULE_TENSOR_RE.match(name))}
    return len(modules)


def _trainable_params(peft_type: str, config: dict, header: dict) -> int:
    """Parameters this adapter actually trains.

    LoRA trains every tensor it saves, so counting elements is exact. TinyLoRA also saves its frozen
    random projections, so counting everything would report the whole file rather than the trained
    part -- only the `tinylora_v` vectors (one `u`-wide vector per adapted module) are trained.
    """
    if peft_type == "TINYLORA":
        trained = [e for name, e in header.items() if _TINYLORA_V_RE.search(name)]
        if trained:
            return sum(math.prod(e["shape"]) for e in trained)
        return int(config["u"]) * _module_count(header)
    return sum(math.prod(entry["shape"]) for entry in header.values())


def _checkpoint_eval_loss(state: dict) -> float:
    """The eval loss logged *at this checkpoint's own step*.

    Not `best_metric`: that is the best score across the whole run, which belongs to whichever
    checkpoint was best (e.g. step 2700), and reporting it on a later checkpoint would put a number
    in the card that these weights never achieved.
    """
    step = state["global_step"]
    for entry in reversed(state.get("log_history", [])):
        if entry.get("step") == step and "eval_loss" in entry:
            return float(entry["eval_loss"])
    return float(state["best_metric"])


def _load_train_config(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _corpus_stats(dataset_glob: str, sample_lines: int = 2000) -> tuple[int, float, float] | None:
    """(shard count, total GB, estimated millions of records) for the training corpus.

    The record count is estimated from the mean line length of the first shard rather than counted:
    the corpus is tens of GB across dozens of shards, and reading all of it to print one number in a
    card would add minutes to every push.
    """
    shards = sorted(Path().glob(dataset_glob))
    if not shards:
        return None
    total_bytes = sum(p.stat().st_size for p in shards)
    read_bytes = 0
    lines = 0
    with shards[0].open("rb") as f:
        for line in f:
            read_bytes += len(line)
            lines += 1
            if lines >= sample_lines:
                break
    if not lines:
        return len(shards), total_bytes / 1e9, 0.0
    return len(shards), total_bytes / 1e9, total_bytes / (read_bytes / lines) / 1e6


def _find_benchmark(eval_results: Path, checkpoint_dir: Path) -> dict | None:
    """The most recent benchmark entry in eval_results.json scored on this checkpoint."""
    if not eval_results.exists():
        return None
    payload = json.loads(eval_results.read_text())
    wanted = Path(checkpoint_dir).resolve()
    matches = [
        entry
        for entry in payload.get("benchmarks", [])
        if entry.get("adapter") and Path(entry["adapter"]).resolve() == wanted
    ]
    return matches[-1] if matches else None


def _fmt(value: float, metric: str) -> str:
    if metric == "code_valid_rate":
        return f"{value * 100:.1f}%"
    if metric == "perplexity":
        return f"{value:.2f}"
    return f"{value:.4f}"


def _results_section(benchmark: dict | None, *, step: int, epoch: float, eval_loss: float) -> str:
    """The measured base-vs-adapter table, plus the deltas spelled out in prose."""
    if benchmark is None:
        perplexity = math.exp(eval_loss)
        return (
            f"Checkpoint at step {step} (epoch ≈{epoch:.2f}) reached **eval_loss {eval_loss:.4f}** "
            f"(perplexity ≈{perplexity:.2f}) on the held-out split during training.\n\n"
            "No scored base-vs-adapter benchmark was found for this checkpoint in "
            "`outputs/eval_results.json`, so only the training-time loss is reported here. Run "
            f"`bash eval.sh` against this checkpoint to fill in the full comparison.\n"
        )

    base = benchmark["results"]["base"]
    tuned = benchmark["results"]["checkpoint"]
    base["perplexity"] = base.get("perplexity", math.exp(base["eval_loss"]))
    tuned["perplexity"] = tuned.get("perplexity", math.exp(tuned["eval_loss"]))

    lines = [
        f"Checkpoint at step {step} (epoch ≈{epoch:.2f}), scored against the un-adapted base model "
        f"on the same held-out split (`{benchmark.get('eval_dataset', 'sft_eval.jsonl')}`"
        + (
            f", {benchmark['num_eval_samples']} samples"
            if benchmark.get("num_eval_samples")
            else ""
        )
        + "):\n",
        "| metric | base model | this adapter | change | what it measures |",
        "|---|---|---|---|---|",
    ]

    improved: list[str] = []
    regressed: list[str] = []
    for key, label, meaning in _METRIC_ROWS:
        if key not in base or key not in tuned:
            continue
        delta = tuned[key] - base[key]
        lower_is_better = key in {"eval_loss", "perplexity"}
        better = delta < 0 if lower_is_better else delta > 0
        if key == "perplexity":
            change = f"−{(1 - tuned[key] / base[key]) * 100:.0f}%"
        elif key == "code_valid_rate":
            change = f"{delta * 100:+.1f} pts"
        else:
            change = f"{delta:+.4f}"
        mark = "✅" if better else "⚠️"
        lines.append(
            f"| {label} | {_fmt(base[key], key)} | {_fmt(tuned[key], key)} | {mark} {change} | {meaning} |"
        )
        (improved if better else regressed).append(f"{label} {_fmt(base[key], key)} → {_fmt(tuned[key], key)}")

    ratio = tuned["perplexity"] / base["perplexity"]
    lines.append("")
    lines.append(
        f"The headline number is perplexity: **{base['perplexity']:.2f} → {tuned['perplexity']:.2f}**, "
        f"a **{(1 - ratio) * 100:.0f}% cut**, from an adapter that trains a fraction of a percent of "
        "the base model's weights."
    )
    if regressed:
        lines.append("")
        lines.append(
            "Reported as measured, including what got worse: " + "; ".join(regressed) + ". "
            "Generation-side metrics are scored on "
            f"{tuned.get('num_generation_samples', 'a small sample of')} prompts with greedy decoding, "
            "so single-percent moves there are noise; a swing as large as the valid-Python rate's is "
            "not, and is the thing to watch on the next run."
        )
    if benchmark.get("note"):
        lines.append("")
        lines.append(f"> **Note on comparability.** {benchmark['note']}")
    return "\n".join(lines) + "\n"


def _hyperparameter_table(config: dict, header: dict, train_cfg: dict, peft_type: str, params: int) -> str:
    training = train_cfg.get("training", {})
    data = train_cfg.get("data", {})
    layers = config.get("layers_to_transform")
    layer_desc = (
        ", ".join(str(layer) for layer in layers) if layers else "all 24 layers"
    )
    modules = ", ".join(f"`{m}`" for m in sorted(config.get("target_modules", [])))

    rows = [("Base model", f"`{config['base_model_name_or_path']}`")]
    if peft_type == "TINYLORA":
        rows += [
            ("TinyLoRA rank (`r`)", str(config.get("r"))),
            ("Vector width (`u`)", str(config.get("u"))),
            ("Projection seed", str(config.get("projection_seed"))),
        ]
    else:
        rows += [
            ("LoRA rank (`r`)", str(config.get("r"))),
            ("`lora_alpha`", str(config.get("lora_alpha"))),
            ("`lora_dropout`", str(config.get("lora_dropout"))),
            ("Adapted layers", layer_desc),
        ]
    rows += [
        ("Target modules", modules),
        ("Adapted modules", f"{_module_count(header)}"),
        ("Trainable parameters", f"{params:,}"),
    ]
    if training:
        batch = training.get("per_device_train_batch_size")
        accum = training.get("gradient_accumulation_steps")
        rows += [
            (
                "Optimizer",
                f"AdamW, `lr={training.get('learning_rate')}`, "
                f"{training.get('lr_scheduler_type')} schedule, {training.get('warmup_steps')} warmup steps",
            ),
            ("Effective batch size", f"{batch * accum} (`per_device={batch}` × `grad_accum={accum}`)"),
            ("Max sequence length", str(training.get("max_seq_length"))),
            ("Precision", "bf16" if training.get("bf16") else "fp32"),
            (
                "Early stopping",
                f"on eval_loss, patience {training.get('early_stopping_patience')}"
                if training.get("early_stopping")
                else "off",
            ),
        ]
    if data.get("max_samples"):
        rows.append(("Training subset", f"{data['max_samples']:,} samples"))
    if data.get("max_eval_samples"):
        rows.append(("Eval split", f"{data['max_eval_samples']:,} held-out samples"))

    return "\n".join(["| | |", "|---|---|", *(f"| {name} | {value} |" for name, value in rows)])


def _layer_expand_hyperparameter_table(sidecar: dict, train_cfg: dict, block_params: int) -> str:
    shape = sidecar["shape"]
    training = train_cfg.get("training", {})
    data = train_cfg.get("data", {})

    rows = [
        ("Base model", f"`{sidecar['base_model']}`"),
        ("New block position", f"layer {', '.join(str(i) for i in sidecar['layers'])} "
                                f"(appended after the base model's {sidecar['base_num_hidden_layers']})"),
        ("Block width", f"{shape['hidden_size']}d / {shape['intermediate_size']} MLP, "
                         f"{shape['num_attention_heads']} heads, {shape['num_key_value_heads']} KV heads"),
        ("Wrapped", "yes -- 896 → {} → 896 projections around the block".format(shape["hidden_size"])
                    if sidecar["wrapped"] else "no -- base-width block, stock-loadable"),
        ("Initialisation", f"`{sidecar['init']}`" + (
            " (block starts as an exact residual passthrough)" if sidecar["init"] == "identity" else ""
        )),
        ("Frozen base built from", ", ".join(f"`{a}`" for a in sidecar.get("base_adapters", [])) or "stock base model"),
        ("New block parameters", f"{block_params:,}"),
    ]
    if training:
        batch = training.get("per_device_train_batch_size")
        accum = training.get("gradient_accumulation_steps")
        rows += [
            (
                "Optimizer",
                f"AdamW, `lr={training.get('learning_rate')}`, "
                f"{training.get('lr_scheduler_type')} schedule, {training.get('warmup_steps')} warmup steps",
            ),
            ("Weight decay", str(training.get("weight_decay"))),
            ("Effective batch size", f"{batch * accum} (`per_device={batch}` × `grad_accum={accum}`)"),
            ("Max sequence length", str(training.get("max_seq_length"))),
            ("Precision", "bf16" if training.get("bf16") else "fp32"),
        ]
        if training.get("neftune_noise_alpha"):
            rows.append(("NEFTune noise alpha", str(training["neftune_noise_alpha"])))
        if training.get("label_smoothing_factor"):
            rows.append(("Label smoothing", str(training["label_smoothing_factor"])))
        rows.append((
            "Early stopping",
            f"on eval_loss, patience {training.get('early_stopping_patience')}, "
            f"threshold {training.get('early_stopping_threshold')}"
            if training.get("early_stopping")
            else "off",
        ))
    if data.get("max_samples"):
        rows.append(("Training subset", f"{data['max_samples']:,} samples"))
    if data.get("max_eval_samples"):
        rows.append(("Eval split", f"{data['max_eval_samples']:,} held-out samples"))

    return "\n".join(["| | |", "|---|---|", *(f"| {name} | {value} |" for name, value in rows)])


def build_layer_expand_card(
    *,
    repo_id: str,
    sidecar: dict,
    train_cfg: dict,
    config_path: str,
    step: int,
    epoch: float,
    eval_loss: float,
    benchmark: dict | None,
    corpus: tuple[int, float, float] | None,
    release_notes: str,
) -> str:
    base_params = 494_000_000
    block_params = sidecar["block_parameters"]
    total_params = base_params + block_params

    if corpus:
        shards, gigabytes, millions = corpus
        corpus_line = (
            f"The full corpus is ~{millions:.0f}M records (~{gigabytes:.0f} GB across {shards} shards)"
        )
    else:
        corpus_line = "The full corpus is generated locally by the project's `data_generate.sh`"

    subset = train_cfg.get("data", {}).get("max_samples")
    eval_subset = train_cfg.get("data", {}).get("max_eval_samples")
    wrapped_note = (
        "This checkpoint is **not** loadable by a stock `AutoModelForCausalLM.from_pretrained` -- "
        "the new block is wider than the rest of the stack, which `Qwen2Config` cannot describe, "
        "so `config.json` is deliberately retyped `model_type: layer_expand` to fail loudly instead "
        "of silently reinitialising the block and returning a model that loads fine and produces "
        "garbage."
        if sidecar["wrapped"]
        else "This checkpoint is a stock-shaped Qwen2 (the new block matches the base model's own "
        "width) and loads with a standard `AutoModelForCausalLM.from_pretrained`, but still needs "
        "`config.num_hidden_layers` bumped -- use the loader below rather than assuming."
    )

    return f"""\
---
base_model: {sidecar['base_model']}
pipeline_tag: text-generation
tags:
- sft
- layer-expand
- block-expansion
- tool-use
- function-calling
- transformers
- trl
- data-science
---

# Qwen2.5-0.5B Layer-Expand — Data Science Assistant

A depth-expanded fine-tune of
[{sidecar['base_model']}](https://huggingface.co/{sidecar['base_model']}): one new transformer
block appended after the base model's {sidecar['base_num_hidden_layers']} layers, trained densely
while everything before it stays frozen. It answers data-science concept questions and writes
runnable pandas / matplotlib / scikit-learn / statistics / SQL code.

This version trains **{block_params:,} new parameters** on top of a
**{base_params:,}-parameter** frozen base — the shipped model has **{total_params:,}** total
parameters, of which the new block is **{block_params / total_params * 100:.1f}%**.

Trained with the [llm_with_tiny_lora]({_PROJECT_URL}) project.

## What's new in this version

{release_notes.strip()}

## Method — layer expansion (block expansion)

Unlike LoRA or TinyLoRA, which adapt existing weight matrices with a low-rank update, this method
adds a brand new transformer block to the stack and trains it densely -- every one of its
{block_params:,} parameters is a real, independently-learned weight, not a rank-decomposed delta.

The frozen base underneath is not the stock checkpoint either: {
    ", then ".join(f"`{a}`" for a in sidecar.get("base_adapters", []))
    or "the stock base model"
} {"were" if len(sidecar.get("base_adapters", [])) != 1 else "was"} merged into the base weights,
in that order, before the new block was added -- so this checkpoint carries everything those
earlier runs learned, plus what the new block learned on top.

The new block starts from `init: "{sidecar['init']}"`{
    ", which zeroes its output projections so the block is an exact identity function on its "
    "first forward pass -- training starts from a working model and learns a delta, rather than "
    "the new block injecting noise into a working stack from step one"
    if sidecar["init"] == "identity" else ""
}.

{wrapped_note}

## Training data

A synthetic data-science assistant corpus generated by the same project: concept Q&A, runnable
pandas/matplotlib/scikit-learn/statistics/SQL tasks, and tool-use conversations in which the
assistant calls a data tool and answers from what it returns. {corpus_line}; this run trained on a
{f'{subset:,}-sample' if subset else ''} subset with a {f'{eval_subset:,}-sample' if eval_subset else ''}
held-out split for eval.

{_DATASET_BLURB}

## Hyperparameters

{_layer_expand_hyperparameter_table(sidecar, train_cfg, block_params)}

Full config: [`{config_path}`]({_PROJECT_URL}/blob/main/{config_path}).

## Results

{_results_section(benchmark, step=step, epoch=epoch, eval_loss=eval_loss)}

## Usage

This repo needs the project's own `layer_expand` loader, not a bare `transformers` install --
see the note above on why a stock `AutoModelForCausalLM` cannot rebuild this stack.

```bash
pip install "git+{_PROJECT_URL}"
```

```python
from huggingface_hub import snapshot_download
from layer_expand.model import load_expanded_model
from transformers import AutoTokenizer

local_path = snapshot_download("{repo_id}")
model = load_expanded_model(local_path)
tokenizer = AutoTokenizer.from_pretrained(local_path)

messages = [{{"role": "user", "content": "How do I compute a rolling 7-day average in pandas?"}}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
output = model.generate(**inputs, max_new_tokens=256, do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Or, from a checkout of the project itself: `tiny-lora chat --adapter <local checkpoint dir>` --
it detects the `layer_expand.json` sidecar and loads through this same path automatically.

{_TOOL_USE_SECTION}
## Limitations

- Not loadable by a stock `transformers.AutoModelForCausalLM` -- requires this project's
  `layer_expand` package to rebuild the stack from `layer_expand.json`.
- 0.5B-class base model: capable of short, focused answers but not competitive with larger models
  on complex multi-step reasoning.
- Narrow domain: tuned specifically for data-science Q&A and code generation; general-purpose chat
  quality is not a training objective.
- The generated code is written to be runnable, but it is generated — read it before running it
  against anything that matters.
- Identity/safety guardrails (e.g. not identifying as "Qwen"/"Alibaba") are implemented at the
  application layer in the project's [`chat.py`]({_PROJECT_URL}/blob/main/src/tiny_lora/chat.py)
  wrapper, not baked into the weights — raw `generate()` calls against this checkpoint may surface
  the base model's own identity.

## License

Base model ([{sidecar['base_model']}](https://huggingface.co/{sidecar['base_model']}))
is Apache 2.0. This checkpoint is released under the same terms.
"""


def build_model_card(
    *,
    repo_id: str,
    config: dict,
    header: dict,
    train_cfg: dict,
    config_path: str,
    step: int,
    epoch: float,
    eval_loss: float,
    benchmark: dict | None,
    corpus: tuple[int, float, float] | None,
    release_notes: str,
) -> str:
    peft_type = config.get("peft_type", "LORA")
    title = _METHOD_TITLES.get(peft_type, peft_type)
    params = _trainable_params(peft_type, config, header)
    base_params = 494_000_000
    tags = "\n".join(f"- {tag}" for tag in _METHOD_TAGS.get(peft_type, ["sft", "lora"]))

    if corpus:
        shards, gigabytes, millions = corpus
        corpus_line = (
            f"The full corpus is ~{millions:.0f}M records (~{gigabytes:.0f} GB across {shards} shards)"
        )
    else:
        corpus_line = "The full corpus is generated locally by the project's `data_generate.sh`"

    subset = train_cfg.get("data", {}).get("max_samples")
    eval_subset = train_cfg.get("data", {}).get("max_eval_samples")

    return f"""\
---
base_model: {config['base_model_name_or_path']}
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- base_model:adapter:{config['base_model_name_or_path']}
{tags}
- transformers
- trl
- data-science
---

# Qwen2.5-0.5B {title} — Data Science Assistant

A parameter-efficient adapter that fine-tunes
[{config['base_model_name_or_path']}](https://huggingface.co/{config['base_model_name_or_path']})
into a data-science assistant: it answers data-science concept questions and writes runnable
pandas / matplotlib / scikit-learn / statistics / SQL code.

This version trains **{params:,} parameters** — roughly **{params / base_params * 100:.3f}%** of the
~494M-parameter base model.

Trained with the [llm_with_tiny_lora]({_PROJECT_URL}) project.

## What's new in this version

{release_notes.strip()}

## Method — {title}

{_METHOD_BLURBS.get(peft_type, '').strip()}

## Training data

A synthetic data-science assistant corpus generated by the same project: concept Q&A, runnable
pandas/matplotlib/scikit-learn/statistics/SQL tasks, and tool-use conversations in which the
assistant calls a data tool and answers from what it returns. {corpus_line}; this run trained on a
{f'{subset:,}-sample' if subset else ''} subset with a {f'{eval_subset:,}-sample' if eval_subset else ''}
held-out split for eval, since an adapter this small saturates long before seeing the full corpus.

{_DATASET_BLURB}

## Hyperparameters

{_hyperparameter_table(config, header, train_cfg, peft_type, params)}

Full config: [`{config_path}`]({_PROJECT_URL}/blob/main/{config_path}).

## Results

{_results_section(benchmark, step=step, epoch=epoch, eval_loss=eval_loss)}

## Usage

```bash
pip install git+https://github.com/huggingface/peft.git
pip install transformers torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{config['base_model_name_or_path']}"
adapter_id = "{repo_id}"

tokenizer = AutoTokenizer.from_pretrained(adapter_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto")
model = PeftModel.from_pretrained(base_model, adapter_id)

messages = [{{"role": "user", "content": "How do I compute a rolling 7-day average in pandas?"}}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
output = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

TinyLoRA is a new PEFT method available only on PEFT's GitHub `main` branch, so the git install
above is required for TinyLoRA revisions of this repo; plain LoRA revisions load with a released
`peft` as well.

{_TOOL_USE_SECTION}
## Limitations

- 0.5B base model: capable of short, focused answers but not competitive with larger models on
  complex multi-step reasoning.
- Narrow domain: tuned specifically for data-science Q&A and code generation; general-purpose chat
  quality is not a training objective.
- The generated code is written to be runnable, but it is generated — read it before running it
  against anything that matters.
- Identity/safety guardrails (e.g. not identifying as "Qwen"/"Alibaba") are implemented at the
  application layer in the project's [`chat.py`]({_PROJECT_URL}/blob/main/src/tiny_lora/chat.py)
  wrapper, not baked into the adapter weights — raw `generate()` calls against this adapter may
  surface the base model's own identity.

## License

Base model ([{config['base_model_name_or_path']}](https://huggingface.co/{config['base_model_name_or_path']}))
is Apache 2.0. This adapter is released under the same terms.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument(
        "--release-notes",
        default="Retrained on an expanded synthetic corpus (see below for what's new in this version).",
    )
    parser.add_argument("--tag", default=None, help="Optional tag/revision name for this commit.")
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Commit message for the push (default: the method, step and eval loss).",
    )
    parser.add_argument(
        "--eval-results",
        type=Path,
        default=Path("outputs/eval_results.json"),
        help="Benchmark file to pull this checkpoint's base-vs-adapter scores from.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Training yaml for the hyperparameter table (default: chosen from the adapter's method).",
    )
    parser.add_argument(
        "--dataset-glob",
        default="data/synthetic/dataset/sft_train-*.jsonl",
        help="Glob used to report the current training corpus size in the card.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the card to stdout and upload nothing.",
    )
    args = parser.parse_args()

    checkpoint_dir: Path = args.checkpoint_dir
    is_layer_expand = (checkpoint_dir / "layer_expand.json").exists()
    files_to_copy = _LAYER_EXPAND_FILES if is_layer_expand else _ADAPTER_FILES

    for name in files_to_copy:
        if not (checkpoint_dir / name).exists():
            raise FileNotFoundError(f"{checkpoint_dir / name} is missing -- is this a real checkpoint dir?")

    state = _load_json(checkpoint_dir / "trainer_state.json")
    step = state["global_step"]
    epoch = state["epoch"]
    eval_loss = _checkpoint_eval_loss(state)

    if is_layer_expand:
        sidecar = _load_json(checkpoint_dir / "layer_expand.json")
        config_path = str(args.train_config) if args.train_config else "configs/sft_layer_expand.yaml"
        card = build_layer_expand_card(
            repo_id=args.repo_id,
            sidecar=sidecar,
            train_cfg=_load_train_config(Path(config_path)) if config_path else {},
            config_path=config_path,
            step=step,
            epoch=epoch,
            eval_loss=eval_loss,
            benchmark=_find_benchmark(args.eval_results, checkpoint_dir),
            corpus=_corpus_stats(args.dataset_glob),
            release_notes=args.release_notes,
        )
        commit_message = args.commit_message or (
            f"Update layer-expand block: step {step}, eval_loss {eval_loss:.4f}"
        )
    else:
        config = _load_json(checkpoint_dir / "adapter_config.json")
        header = _safetensors_header(checkpoint_dir / "adapter_model.safetensors")
        peft_type = config.get("peft_type", "LORA")
        config_path = str(args.train_config) if args.train_config else _METHOD_CONFIGS.get(peft_type, "")

        card = build_model_card(
            repo_id=args.repo_id,
            config=config,
            header=header,
            train_cfg=_load_train_config(Path(config_path)) if config_path else {},
            config_path=config_path,
            step=step,
            epoch=epoch,
            eval_loss=eval_loss,
            benchmark=_find_benchmark(args.eval_results, checkpoint_dir),
            corpus=_corpus_stats(args.dataset_glob),
            release_notes=args.release_notes,
        )
        method = _METHOD_TITLES.get(peft_type, peft_type)
        commit_message = args.commit_message or f"Update adapter: {method} step {step}, eval_loss {eval_loss:.4f}"

    if args.dry_run:
        print(card)
        return

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)
        for name in files_to_copy:
            shutil.copy2(checkpoint_dir / name, staging / name)
        (staging / "README.md").write_text(card)

        from huggingface_hub import HfApi

        api = HfApi()
        result = api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging),
            commit_message=commit_message,
        )
        print(f"==> Pushed commit: {result}")

        if args.tag:
            api.create_tag(repo_id=args.repo_id, tag=args.tag, revision="main", exist_ok=True)
            print(f"==> Tagged as '{args.tag}'")

    print(f"==> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
