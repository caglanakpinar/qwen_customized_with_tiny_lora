# LLM with Tiny LoRA

Fine-tune large language models with **TinyLoRA** — an extremely parameter-efficient adaptation method that trains as few as 1–32 scalars instead of full low-rank matrices.

## Overview

[TinyLoRA](https://arxiv.org/abs/2602.04118) builds on LoRA and LoRA-XS by replacing trainable low-rank matrices with a weighted sum of fixed random projection matrices:

```
R = Σᵢ vᵢ Pᵢ
```

Where `v ∈ R^u` is a tiny trainable vector (typically 13–32 parameters) and `Pᵢ` are fixed random matrices. Combined with reinforcement learning (e.g. GRPO), this approach can recover ~90% of full fine-tuning gains while training **1000× fewer parameters**.

## Installation

Requires [Poetry](https://python-poetry.org/) and Python 3.10+.

```bash
git clone https://github.com/caglanakpinar/llm_with_tiny_lora.git
cd llm_with_tiny_lora

poetry install

# Optional: 4-bit quantization (Linux/CUDA only)
poetry install --extras quant
```

> **Note:** TinyLoRA requires PEFT from GitHub main (configured in `pyproject.toml`). On macOS, skip the `quant` extra and use `--no-quant` when training.

## CLI Usage

All commands run through the Click CLI:

```bash
# Show help
poetry run tiny-lora --help

# Print config and trainable parameter count
poetry run tiny-lora info --config configs/grpo_default.yaml

# Supervised fine-tuning (SFT)
poetry run tiny-lora sft --config configs/sft_default.yaml

# SFT with a from-scratch LoRA + Qwen implementation on TensorFlow -- no trl SFTTrainer/
# GRPOTrainer, no peft; see src/lora_base/. Requires `poetry install -E tf-lora`.
poetry run tiny-lora sft-tf --config configs/sft_lora_base.yaml

# SFT on specific transformer layers only -- see `layer_lora` below
poetry run layer_lora sft --config configs/sft_layer_lora.yaml --layers 20-23

# GRPO reinforcement learning (recommended for reasoning)
poetry run tiny-lora grpo --config configs/grpo_default.yaml

# Override config from CLI
poetry run tiny-lora grpo \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --u 32 \
  --weight-tying 1.0 \
  --max-samples 200 \
  --output-dir outputs/grpo-u32

# --adapter below is outputs/sft-ds-assistant/adapter/ (written once SFT completes) or, while a
# run is still in progress or was interrupted early, the newest checkpoint-N/ under the same dir:
ADAPTER="outputs/sft-ds-assistant/adapter"
[ -d "$ADAPTER" ] || ADAPTER="$(ls -dt outputs/sft-ds-assistant/checkpoint-*/ 2>/dev/null | head -1)"

# Compare base-model vs checkpoint eval loss/perplexity on the configured eval split
poetry run tiny-lora eval \
  --config configs/sft_ds_assistant.yaml \
  --adapter "$ADAPTER" \
  --max-eval-samples 200

# Interactive chat REPL against a trained adapter
poetry run tiny-lora chat --adapter "$ADAPTER" --no-quant

# Chat with every option set explicitly
poetry run tiny-lora chat \
  --adapter "$ADAPTER" \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --no-quant \
  --max-new-tokens 512 \
  --temperature 0.7 \
  --system "You are a senior data scientist." \
  --trust-remote-code \
  --summarize-after 6 \
  --keep-recent 2 \
  --memory-dir outputs/chat_memory \
  --db-path data/data_science_dbs

# Browser chat UI against a trained adapter (same options as `chat`, plus --host/--port)
poetry run tiny-lora serve --adapter "$ADAPTER" --no-quant

# KServe payload/response chat inference API (same options as `chat`, plus --model-name/--http-port)
poetry run tiny-lora chat-api --adapter "$ADAPTER" --no-quant
```

### Commands

| Command | Description |
|---------|-------------|
| `sft` | Supervised fine-tuning with TinyLoRA |
| `grpo` | GRPO RL training with TinyLoRA |
| `eval` | Compare base-model vs checkpoint eval loss/perplexity on the eval split |
| `chat` | Interactive chat REPL against a trained adapter |
| `serve` | Browser chat UI (`web/`) against a trained adapter |
| `chat-api` | KServe payload/response chat inference API against a trained adapter |
| `info` | Print config and trainable parameter count |
| `show-config` | Display a YAML config as JSON |

`eval` writes its results to `<adapter>/evals/eval-<timestamp>.json` alongside printing them --
one record per run, so re-evaluating a checkpoint against a different split or sample cap builds up
a history rather than overwriting the previous number. Each record carries the config, eval split,
row count and the base-vs-checkpoint delta. Use `--evals-dir` to write elsewhere, or `--no-save` to
only print:

```bash
poetry run tiny-lora eval \
  --config configs/sft_ds_assistant.yaml \
  --adapter outputs/sft-ds-assistant/checkpoint-5000 \
  --max-eval-samples 1000
# Results on the held-out eval split (lower is better):
#                eval_loss    perplexity
# base              2.5833       13.2408
# checkpoint        1.8204        6.1745
#
# Saved to outputs/sft-ds-assistant/checkpoint-5000/evals/eval-20260822T164500Z.json
```

> **Note:** `--adapter` takes any saved adapter or checkpoint dir, e.g. `outputs/sft-ds-assistant/adapter`
> (written once SFT finishes) or an intermediate `outputs/sft-ds-assistant/checkpoint-500` (written
> every `training.save_steps`, so one exists as soon as the first checkpoint is saved — no need to
> wait for the run to finish or complete every `max_steps`). The base model is read automatically
> from the adapter's `adapter_config.json`. `eval` scores the full `data.eval_dataset_name` split by
> default — set `data.max_eval_samples` (or pass `--max-eval-samples`) to cap it, since it loads and
> scores two full models (base + checkpoint).

### `chat` arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--adapter` (required) | — | Saved adapter or checkpoint dir, e.g. `outputs/sft-ds-assistant/adapter` or `outputs/sft-ds-assistant/checkpoint-500`. |
| `--model` | read from the adapter | Override the base model instead of resolving it from `adapter_config.json`. |
| `--no-quant` | off | Disable 4-bit quantization and load in bf16 (required on macOS, where bitsandbytes is Linux/CUDA-only). |
| `--max-new-tokens` | `512` | Max tokens generated per reply. |
| `--temperature` | `0.7` | Sampling temperature; `0` for greedy decoding. |
| `--system` | none | Optional system prompt prepended to the conversation. |
| `--trust-remote-code` | off | Trust remote code when loading the base model. |
| `--summarize-after` | `6` | Fold older turns into a running key-points summary once history exceeds this many turns. |
| `--keep-recent` | `2` | Number of most recent turns kept verbatim (unsummarized) when folding. |
| `--memory-dir` | `outputs/chat_memory` | Where persisted running-summaries are written — see below. |
| `--db-path` | none | Knowledge base dir, e.g. `data/data_science_dbs` (must contain `faiss_index/` and `chroma_db/`, built by that store's `dataset.py`). When set, every turn is retrieval-augmented with the top matches from it. |

Every 5th time the chat folds turns into a summary, that summary is embedded with the chat model's
own hidden states and appended to a FAISS index (`<memory-dir>/faiss_index/`) and a Chroma collection
(`<memory-dir>/chroma_db/`), so long conversations leave a searchable trail of what was discussed.

#### Identity guardrail

Asked who it is, the base Qwen2.5 answers "a large language model developed by Alibaba Cloud" — true
of the base model, wrong for this assistant. Two layers handle that, in [`chat.py`](src/tiny_lora/chat.py).

**Identity questions are answered, not generated.** "Who are you", "describe yourself", "who made
you", "what model are you", "are you ChatGPT" and similar are matched by `_IDENTITY_QUESTION_RE` and
answered with `IDENTITY_REPLY` verbatim, with no forward pass at all. Prompting the model with the
persona instead was tried and abandoned: it stopped the Alibaba leak but had the 0.5B base
confabulating replacements — *"I'm a data scientist trained at Oxford University"*, *"I'm an AI
assistant here at Google"* — neither containing a banned word, both false. A model this size does not
follow a persona instruction reliably enough to be the last word on what it is, and the question has
exactly one correct answer, so sampling one buys nothing. Edit `IDENTITY_REPLY` to change the
persona, or pass `identity_reply=None` to a `ChatSession` to generate these answers instead.

**Every other reply is filtered.** Output is checked against `BANNED_TERMS` (`qwen`, `alibaba`, `anthropic`, `claude`, `openai`, `chatgpt`
and variants, matched case-insensitively on word boundaries). On a hit the model is asked to rewrite,
told which words it used; after `GUARDRAIL_RETRIES` (2) failed rewrites the offending *sentences* are
deleted, and if that empties the reply `GUARDRAIL_FALLBACK` is sent instead. A clean reply — the
common case — costs one generation, so the guardrail is free unless it actually fires.

The word list deliberately omits `google`, `meta`, `gemini` and `llama`: they are ordinary vocabulary for
a data-science assistant (Google Colab, meta-learning), and banning them would delete good answers.
Edit `BANNED_TERMS` if that trade-off runs the other way for you. The guardrail covers assistant
replies in every entry point (`chat`, `serve`, `chat-api`, all of which go through `ChatSession.send`);
it does not filter the running conversation summaries, which paraphrase what the *user* said too.

### `serve`: browser chat UI

`serve` takes every `chat` argument above (same model/summary/memory/knowledge-base options — one
model is loaded once at startup and shared across browser tabs, each tab getting its own session via a
`session_id` held in `localStorage`) plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host interface to bind the web server to. |
| `--port` | `8000` | Port to serve the chat UI on. |

Three shared flags default higher here than in `chat`, so a browser conversation keeps much more of
itself verbatim before anything is folded away: `--max-new-tokens` is `1024`, `--summarize-after` is
`20` turns, and `--keep-recent` is `8` turns. Qwen2.5-0.5B-Instruct's own context window is 32k
tokens, so those still leave headroom; raise them further (or lower them, if generation gets slow) on
the command line.

The frontend lives under [`web/`](web/) — `web/app.py` is a small FastAPI app (`GET /`, `POST
/api/chat`, `POST /api/reset`) and `web/static/` is a vanilla HTML/CSS/JS chat interface with no
build step.

### `chat-api`: KServe inference API

`chat-api` takes every `chat` argument above (one model loaded once at startup, one `ChatSession`
per `session_id`, identical generation/summarization/retrieval behavior) plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | `tinylora-chat` | KServe model name; requests go to `/v1/models/<model-name>:predict`. |
| `--http-port` | `8080` | Port KServe serves the API on. |

```bash
curl http://127.0.0.1:8080/v1/models/tinylora-chat:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"message": "What is precision at k?"}]}'
# -> {"predictions": [{"session_id": "...", "reply": "..."}]}
# pass that session_id back on the next call to continue the same conversation
```

This wraps [`src/tiny_lora/chat_api.py`](src/tiny_lora/chat_api.py)'s `ChatModel` (a
`kserve.Model`) so the adapter can be deployed as a standard KServe `InferenceService` on
Kubernetes. **`kserve` is not installed by `poetry install`** — it pins `protobuf>=6`, which
conflicts with this project's `protobuf ^4.25.0` (needed by transformers/sentencepiece). Install it
separately in whichever environment runs `chat-api`, e.g. `pip install kserve` (the same
arrangement `grpo` uses for `vllm`).

## `layer_lora`: fine-tuning specific layers

`layer_lora` trains a standard LoRA adapter on **only the transformer layers you name**, leaving
every other layer at its base weights. It has its own CLI:

```bash
# Adapt layers 20-23 (the last four of Qwen2.5-0.5B's 24)
poetry run layer_lora sft --config configs/sft_layer_lora.yaml --layers 20-23 --no-quant

# A single layer, or an arbitrary mix -- ranges are inclusive on both ends
poetry run layer_lora sft --config configs/sft_layer_lora.yaml --layers 7
poetry run layer_lora sft --config configs/sft_layer_lora.yaml --layers 0-3,11,20-23
```

Which layers to adapt is the point of the module, so it is checked rather than assumed: an index
outside the model's layer count is rejected before any weights load, and the run prints the layers
and projections that actually received an adapter.

```
Adapted layer 22: k_proj, q_proj, v_proj, layer 23: k_proj, q_proj, v_proj.
trainable params: 61,440 || all params: 494,094,208 || trainable%: 0.0124
```

| `layer_lora:` key | Default | Description |
|---|---|---|
| `layers` | *(required)* | Transformer layers to adapt, by 0-based index. |
| `target_modules` | `q_proj, k_proj, v_proj` | Projections adapted within each of those layers. |
| `layers_pattern` | `layers` | Name segment before the layer index in `model.layers.N`. |
| `r` / `lora_alpha` | `8` / `16` | LoRA rank and scaling numerator. |
| `init_from_checkpoint` | `null` | `null`, `"auto"`, or a path — see below. |

### Continuing from a checkpoint

Two separate mechanisms pick up earlier weights, and they do different things:

- **`init_from_checkpoint`** starts a *new* run from a **finished** adapter. `"auto"` uses
  `<output_dir>/adapter` if a previous run left one and starts from the base model otherwise; a
  path points anywhere, including another run's `checkpoint-N`.
- **An interrupted run** is resumed automatically from `<output_dir>/checkpoint-N`, restoring the
  optimizer and LR-schedule state along with the weights. This is the more complete restore, so it
  takes precedence when both apply.

A checkpoint whose saved config does not match the yaml is **refused, not loaded** — PEFT treats a
saved adapter's own config as authoritative, so loading a mismatched one would quietly train
different layers than the config asks for:

```
ValueError: .../adapter adapts layers [23], but this config asks for [10]. The saved adapter's
own config wins when it is loaded, so continuing from it would not train the layers configured
here. Align layer_lora.layers with the checkpoint, or drop init_from_checkpoint to start a
fresh adapter.
```

TinyLoRA checkpoints (`outputs/sft-ds-assistant/…`) are a different PEFT method and cannot be
resumed here; that is reported the same way.

## Shell scripts

The `*.sh` files at the repo root are thin wrappers over the CLI above. Each one resolves the
boring parts -- installing dependencies, downloading the dataset, finding the newest checkpoint --
prints what it resolved, then execs `poetry run …`. They take **no positional arguments**:
everything is set through environment variables, e.g. `CHECKPOINT=3200 bash eval.sh`.

| Script | What it does | Runs |
|---|---|---|
| `install.sh` | Clone, install, fetch the dataset, start a TinyLoRA SFT run | `tiny-lora sft` |
| `layer_lora_install.sh` | Same preamble, but trains a layer-scoped LoRA adapter | `layer_lora sft` |
| `eval.sh` | Resolve a checkpoint of a run and score it against the base model | `tiny-lora eval` |
| `chat.sh` | Terminal chat REPL against a trained adapter | `tiny-lora chat` |
| `web.sh` | Browser chat UI against a trained adapter | `tiny-lora serve` |
| `upload_to_hf.sh` | Push a checkpoint + regenerated model card to the Hugging Face Hub | `scripts/push_to_hub.py` |

`data_generate.sh` and `data_generator_code_base.sh` build the training corpora and are documented
in their own file headers.

### `install.sh`: install and train (TinyLoRA)

Clones the repo (or `git pull`s an existing checkout), installs Poetry and the dependencies with
the `gdrive` extra, downloads/extracts the dataset zip named by the config's
`data.gdrive.zip_file_id`, then starts SFT with `--no-quant`. The dataset is fetched *before* the
model loads on purpose, so a bad `zip_file_id` fails in seconds rather than minutes into the run.

```bash
# one-liner on a fresh machine
curl -fsSL https://raw.githubusercontent.com/caglanakpinar/qwen_customized_with_tiny_lora/main/install.sh | bash

# from a copy of the file
bash install.sh

# install everything but stop before training -- it prints the command it skipped
SKIP_TRAIN=1 bash install.sh

# a different config and clone target
CONFIG=configs/sft_ds_assistant_27B.yaml REPO_DIR=~/runs/tinylora bash install.sh
```

| Variable | Default | Description |
|---|---|---|
| `REPO_DIR` | `qwen_customized_with_tiny_lora` | Where to clone to; an existing checkout is pulled, not re-cloned. |
| `CONFIG` | `configs/sft_ds_assistant.yaml` | Training config to run. |
| `SKIP_TRAIN` | unset | `1` installs and prepares data, then stops before training. |

### `layer_lora_install.sh`: install and train (layer-scoped LoRA)

The `layer_lora` counterpart to `install.sh` -- same clone/install/dataset preamble, but it ends in
`poetry run layer_lora sft`. Run from inside an existing checkout it detects that (by
`pyproject.toml`'s `name = "tiny-lora"`) and stays put rather than cloning a second copy underneath
itself.

```bash
# defaults: layer 21 alone, configs/sft_layer_lora.yaml
bash layer_lora_install.sh

# the last four layers, capped run
LAYERS=20-23 MAX_STEPS=2000 bash layer_lora_install.sh

# widen the adapter to the attention output projection too
LAYERS=23 TARGET_MODULES="q_proj,k_proj,v_proj,o_proj" bash layer_lora_install.sh

# continue from a finished adapter, into a fresh output dir
INIT_CHECKPOINT=outputs/sft-layer-lora/adapter OUTPUT_DIR=outputs/sft-layer-lora-v2 \
  bash layer_lora_install.sh
```

| Variable | Default | Description |
|---|---|---|
| `REPO_DIR` | `qwen_customized_with_tiny_lora` | Clone target; ignored when run from inside a checkout. |
| `CONFIG` | `configs/sft_layer_lora.yaml` | Training config to run. |
| `LAYERS` | `21` | Layers to adapt -- `21`, `20-23`, `0-3,11,20-23`. |
| `TARGET_MODULES` | the config's list | Comma-separated projections adapted within those layers. |
| `INIT_CHECKPOINT` | unset | Continue from a saved adapter -- a path, or `auto`. |
| `OUTPUT_DIR` | the config's `output_dir` | Where checkpoints are written. |
| `MAX_STEPS` / `LEARNING_RATE` / `MAX_SAMPLES` | the config's values | The usual training overrides. |
| `SKIP_TRAIN` | unset | `1` installs and prepares data, then stops before training. |

> **`INIT_CHECKPOINT` pointing inside `OUTPUT_DIR`** is supported, but set the config's
> `save_total_limit` to `null` first -- otherwise checkpoint rotation would delete the weights the
> run started from, and `layer_lora` refuses to start.

### `eval.sh`: score a checkpoint against the base model

Resolves a checkpoint, checks it really is an adapter dir (`adapter_config.json`), installs
dependencies, then runs `tiny-lora eval`. Resolution order, most explicit first:

1. `ADAPTER` -- a directory, used as given.
2. `CHECKPOINT=N` -- `<output_dir>/checkpoint-N`. This is **not** a fallback: a missing checkpoint
   is an error listing what is available, never a silent slide onto `adapter/`.
3. Neither set -- `<output_dir>/adapter` if the run finished, else the newest `checkpoint-N/`.

`<output_dir>` is read out of `CONFIG` itself, so the checkpoints, the eval split and the saved
results all come from the same run.

```bash
bash eval.sh                                             # newest checkpoint (or final adapter)
CHECKPOINT=5000 bash eval.sh                             # outputs/sft-ds-assistant/checkpoint-5000
MAX_EVAL_SAMPLES=200 GENERATION_SAMPLES=0 bash eval.sh   # quick loss/perplexity-only pass
NO_SAVE=1 bash eval.sh                                   # print without writing the JSON record
```

| Variable | Default | Description |
|---|---|---|
| `CONFIG` | `configs/sft_ds_assistant.yaml` | Config supplying the eval split and the run's `output_dir`. |
| `CHECKPOINT` | unset | Checkpoint number to evaluate, resolved under `output_dir`. |
| `ADAPTER` | unset | Adapter/checkpoint dir outright; wins over `CHECKPOINT`. |
| `OUTPUT_DIR` | `training.output_dir` from `CONFIG` | Run dir holding `adapter/` and `checkpoint-N/`. |
| `MAX_EVAL_SAMPLES` | the config's cap | Cap the eval split to this many rows. |
| `GENERATION_SAMPLES` | `30` | Examples also scored by generating (ROUGE-L / token-F1 / code-valid-rate); `0` skips them. |
| `MAX_NEW_TOKENS` | `256` | Tokens per generated reply. |
| `EVALS_DIR` | `<adapter>/evals/` | Where the JSON record is written. |
| `NO_SAVE` | unset | `1` prints the results without writing them to disk. |
| `SKIP_INSTALL` | unset | `1` uses whatever is already installed instead of running `poetry install`. |

#### Evaluating a `layer_lora` run

`layer_lora` trains into its own `output_dir` (`outputs/sft-layer-lora`, not
`outputs/sft-ds-assistant`), so evaluating one is a matter of pointing `CONFIG` at its config --
the run dir, the checkpoint list and the eval split all follow from that one file:

```bash
# newest checkpoint under outputs/sft-layer-lora (or its adapter/ once the run finishes)
CONFIG=configs/sft_layer_lora.yaml bash eval.sh

# a specific step -> outputs/sft-layer-lora/checkpoint-3200
CONFIG=configs/sft_layer_lora.yaml CHECKPOINT=3200 bash eval.sh

# cheap pass: cap the split, skip the generation metrics, write nothing
CONFIG=configs/sft_layer_lora.yaml CHECKPOINT=3200 \
  MAX_EVAL_SAMPLES=200 GENERATION_SAMPLES=0 NO_SAVE=1 bash eval.sh

# an adapter dir from anywhere -- CONFIG is still what supplies the eval split
CONFIG=configs/sft_layer_lora.yaml \
  ADAPTER=outputs/sft-layer-lora/checkpoint-2750 bash eval.sh

# deps already installed, results collected outside the checkpoint
CONFIG=configs/sft_layer_lora.yaml CHECKPOINT=2750 \
  SKIP_INSTALL=1 EVALS_DIR=outputs/evals/layer-lora bash eval.sh
```

The script always calls `tiny-lora eval`, for TinyLoRA and `layer_lora` checkpoints alike -- there
is no `layer_lora eval`, and none is needed: `layer_lora` writes a standard PEFT adapter, and the
base model id is read out of its `adapter_config.json` exactly the same way.

### `chat.sh` / `web.sh`: talk to a trained adapter

Both default to `outputs/sft-ds-assistant/adapter`, falling back to the newest `checkpoint-N/`
under the same directory while a run is still in progress. `--no-quant` is on by default (4-bit
loading is Linux/CUDA-only), and `--db-path` is passed only when that directory actually exists,
so retrieval turns itself off rather than erroring when the knowledge base has not been built.

```bash
bash chat.sh                                          # terminal REPL
bash web.sh                                           # browser UI on http://127.0.0.1:8000

# a layer_lora checkpoint instead of the default TinyLoRA run
ADAPTER=outputs/sft-layer-lora/checkpoint-3200 bash chat.sh

# expose the UI on the LAN, with a persona
HOST=0.0.0.0 PORT=8080 SYSTEM="You are a senior data scientist." bash web.sh
```

| Variable | Default | Description |
|---|---|---|
| `ADAPTER` | `outputs/sft-ds-assistant/adapter`, else the newest `checkpoint-N/` | Adapter or checkpoint dir to load. |
| `MODEL` | read from the adapter | Base model override. |
| `DB_PATH` | `data/data_science_dbs` | Knowledge base for retrieval; skipped when the dir is absent. |
| `NO_QUANT` | `1` | `1` loads in bf16; any other value re-enables 4-bit quantization. |
| `SYSTEM` | unset | System prompt prepended to the conversation. |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | `web.sh` only -- where the UI is served. |

### `upload_to_hf.sh`: publish a checkpoint

Interactive. It clears any stale `HF_TOKEN`/login, prompts for a **write** token and verifies it,
collects release notes (empty line to finish; leave blank for the defaults), then uploads the
checkpoint and a regenerated model card in one commit via
[`scripts/push_to_hub.py`](scripts/push_to_hub.py).

```bash
bash upload_to_hf.sh

# a different checkpoint and repo, tagged as a release
REPO_ID=your-user/your-adapter \
  CHECKPOINT_DIR=outputs/sft-layer-lora/checkpoint-3200 \
  TAG=v0.3 bash upload_to_hf.sh
```

| Variable | Default | Description |
|---|---|---|
| `REPO_ID` | `Caglana/qwen0.5b-tinylora-ds-assistant` | Target Hub repo. |
| `CHECKPOINT_DIR` | `outputs/sft-ds-assistant/checkpoint-12000` | Local checkpoint to push. |
| `TAG` | unset | Tag/revision name created for this commit. |

## Benchmark Results

Five adapter runs evaluated against the base model, raw numbers in
[`outputs/eval_results.json`](outputs/eval_results.json). The three groups do **not** share an eval
split: run 4 (`checkpoint-3200`) was scored after a newer generated dataset version — weighted
toward code-generation tasks — was added, with different `num_eval_samples`/generation settings too
(500 rows for run 4, an unrecorded count for 1-3), and run 5 (`checkpoint-7500`, continued from
`checkpoint-3200`) was scored against a rebuilt split again. The base model is frozen, so a base
row that moves is proof the data moved. Each group is scored against its own base row — compare
within a group, not across:

| Run | eval_loss | perplexity | rouge_l_f1 | token_f1 | code_valid_rate |
|---|---|---|---|---|---|
| base (runs 1-3) | 2.5823 | 13.2276 | 0.1086 | 0.2163 | 0.0385 |
| 1. tiny_lora `checkpoint-5750` (run 1) | 1.8519 | 6.3720 | 0.0988 | 0.1629 | 0.2308 |
| 2. tiny_lora `checkpoint-5750` (run 2) | **1.8114** | **6.1187** | 0.1183 | 0.1997 | 0.6538 |
| 3. layer_lora `checkpoint-2750` | **1.2613** | **3.5300** | **0.1159** | **0.2284** | 0.1538 |
| base (run 4) | 2.4269 | 11.3240 | 0.0755 | 0.1781 | 0.2667 |
| 4. layer_lora `checkpoint-3200` | **1.4074** | **4.0852** | 0.0768 | 0.1802 | 0.0333 |
| base (run 5) | 2.6143 | 13.6570 | 0.1109 | 0.2230 | 0.1154 |
| 5. layer_lora `checkpoint-7500` | **1.3067** | **3.6939** | **0.1184** | **0.2361** | 0.1923 |

### Interpretation

- **Teacher-forced loss/perplexity favor layer_lora.** All three layer_lora checkpoints roughly
  halve to a third of the base model's eval loss (2.58 → 1.26 at step 2750, 2.43 → 1.41 at step
  3200, 2.61 → 1.31 at step 7500) and clear both tiny_lora runs by a wide margin. Adapting full LoRA matrices on the last few transformer
  layers gives the model more effective capacity than TinyLoRA's shared low-dimensional `v` vector,
  and that shows up directly in next-token prediction.
- **layer_lora's loss ticked back up between step 2750 and step 3200** (1.26 → 1.41) even though 3200
  is the later checkpoint — read this alongside the caveat above, since the two evals used different
  `num_eval_samples`, not as confirmed regression-with-more-training on its own.
- **`code_valid_rate` at checkpoint-3200 drops well below its own base (0.033 vs 0.267)** — the
  checkpoint produces parseable Python less often than the un-adapted model, on the same eval call.
  That's the opposite direction from every other run here, where the checkpoint's code-valid rate
  rose over its base. Part of this is a harder yardstick, not just the checkpoint: this eval ran
  against the newer, code-generation-weighted dataset version, so the base model's own code_valid_rate
  jumped too (0.267 here vs 0.039 for runs 1-3) — the new split has more/harder code tasks to get
  right, base included. That said, only the *checkpoint* got worse relative to its base, so it isn't
  purely the harder split; combined with the loss upturn, step 3200 still looks like it has started
  overfitting away from code-formatting conventions past step 2750. `checkpoint-2750` remains the
  better layer_lora checkpoint to ship, but a same-split, same-conditions re-eval of both checkpoints
  against the new code-generation dataset would confirm it rather than mixing old- and new-split
  numbers as done here. **Run 5 (step 7500) no longer shows that collapse** — though on a rebuilt
  split again, so it is not the controlled re-eval this bullet asked for; see below.
- **tiny_lora's two runs at the identical checkpoint disagree sharply on `code_valid_rate`** (0.23 vs
  0.65) while eval_loss/perplexity barely move (1.85 vs 1.81) and ROUGE-L/token-F1 are close. Since
  generation is greedy (deterministic) and both runs read the same `checkpoint-5750` weights, the gap
  is almost certainly in the harness around the run, not the model itself — e.g. a different
  `--max-eval-samples`/generation-sample count, an eval-split or prompt-formatting change between
  runs, or a code-extraction/parsing tweak in `eval.py` between when the two were captured. Treat
  run 2's `code_valid_rate` as the more reliable of the pair only if you can confirm it was captured
  after such a fix — otherwise the two rows are evidence the metric is noisy at this sample size
  rather than evidence the checkpoint improved.
- **ROUGE-L / token-F1 barely separate any of the runs** (0.08–0.12 and 0.16–0.24
  respectively) — all are still far from fluent instruction-following at this model size and
  training budget, so free-generation text overlap is a weak discriminator here compared to
  teacher-forced loss.
- **Run 5 moved two variables at once, so its gain is not yet attributable.** `checkpoint-7500`
  continues training from `checkpoint-3200`, *and* its base row moved (2.4269 → 2.6143), so the
  split changed too. Base-relative, the perplexity cut goes from 64% (11.32 → 4.09) to 73%
  (13.66 → 3.69) and ROUGE-L/token-F1 from roughly flat (+0.001/+0.002) to slightly positive
  (+0.008/+0.013). Directionally the best layer_lora result recorded — but how much is the longer
  run (step 7500 at epoch 4.8, against step 3200 at epoch 2.0) and how much is the yardstick cannot
  be separated from these rows. Worth knowing from its own `trainer_state`: `best_metric` 1.1515
  lands at this very step with the early-stopping counter at 0, so against a `max_steps` of 232,000
  the run had not plateaued when this checkpoint was written.
- **The `code_valid_rate` collapse at step 3200 did not persist at 7500.** It scores 0.1154 → 0.1923,
  i.e. *above* its base rather than far below it — the opposite sign from run 4, and the sign every
  other run shows. Encouraging, and weaker evidence than it looks on two counts: the base model's own
  code-valid rate fell from 0.267 to 0.115 across the two splits, and the rates are 3/26 and 5/26
  correct, whose Wilson 95% intervals ([4.0%, 29.0%] and [8.5%, 37.9%]) overlap almost entirely.
  A two-example difference cannot carry the claim in either direction — which also means step 3200's
  dramatic drop was never solid evidence of overfitting. The 100-case smoke test
  ([`scripts/test_hf_model.py`](scripts/test_hf_model.py)) is the better instrument: 80 code prompts
  against the published adapter returned 24% valid code, tighter than anything a 26-sample rate can
  say and consistent with run 5's 19.2%.
- **Watch the base row for split drift.** Between runs 4 and 5 the base model got *worse*
  (2.4269 → 2.6143) while the adapter got *better* (1.4074 → 1.3067). A gap widening from both ends
  is what an eval split drifting toward the fine-tune's own distribution looks like. If a rebuilt
  `sft_eval.jsonl` was generated from a corpus whose generators changed, part of that 73% is the
  yardstick moving, not the model reaching further — so treat cross-split perplexity cuts as
  directional, and re-derive a baseline whenever the split is rebuilt.
- **What would settle it**, neither step requiring a retrain: score `checkpoint-7500` on run 4's
  split (or `checkpoint-3200` on run 5's), which puts one ruler under both and separates the extra
  training from the split; and re-run the 100-case smoke test against 7500, where 80 code prompts
  can support a claim about code validity that 26 cannot.
- **Net takeaway:** for this dataset and model size, restricting a full-rank LoRA to a handful of
  late transformer layers (`layer_lora`) recovered more quality per training step than TinyLoRA's
  extreme parameter budget did in these runs, and step 7500 is the strongest of the three on its
  own base row. `code_valid_rate` looked non-monotonic with more steps, then reversed sign again at
  7500, so at 26 generation samples that metric is too noisy to select checkpoints on — select on
  eval_loss, and confirm code behaviour on the 100-case smoke test rather than on the eval's
  code-valid rate. This is a
  specific-to-this-setup result, not a general claim about TinyLoRA — see the
  [TinyLoRA paper](https://arxiv.org/abs/2602.04118) for the regime (larger models, GRPO/RL) where
  its parameter efficiency is shown to pay off.

## Project Structure

```
llm_with_tiny_lora/
├── pyproject.toml          # Poetry dependencies
├── install.sh              # clone + install + start a TinyLoRA SFT run -- see "Shell scripts"
├── layer_lora_install.sh   # the same, ending in a layer-scoped LoRA run
├── eval.sh                 # base-vs-checkpoint eval for any run (CONFIG picks which)
├── chat.sh / web.sh        # terminal REPL / browser UI against a trained adapter
├── upload_to_hf.sh         # push a checkpoint + model card to the Hugging Face Hub
├── data_generate.sh        # build the synthetic data-science set
├── data_generator_code_base.sh  # build the Kaggle-grounded code corpus
├── scripts/
│   └── push_to_hub.py          # Hub upload + model-card generation, driven by upload_to_hf.sh
├── configs/
│   ├── sft_default.yaml        # SFT defaults (gsm8k)
│   ├── grpo_default.yaml       # GRPO defaults (gsm8k)
│   ├── sft_ds_assistant.yaml   # SFT on the synthetic data-science set
│   └── sft_layer_lora.yaml     # Layer-scoped LoRA on the same set
├── outputs/                 # Checkpoints + saved adapters, written by `sft`/`grpo` (per output_dir)
│   └── sft-ds-assistant/
│       ├── checkpoint-N/        # periodic checkpoint, written every training.save_steps
│       └── adapter/             # final adapter, written once training completes
├── web/                     # Browser chat UI, served by `tiny-lora serve`
│   ├── app.py                  # FastAPI backend (/, /api/chat, /api/reset)
│   └── static/                  # HTML/CSS/JS chat frontend, no build step
├── mobile/                  # Android client for the same /api/chat routes (see mobile/README.md)
├── src/layer_lora/          # Layer-scoped LoRA (`poetry run layer_lora sft`)
│   ├── cli.py                  # Click CLI entry point
│   ├── config.py               # LayerLoraConfig (`layer_lora:` yaml section)
│   ├── model.py                # Layer-restricted adapter build + checkpoint reuse
│   └── train_sft.py            # SFT pipeline, reusing tiny_lora's run_sft_core
└── src/tiny_lora/
    ├── cli.py              # Click CLI entry point
    ├── config.py           # Config dataclasses & YAML loader
    ├── model.py            # Model loading + TinyLoRA adapters
    ├── data.py             # Dataset preparation (gsm8k, chat JSONL, shard globs)
    ├── rewards.py          # GRPO reward functions
    ├── train_sft.py        # SFT pipeline
    ├── train_grpo.py       # GRPO pipeline
    ├── eval.py             # Base-model vs checkpoint eval loss/perplexity
    ├── chat.py             # Chat session logic (ChatSession) + CLI REPL, shared with web/app.py
    ├── chat_memory.py      # Persists chat summaries to FAISS/Chroma, embedded via the chat model
    ├── knowledge_db.py     # Queries the offline FAISS/Chroma knowledge stores for retrieval
    └── chat_api.py         # KServe payload/response inference API (`chat-api`), shared ChatSession
```

## Configuration

Edit YAML files under `configs/` or pass overrides via CLI flags:

```yaml
tinylora:
  r: 2              # SVD rank (paper recommends 2)
  u: 32             # trainable vector dimension
  weight_tying: 1.0 # 1.0 = single shared v across all modules
  target_modules: [q_proj, v_proj]

model:
  model_name_or_path: "Qwen/Qwen2.5-0.5B-Instruct"
  load_in_4bit: true

data:
  dataset_name: "openai/gsm8k"
  max_samples: 500
```

`dataset_name` accepts a Hugging Face repo id, a local `.jsonl` path, or a shard glob. A local
file with a `messages` column is rendered through the model's chat template automatically:

```yaml
data:
  dataset_name: "data/synthetic/dataset/sft_train-*.jsonl"
  eval_dataset_name: "data/synthetic/dataset/sft_eval.jsonl"   # enables eval loss during SFT
  max_samples: 50000
  max_eval_samples: 200   # cap the eval split; unset scores it in full (used by SFT and `eval`)
```

`configs/sft_ds_assistant.yaml` trains on a synthetically generated data-science assistant corpus —
concept Q&A plus runnable pandas/matplotlib/scikit-learn/statistics/SQL tasks. Rather than shipping
that dataset in the repo, `data.reader: "gdrive"` points at a zip on Google Drive by its file id and
downloads it on first use:

```yaml
data:
  reader: "gdrive"
  dataset_name: "data/synthetic/dataset/sft_train-*.jsonl"
  eval_dataset_name: "data/synthetic/dataset/sft_eval.jsonl"
  gdrive:
    zip_file_id: 1d9sIZg95DlBDPSyvWfIGpHtfIcHpYoZB   # share the zip "anyone with the link", paste its id here
    cache_dir: "data/synthetic/dataset"               # extracted here once, then reused on later runs
```

Requires the `gdrive` extra (`poetry install -E gdrive`). `dataset_name`/`eval_dataset_name` still
point at the paths the zip extracts to — once downloaded, it's read exactly like a local dataset.

## References

- [Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118) — TinyLoRA paper
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoRA-XS](https://arxiv.org/abs/2405.17604)
- [Hugging Face PEFT — TinyLoRA docs](https://huggingface.co/docs/peft/main/en/package_reference/tinylora)

## License

TBD
