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

## Benchmark Results

Three adapter runs evaluated against the same base model and eval split
raw numbers in [`outputs/eval_results.json`](outputs/eval_results.json):

| Run | eval_loss | perplexity | rouge_l_f1 | token_f1 | code_valid_rate |
|---|---|---|---|---|---|
| base (all three rows below) | 2.5823 | 13.2276 | 0.1086 | 0.2163 | 0.0385 |
| 1. tiny_lora `checkpoint-5750` (run 1) | 1.8519 | 6.3720 | 0.0988 | 0.1629 | 0.2308 |
| 2. tiny_lora `checkpoint-5750` (run 2) | **1.8114** | **6.1187** | 0.1183 | 0.1997 | 0.6538 |
| 3. layer_lora `checkpoint-2750` | **1.2613** | **3.5300** | **0.1159** | **0.2284** | 0.1538 |

### Interpretation

- **Teacher-forced loss/perplexity favor layer_lora.** `layer_lora` at step 2750 roughly halves the
  base model's eval loss (2.58 → 1.26, perplexity 13.2 → 3.5) and clears both tiny_lora runs by a
  wide margin, despite training for fewer steps. Adapting full LoRA matrices on the last few
  transformer layers gives the model more effective capacity than TinyLoRA's shared low-dimensional
  `v` vector, and that shows up directly in next-token prediction.
- **tiny_lora's two runs at the identical checkpoint disagree sharply on `code_valid_rate`** (0.23 vs
  0.65) while eval_loss/perplexity barely move (1.85 vs 1.81) and ROUGE-L/token-F1 are close. Since
  generation is greedy (deterministic) and both runs read the same `checkpoint-5750` weights, the gap
  is almost certainly in the harness around the run, not the model itself — e.g. a different
  `--max-eval-samples`/generation-sample count, an eval-split or prompt-formatting change between
  runs, or a code-extraction/parsing tweak in `eval.py` between when the two were captured. Treat
  run 2's `code_valid_rate` as the more reliable of the pair only if you can confirm it was captured
  after such a fix — otherwise the two rows are evidence the metric is noisy at this sample size
  rather than evidence the checkpoint improved.
- **ROUGE-L / token-F1 barely separate the three checkpoints** (0.10–0.12 and 0.16–0.23
  respectively) — all three are still far from fluent instruction-following at this model size and
  training budget, so free-generation text overlap is a weak discriminator here compared to
  teacher-forced loss.
- **Net takeaway:** for this dataset and model size, restricting a full-rank LoRA to a handful of
  late transformer layers (`layer_lora`) recovered more quality per training step than TinyLoRA's
  extreme parameter budget did in these runs. That is a specific-to-this-setup result, not a general
  claim about TinyLoRA — see the [TinyLoRA paper](https://arxiv.org/abs/2602.04118) for the regime
  (larger models, GRPO/RL) where its parameter efficiency is shown to pay off.

## Project Structure

```
llm_with_tiny_lora/
├── pyproject.toml          # Poetry dependencies
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

## Open-source LLMs

Nothing in the pipeline is Qwen-specific. The base model goes through `AutoModelForCausalLM`, so
any causal LM on the Hub works — swap one line:

```yaml
model:
  model_name_or_path: "meta-llama/Llama-3.2-3B-Instruct"
```

Four things have to line up:

| Requirement | Why | Applies to |
|---|---|---|
| ships a **chat template** | the training set is `messages` conversations, rendered by `tokenizer.apply_chat_template` | every run |
| reports `num_hidden_layers` | layer indices are validated before any weights load | `layer_lora` |
| `target_modules` names match the architecture | PEFT matches modules by name — a name nothing matches trains nothing, silently | every run |
| `layers_pattern` matches the layer container | it is the path segment before the layer index | `layer_lora` |

The chat template is the one that catches people out: use the **`-Instruct` / `-it` / `-chat`**
variant. Base (pretrain) checkpoints usually ship no template, and `apply_chat_template` raises.

### Drop-in

Llama-style naming — `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` with
`layers_pattern: layers`. Nothing to change but `model_name_or_path`.

| Family | Example ids | Notes |
|---|---|---|
| Qwen 2.5 / 3 | `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B` | what this repo ships with |
| Llama 3.x | `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | gated — accept the licence, then `hf auth login` |
| Gemma 2 / 3 | `google/gemma-2-2b-it`, `google/gemma-3-4b-it` | gated |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | |
| SmolLM2 | `HuggingFaceTB/SmolLM2-360M-Instruct`, `HuggingFaceTB/SmolLM2-1.7B-Instruct` | smallest sensible swap; 360M trains on a laptop |
| OLMo 2 | `allenai/OLMo-2-1124-7B-Instruct` | fully open weights *and* data |
| Granite 3 | `ibm-granite/granite-3.1-2b-instruct` | |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | |
| DeepSeek-R1 distills | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | Qwen2/Llama architecture underneath |

### Needs different names

| Family | `target_modules` | `layers_pattern` | Why |
|---|---|---|---|
| Phi-3 / 3.5 | `qkv_proj, o_proj, gate_up_proj, down_proj` | `layers` | QKV and gate/up are **fused** — `q_proj` matches nothing |
| StarCoder2 | `q_proj, k_proj, v_proj, o_proj, c_fc, c_proj` | `layers` | GPT-style MLP names |
| Mixtral | `q_proj, k_proj, v_proj, o_proj` | `layers` | attention only; the MoE experts are not separate `Linear`s |
| Falcon | `query_key_value, dense, dense_h_to_4h, dense_4h_to_h` | `h` | fused QKV, and the stack is `transformer.h.N` |
| GPT-NeoX | `query_key_value, dense, dense_h_to_4h, dense_4h_to_h` | `layers` | fused QKV |
| MPT | `Wqkv, out_proj, up_proj, down_proj` | `blocks` | the stack is `transformer.blocks.N` |

### Reading the names off a model

For anything not listed, ask the model directly rather than guessing:

```bash
poetry run python -c "
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
mid = 'Qwen/Qwen2.5-0.5B-Instruct'
print('arch:', AutoConfig.from_pretrained(mid).model_type,
      '| layers:', AutoConfig.from_pretrained(mid).num_hidden_layers)
m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype='auto')
print('target_modules:', sorted({n.rsplit(\".\",1)[-1] for n, x in m.named_modules()
                                 if isinstance(x, torch.nn.Linear)} - {'lm_head'}))
print('layer 0 path:', next(n for n, _ in m.named_modules() if n.endswith('.0')))
print('chat template:', 'yes' if AutoTokenizer.from_pretrained(mid).chat_template else 'NO')
"
```

```
arch: qwen2 | layers: 24
target_modules: ['down_proj', 'gate_proj', 'k_proj', 'o_proj', 'q_proj', 'up_proj', 'v_proj']
layer 0 path: model.layers.0        # -> layers_pattern: "layers"
chat template: yes
```

### What does not carry over

**Adapters are tied to the base model they were trained on.** `outputs/sft-ds-assistant/`'s
checkpoints only load against `Qwen/Qwen2.5-0.5B-Instruct` — the shapes, the layer count and, for
TinyLoRA, the SVD of each frozen weight all come from that model. Switching models means training
from scratch, not re-pointing `init_from_checkpoint`. Give the new run its own `output_dir`.

Two smaller consequences:

- **TinyLoRA takes an SVD of every target module at load time.** It is a one-time cost, but on a
  7B model with seven target modules it is minutes, not seconds.
- `chat`, `serve` and `eval` need no change — they read the base model id out of the adapter's own
  `adapter_config.json`.

## References

- [Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118) — TinyLoRA paper
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoRA-XS](https://arxiv.org/abs/2405.17604)
- [Hugging Face PEFT — TinyLoRA docs](https://huggingface.co/docs/peft/main/en/package_reference/tinylora)

## License

TBD
