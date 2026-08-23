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

## Project Structure

```
llm_with_tiny_lora/
├── pyproject.toml          # Poetry dependencies
├── configs/
│   ├── sft_default.yaml        # SFT defaults (gsm8k)
│   ├── grpo_default.yaml       # GRPO defaults (gsm8k)
│   └── sft_ds_assistant.yaml   # SFT on the synthetic data-science set
├── outputs/                 # Checkpoints + saved adapters, written by `sft`/`grpo` (per output_dir)
│   └── sft-ds-assistant/
│       ├── checkpoint-N/        # periodic checkpoint, written every training.save_steps
│       └── adapter/             # final adapter, written once training completes
├── web/                     # Browser chat UI, served by `tiny-lora serve`
│   ├── app.py                  # FastAPI backend (/, /api/chat, /api/reset)
│   └── static/                  # HTML/CSS/JS chat frontend, no build step
├── mobile/                  # Android client for the same /api/chat routes (see mobile/README.md)
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
