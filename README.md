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
```

### Commands

| Command | Description |
|---------|-------------|
| `sft` | Supervised fine-tuning with TinyLoRA |
| `grpo` | GRPO RL training with TinyLoRA |
| `info` | Print config and trainable parameter count |
| `show-config` | Display a YAML config as JSON |

## Project Structure

```
llm_with_tiny_lora/
├── pyproject.toml          # Poetry dependencies
├── configs/
│   ├── sft_default.yaml        # SFT defaults (gsm8k)
│   ├── grpo_default.yaml       # GRPO defaults (gsm8k)
│   └── sft_ds_assistant.yaml   # SFT on the synthetic data-science set
├── data/                   # Knowledge stores + synthetic dataset — see data/README.md
│   ├── data_engineer_dbs/       68 concepts  → Chroma + FAISS
│   ├── feature_engineering_dbs/ 43 concepts  → Chroma + FAISS
│   ├── data_science_dbs/        52 concepts  → Chroma + FAISS
│   └── synthetic/               the dataset generator, and its output
└── src/tiny_lora/
    ├── cli.py              # Click CLI entry point
    ├── config.py           # Config dataclasses & YAML loader
    ├── model.py            # Model loading + TinyLoRA adapters
    ├── data.py             # Dataset preparation (gsm8k, chat JSONL, shard globs)
    ├── rewards.py          # GRPO reward functions
    ├── train_sft.py        # SFT pipeline
    └── train_grpo.py       # GRPO pipeline
```

## Data-Science Assistant Dataset

`data/` holds three ChromaDB knowledge stores — data engineering, feature engineering and data
science, **163 hand-written concepts across 36 topics** — and a generator that turns them into an
instruction-tuning set for a data-science assistant: concept Q&A plus runnable pandas, matplotlib,
scikit-learn, statistics and SQL tasks.

```bash
poetry install --with data          # chromadb, faiss-cpu, numpy

python -m data.data_science_dbs.dataset   # rebuild the data-science store
python -m data.synthetic.build            # generate the training set (5 GB by default)
python -m data.synthetic.build --target-gb 0.05   # or a small one to look at

poetry run tiny-lora sft --config configs/sft_ds_assistant.yaml --no-quant
```

Generation is deterministic and offline — no API key, no model in the loop. Answers are assembled
from facet text read back out of Chroma, so changing a knowledge store changes the dataset with no
code change. See [data/README.md](data/README.md) for how the stores are shaped, where the variety
comes from, and how the generated code is verified.

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
```

## References

- [Learning to Reason in 13 Parameters](https://arxiv.org/abs/2602.04118) — TinyLoRA paper
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [LoRA-XS](https://arxiv.org/abs/2405.17604)
- [Hugging Face PEFT — TinyLoRA docs](https://huggingface.co/docs/peft/main/en/package_reference/tinylora)

## License

TBD
