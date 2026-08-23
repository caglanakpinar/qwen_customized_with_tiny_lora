"""Dataset loading and formatting for SFT and GRPO.

Three dataset shapes are supported:

    chat       a `messages` column of role/content dicts -- what `data/synthetic/` writes
    text       a pre-rendered `text` column, used as-is
    gsm8k      the `question`/`answer` columns of openai/gsm8k

`dataset_name` may be a Hugging Face repo id, a local `.json`/`.jsonl` path, or a shard glob --
which is how the synthetic training set built from the knowledge stores is consumed:

    data:
      dataset_name: "data/synthetic/dataset/sft_train-*.jsonl"
      eval_dataset_name: "data/synthetic/dataset/sft_eval.jsonl"
      max_samples: 50000

With `max_samples` set, only as many shards as that needs are read -- the full set is several
gigabytes and `load_dataset` would otherwise convert all of it to Arrow before selecting anything.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

from tiny_lora.config import DataConfig

LOCAL_SUFFIXES = {".json", ".jsonl"}

# Columns carried for provenance that must not reach the trainer.
METADATA_COLUMNS = ("meta",)


def _is_local_file(name: str) -> bool:
    return Path(name).suffix in LOCAL_SUFFIXES


def _resolve_data_files(name: str) -> list[str]:
    """Expand a path or a shard glob into the concrete files to read.

    A multi-gigabyte set is written as `sft_train-00000-of-00021.jsonl` and friends, so configs
    point at `sft_train-*.jsonl`. Expanding here — rather than handing the pattern to
    `load_dataset` — means a typo that matches nothing fails immediately with the pattern in the
    message, instead of loading an empty dataset.
    """
    path = Path(name)
    if not path.is_absolute():
        path = Path.cwd() / path

    if any(char in name for char in "*?["):
        matches = sorted(str(match) for match in path.parent.glob(path.name))
        if not matches:
            raise FileNotFoundError(
                f"No files match {path}. Build the dataset with "
                "`python -m data.synthetic.build`."
            )
        return matches

    if not path.exists():
        raise FileNotFoundError(
            f"Local dataset not found: {path}. "
            "Build it with `python -m data.synthetic.build`."
        )
    return [str(path)]


def _load_until(data_files: list[str], max_samples: int) -> Dataset:
    """Read shards in order until `max_samples` rows are available, then truncate."""
    loaded: Dataset | None = None
    for path in data_files:
        shard = load_dataset("json", data_files=path, split="train")
        loaded = shard if loaded is None else concatenate_datasets([loaded, shard])
        if len(loaded) >= max_samples:
            break
    assert loaded is not None  # _resolve_data_files never returns an empty list
    if len(loaded) < max_samples:
        raise ValueError(
            f"max_samples={max_samples} exceeds the {len(loaded)} rows across "
            f"{len(data_files)} shard(s)."
        )
    return loaded.select(range(max_samples))


def ensure_gdrive_dataset(cache_dir: Path, zip_file_id: str | None) -> Path:
    """Download and extract the dataset zip from Google Drive into `cache_dir`.

    Skipped once the cache dir is populated, so this only pays the download cost once --
    later runs (and the eval split, loaded through the same call) read the extracted files
    exactly like a local dataset. Returns the cache dir either way.

    `data/synthetic/build.py` calls this too, so a build configured to append starts from the
    same bytes the trainer would have read. Keeping one implementation matters because the
    "already populated, leave it alone" rule is the thing both sides have to agree on: if the
    builder re-downloaded over a dataset it was about to append to, the append would be lost.
    """
    cache_dir = Path(cache_dir)
    if cache_dir.is_dir() and any(cache_dir.iterdir()):
        return cache_dir

    if not zip_file_id:
        raise ValueError(
            "data.gdrive.zip_file_id must be set when data.reader is 'gdrive'."
        )

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "Reading from Google Drive requires the 'gdown' package. "
            "Install it with `poetry install -E gdrive`."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "_gdrive_dataset.zip"
    gdown.download(id=zip_file_id, output=str(zip_path), quiet=False)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    zip_path.unlink()
    return cache_dir


def _ensure_gdrive_dataset(data_cfg: DataConfig) -> None:
    """`ensure_gdrive_dataset` driven by a `DataConfig`, and a no-op for any other reader."""
    if data_cfg.reader != "gdrive":
        return
    ensure_gdrive_dataset(Path(data_cfg.gdrive_cache_dir), data_cfg.gdrive_zip_file_id)


def load_raw_dataset(data_cfg: DataConfig, dataset_name: str | None = None) -> Dataset:
    """Load a split from the Hub, or a local JSON/JSONL file.

    `dataset_name` overrides `data_cfg.dataset_name`, which is how the eval split is loaded
    through the same path as the training split. When `data_cfg.reader` is "gdrive", the
    backing zip is downloaded and extracted first so the rest of this function can treat it
    exactly like a local dataset.
    """
    _ensure_gdrive_dataset(data_cfg)
    name = dataset_name or data_cfg.dataset_name

    if _is_local_file(name):
        data_files = _resolve_data_files(name)
        # Read shards one at a time when a sample size is set. `load_dataset` converts every
        # file it is given to Arrow before anything can be selected from it, so pointing at a
        # multi-gigabyte glob to take 50k rows would rewrite the whole set into the cache first.
        if data_cfg.max_samples is not None and len(data_files) > 1:
            return _load_until(data_files, data_cfg.max_samples)
        dataset = load_dataset("json", data_files=data_files, split="train")
    else:
        kwargs: dict = {"path": name, "split": data_cfg.split}
        if data_cfg.dataset_config:
            kwargs["name"] = data_cfg.dataset_config
        dataset = load_dataset(**kwargs)

    if data_cfg.max_samples is not None:
        dataset = dataset.select(range(min(data_cfg.max_samples, len(dataset))))
    return dataset


def _drop_metadata(dataset: Dataset) -> Dataset:
    present = [column for column in METADATA_COLUMNS if column in dataset.column_names]
    return dataset.remove_columns(present) if present else dataset


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_chat_sft(example: dict, tokenizer) -> dict:
    """Render a `messages` conversation through the tokenizer's chat template."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def format_gsm8k_sft(example: dict, tokenizer) -> dict:
    question = example["question"]
    answer = example["answer"]
    messages = [
        {
            "role": "user",
            "content": (
                f"{question}\n"
                "Think step by step and provide the final numeric answer "
                "after '####'."
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def format_gsm8k_grpo(example: dict) -> dict:
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"{example['question']}\n"
                    "Think step by step and provide the final numeric answer "
                    "after '####'."
                ),
            }
        ],
        "answer": example["answer"],
    }


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_sft_dataset(
    data_cfg: DataConfig,
    tokenizer,
    dataset_name: str | None = None,
) -> Dataset:
    """Return a dataset with a single `text` column ready for `SFTTrainer`."""
    dataset = load_raw_dataset(data_cfg, dataset_name=dataset_name)

    if "messages" in dataset.column_names:
        return dataset.map(
            lambda ex: format_chat_sft(ex, tokenizer),
            remove_columns=dataset.column_names,
        )
    if "gsm8k" in (dataset_name or data_cfg.dataset_name):
        return dataset.map(
            lambda ex: format_gsm8k_sft(ex, tokenizer),
            remove_columns=dataset.column_names,
        )
    if "text" in dataset.column_names:
        return _drop_metadata(dataset)
    raise ValueError(
        f"Unsupported dataset for SFT: {dataset_name or data_cfg.dataset_name}. "
        "Provide a dataset with a 'messages' or 'text' column, or use openai/gsm8k."
    )


def prepare_sft_eval_dataset(data_cfg: DataConfig, tokenizer) -> Dataset | None:
    """The eval split, if one is configured."""
    if not data_cfg.eval_dataset_name:
        return None
    dataset = prepare_sft_dataset(data_cfg, tokenizer, dataset_name=data_cfg.eval_dataset_name)
    if data_cfg.max_eval_samples is not None:
        dataset = dataset.select(range(min(data_cfg.max_eval_samples, len(dataset))))
    return dataset


def prepare_grpo_dataset(data_cfg: DataConfig, dataset_name: str | None = None) -> Dataset:
    """Return a dataset with `prompt` and `answer` columns ready for `GRPOTrainer`."""
    dataset = load_raw_dataset(data_cfg, dataset_name=dataset_name)

    if "gsm8k" in (dataset_name or data_cfg.dataset_name):
        return dataset.map(
            format_gsm8k_grpo,
            remove_columns=dataset.column_names,
        )
    if "prompt" in dataset.column_names:
        return _drop_metadata(dataset)
    raise ValueError(
        f"Unsupported dataset for GRPO: {dataset_name or data_cfg.dataset_name}. "
        "Provide a dataset with a 'prompt' column or use openai/gsm8k."
    )
