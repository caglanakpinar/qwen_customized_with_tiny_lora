"""Checkpoint acquisition and introspection -- I/O only, no `transformers` model classes.

`transformers` never ported the Qwen architecture to TensorFlow (there is no `TFQwen*` family),
so `TFAutoModelForCausalLM` cannot load these checkpoints. This module instead fetches the raw
`config.json` + `*.safetensors` files and reads them as plain dicts/arrays; `qwen_model.py`
builds the actual TF graph and assigns these arrays into it by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Every file we might need out of a checkpoint repo -- config for architecture dims, tokenizer
# files for `tiny_lora.model.load_tokenizer`, and weights either as one file or sharded with an
# index. Patterns, not exact names, since sharded weights are named model-00001-of-00004.safetensors.
_ALLOW_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
]


def fetch_checkpoint_files(model_name_or_path: str) -> Path:
    """Return a local directory holding `config.json` and the `*.safetensors` weight file(s).

    A local path is used as-is. A Hub repo id is downloaded (and cached) with
    `snapshot_download`, restricted to just the files this module reads -- skips the vision
    tower / extra modality weights some checkpoints ship alongside the text model, and avoids
    re-downloading files this pipeline never touches.
    """
    local_dir = Path(model_name_or_path)
    if local_dir.is_dir():
        return local_dir

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_name_or_path, allow_patterns=_ALLOW_PATTERNS))


def load_hf_config(checkpoint_dir: Path) -> dict:
    """Read `config.json` -- the architecture dimensions `qwen_model.py` builds the graph from."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {checkpoint_dir}. Point model.model_name_or_path at a Hub repo "
            "id or a local directory holding a Hugging Face checkpoint."
        )
    return json.loads(config_path.read_text())


def load_state_dict(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    """Read every `*.safetensors` file in `checkpoint_dir` into one flat name -> array dict.

    Handles both a single `model.safetensors` and a sharded checkpoint (multiple
    `model-NNNNN-of-MMMMM.safetensors` files) identically -- the index json only maps names to
    shard files for lazy loading, which this pipeline doesn't need since it reads every tensor.
    """
    from safetensors.numpy import load_file

    shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    if not shard_paths:
        raise FileNotFoundError(
            f"No *.safetensors files in {checkpoint_dir}. This pipeline reads safetensors "
            "checkpoints only (the format virtually every current Hub repo publishes)."
        )

    state_dict: dict[str, np.ndarray] = {}
    for shard_path in shard_paths:
        state_dict.update(load_file(str(shard_path)))
    return state_dict


def has_key(state_dict: dict[str, np.ndarray], suffix: str) -> bool:
    """True if any key in `state_dict` ends with `suffix`.

    Used to auto-detect optional architecture pieces (e.g. Qwen3's per-head `q_norm`/`k_norm`,
    which Qwen2 checkpoints don't have) from the checkpoint's own keys instead of hardcoding
    behavior per model family/name.
    """
    return any(key.endswith(suffix) for key in state_dict)
