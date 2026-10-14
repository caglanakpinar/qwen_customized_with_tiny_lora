#!/usr/bin/env python3
"""Helpers for layer_grow_multiple_run.sh -- the parts that are easier in Python than in bash.

The shell script owns the control flow (the layer loop, the dataset loop, logging, resuming); this
file owns everything that needs to read a model, a sidecar or a yaml:

    download   fetch the Hugging Face model into a local directory, once
    layers     read each layer's dimensions out of that local model; print the indices to train
    settings   print the training settings every iteration will use, and warn about risky ones
    prefetch   download + validate one dataset zip into a cache directory of its own
    config     write one iteration's training config, derived from configs/sft_layer.yaml
    paste      put a finished run's weights back into the local model, and record the progress

Results go to stdout (only where the shell needs to capture something), notes to stderr. A problem
the user can act on is reported as one `layer_grow_multi: ...` line and exit status 2, no traceback.

Nothing here imports torch or loads weights: the model is inspected through its safetensors headers
(a few hundred KB of a multi-gigabyte file), so a check costs milliseconds and can run before every
iteration.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path

SIDECAR_NAME = "layer_grow.json"
DIM_KEYS = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")
_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")


class HelperError(Exception):
    """A problem the user can act on. Reported as a single line, without a traceback."""


def note(message: str = "") -> None:
    print(message, file=sys.stderr, flush=True)


def read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise HelperError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text())
    except ValueError as exc:
        raise HelperError(f"{what} is not valid JSON ({path}): {exc}") from exc


# --------------------------------------------------------------------------------------
# Reading a model without loading it
# --------------------------------------------------------------------------------------


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """The tensor table of one .safetensors file, and the byte offset where its data starts.

    The format is an 8-byte little-endian header length followed by that many bytes of JSON, so
    this reads a few hundred KB of a multi-gigabyte file and needs no safetensors or torch import.
    It also checks the file's length against what the header promises: a download cut short fails
    here, naming the file, instead of surfacing as an opaque load error hours into a run.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) < 8:
            raise HelperError(f"{path} is too short to be a safetensors file ({size} bytes).")
        (length,) = struct.unpack("<Q", raw)
        if length > min(size - 8, 100 * 1024 * 1024):
            raise HelperError(
                f"{path} does not look like a safetensors file (its header claims {length:,} bytes)."
            )
        try:
            table = json.loads(handle.read(length))
        except ValueError as exc:
            raise HelperError(f"{path}: unreadable safetensors header ({exc}).") from exc

    table.pop("__metadata__", None)
    data_start = 8 + length
    expected = data_start + max((entry["data_offsets"][1] for entry in table.values()), default=0)
    if expected != size:
        raise HelperError(
            f"{path} is {size:,} bytes but its header describes {expected:,} -- a truncated or "
            "partial file. Delete it and re-run to fetch it again."
        )
    return table, data_start


def weight_files(model_dir: Path) -> list[Path]:
    """The .safetensors file(s) holding a model's weights, found in the order layer_expand's own
    `read_state_dict` looks for them: an index's shards, else model.safetensors, else loose
    *.safetensors files. `adapter_*` files are the TinyLoRA adapter, not the model, so a loose
    match skips them."""
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(read_json(index, "weights index")["weight_map"].values()))
        return [model_dir / shard for shard in shards]
    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single]
    return sorted(p for p in model_dir.glob("*.safetensors") if not p.name.startswith("adapter_"))


def weight_artifacts(model_dir: Path) -> list[Path]:
    """`weight_files` plus the shard index, if there is one -- everything a swap has to move."""
    files = weight_files(model_dir)
    index = model_dir / "model.safetensors.index.json"
    return files + [index] if index.is_file() else files


def tensor_index(files: list[Path]) -> dict[str, tuple[str, tuple[int, ...]]]:
    """{tensor name: (dtype, shape)} across `files`, from their headers alone."""
    found: dict[str, tuple[str, tuple[int, ...]]] = {}
    for path in files:
        table, _ = read_safetensors_header(path)
        for name, entry in table.items():
            found[name] = (entry["dtype"], tuple(entry["shape"]))
    return found


def _dims_from_tensors(index: dict, layer: int, head_dim: int) -> dict | None:
    """One layer's dimensions, read off its own weight shapes -- the ground truth the sidecar's
    shape records are checked against. A width-wrapped block keeps its attention/MLP under
    `inner.`, a base block does not; q_proj is [heads*head_dim, hidden], k_proj [kv_heads*head_dim,
    hidden], gate_proj [intermediate, hidden]."""
    for prefix in (f"model.layers.{layer}.inner.", f"model.layers.{layer}."):
        if f"{prefix}self_attn.q_proj.weight" in index:
            break
    else:
        return None
    try:
        q_out, hidden = index[f"{prefix}self_attn.q_proj.weight"][1]
        k_out = index[f"{prefix}self_attn.k_proj.weight"][1][0]
        intermediate = index[f"{prefix}mlp.gate_proj.weight"][1][0]
    except KeyError:
        return None
    return {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": q_out // head_dim,
        "num_key_value_heads": k_out // head_dim,
    }


def describe_model(model_dir: Path, base_layers: int) -> dict:
    """Every layer of the local model with its dimensions, cross-checked three ways.

    The sidecar is replayed the way `layer_expand.expand_model` builds the stack (each round
    inserts its block(s) at the recorded positions, so a position that repeats pushes the earlier
    block down), the result must match `config.json`'s layer count, and -- when the weights are on
    disk -- every layer's own tensor shapes. A model whose pieces disagree would only fail later,
    inside a training run; this fails first, in milliseconds, naming what disagrees.
    """
    spec = read_json(model_dir / SIDECAR_NAME, "layer_grow sidecar")
    config = read_json(model_dir / "config.json", "config.json")

    if spec.get("base_num_hidden_layers") != base_layers:
        raise HelperError(
            f"BASE_LAYERS={base_layers}, but {model_dir / SIDECAR_NAME} says the base model has "
            f"{spec.get('base_num_hidden_layers')} layers."
        )

    heads = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim") or int(config["hidden_size"]) // heads)
    base_shape = {
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "num_attention_heads": heads,
        "num_key_value_heads": int(config.get("num_key_value_heads", heads)),
    }

    stack = [dict(base_shape, kind="base", round=0) for _ in range(base_layers)]
    for number, round_spec in enumerate(spec["rounds"], start=1):
        shape = {key: int(round_spec["shape"][key]) for key in DIM_KEYS}
        for position in round_spec["layers"]:
            if not 0 <= position <= len(stack):
                raise HelperError(
                    f"{SIDECAR_NAME} round {number} inserts a block at position {position}, "
                    f"outside the {len(stack)}-layer stack it builds on."
                )
            stack.insert(position, dict(shape, kind="grown", round=number))

    if config.get("num_hidden_layers") not in (None, len(stack)):
        raise HelperError(
            f"{model_dir / 'config.json'} says {config['num_hidden_layers']} layers but "
            f"{SIDECAR_NAME} describes {len(stack)}."
        )

    files = weight_files(model_dir)
    index = tensor_index(files) if files else None
    problems: list[str] = []
    for position, layer in enumerate(stack):
        layer["index"], layer["number"] = position, position + 1
        layer["params"], layer["dims_from"] = None, "sidecar"
        if index is None:
            continue
        measured = _dims_from_tensors(index, position, head_dim)
        if measured is None:
            problems.append(f"layer index {position}: no attention/MLP weights found")
            continue
        layer["dims_from"] = "weights"
        layer["params"] = sum(
            math.prod(shape)
            for name, (_, shape) in index.items()
            if name.startswith(f"model.layers.{position}.")
        )
        recorded = {key: layer[key] for key in DIM_KEYS}
        if measured != recorded:
            problems.append(
                f"layer index {position}: weights are {measured} but the sidecar says {recorded}"
            )
    if index is not None:
        deepest = max((int(m.group(1)) for n in index if (m := _LAYER_KEY.match(n))), default=-1)
        if deepest + 1 != len(stack):
            problems.append(f"weights hold {deepest + 1} layers, the sidecar describes {len(stack)}")
    if problems:
        raise HelperError(
            f"{model_dir} is inconsistent -- " + "; ".join(problems[:3])
            + ("; ..." if len(problems) > 3 else "")
        )

    return {
        "spec": spec,
        "config": config,
        "layers": stack,
        "total": len(stack),
        "base_layers": base_layers,
        "has_weights": index is not None,
    }


def print_layer_table(model_dir: Path, info: dict, selected: list[int]) -> None:
    base_layers, total = info["base_layers"], info["total"]
    base = info["layers"][0]
    note(
        f"Model {model_dir}: {total} layers = {base_layers} base + {total - base_layers} extra "
        f"(base model {info['spec']['base_model']})"
    )
    note(
        f"  layers  1-{base_layers:<2} (index 0-{base_layers - 1}):  base     "
        f"{base['hidden_size']}d / {base['intermediate_size']} MLP / "
        f"{base['num_attention_heads']} heads / {base['num_key_value_heads']} KV   frozen, never trained here"
    )
    for layer in info["layers"][base_layers:]:
        params = f"{layer['params']:>14,} params" if layer["params"] else " " * 21
        mark = "  <- will be trained" if layer["index"] in selected else ""
        note(
            f"  layer  {layer['number']:<3}     (index {layer['index']:>2}):  extra    "
            f"{layer['hidden_size']}d / {layer['intermediate_size']} MLP / "
            f"{layer['num_attention_heads']} heads / {layer['num_key_value_heads']} KV  {params}{mark}"
        )
    if not info["has_weights"]:
        note("  (weights are not on disk yet: dimensions come from the sidecar alone, not checked against them)")


def parse_numbers(text: str, base_layers: int, total: int) -> list[int]:
    """`LAYER_NUMBERS` -- 1-based layer numbers, the way the extra layers are counted (25..30 for a
    24-layer base grown to 30) -- as 0-based indices, the way `layer_grow.layers` counts them."""
    try:
        numbers = [int(token) for token in re.split(r"[\s,]+", text.strip()) if token]
    except ValueError as exc:
        raise HelperError(f"LAYER_NUMBERS must be whole numbers, got {text!r}.") from exc
    bad = [n for n in numbers if not base_layers < n <= total]
    if bad or not numbers:
        raise HelperError(
            f"LAYER_NUMBERS={text!r}: only the extra layers {base_layers + 1}-{total} can be "
            f"trained here (layers 1-{base_layers} are the frozen base)."
            + (f" Not in range: {bad}." if bad else "")
        )
    if len(set(numbers)) != len(numbers):
        raise HelperError(f"LAYER_NUMBERS={text!r} repeats a layer.")
    return [n - 1 for n in numbers]


# --------------------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace) -> None:
    """Fetch the whole model repo into `--model-dir`, unless a finished download is recorded.

    The marker is what makes a re-run safe: `snapshot_download` into an existing directory
    compares every file with the Hub's and re-fetches any that differ -- which, once a loop has
    pasted fine-tuned weights over model.safetensors, would silently throw that work away.
    """
    model_dir, marker = Path(args.model_dir), Path(args.marker)
    if marker.is_file() and not args.metadata_only:
        note(f"Model already downloaded ({marker}); leaving {model_dir} as it is.")
        return
    if args.metadata_only and (model_dir / SIDECAR_NAME).is_file():
        return

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise HelperError("huggingface_hub is not installed -- run `poetry install -E gdrive`.") from exc

    # Pinned to one commit so every file comes from the same version of the repo, even if it is
    # pushed to while the multi-gigabyte weights are still downloading.
    sha = HfApi().model_info(args.repo_id, revision=args.revision).sha
    patterns = ["config.json", SIDECAR_NAME, "layer_expand.json"] if args.metadata_only else None
    note(
        f"Downloading {args.repo_id}@{sha[:10]} "
        f"{'(metadata only) ' if args.metadata_only else ''}into {model_dir} ..."
    )
    snapshot_download(args.repo_id, revision=sha, local_dir=str(model_dir), allow_patterns=patterns)
    if args.metadata_only:
        return

    missing = [name for name in ("config.json", SIDECAR_NAME) if not (model_dir / name).is_file()]
    if missing or not weight_files(model_dir):
        raise HelperError(
            f"{args.repo_id} did not give a whole layer_grow model "
            f"(missing: {', '.join(missing) or 'safetensors weights'}). It needs config.json, "
            f"{SIDECAR_NAME} and the full model weights."
        )
    # The Hub client's resume bookkeeping. Not part of the model, and no longer needed once the
    # download is done and never repeated.
    shutil.rmtree(model_dir / ".cache", ignore_errors=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"repo_id": args.repo_id, "revision": sha}, indent=2) + "\n")


# --------------------------------------------------------------------------------------
# layers / settings
# --------------------------------------------------------------------------------------


def cmd_layers(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_dir)
    info = describe_model(model_dir, args.base_layers)
    extra = [layer["index"] for layer in info["layers"][args.base_layers :]]
    if not extra:
        raise HelperError(f"{model_dir} has no layers beyond the {args.base_layers} base layers.")
    selected = parse_numbers(args.numbers, args.base_layers, info["total"]) if args.numbers else extra
    print_layer_table(model_dir, info, selected)
    # The two lines the shell reads: the model's layer count, then the 0-based indices to train.
    print(info["total"])
    print(" ".join(str(index) for index in selected))


def load_base_config(path: str) -> dict:
    import yaml

    source = Path(path)
    if not source.is_file():
        raise HelperError(f"base config not found: {source}")
    config = yaml.safe_load(source.read_text()) or {}
    for section in ("model", "layer_grow", "data", "training"):
        if not isinstance(config.get(section), dict):
            config[section] = {}
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    """The named knobs first, then every `--set dotted.key=value`, so a --set can override
    anything -- including a value one of the named knobs just wrote."""
    import yaml

    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_samples is not None:
        config["data"]["max_samples"] = args.max_samples
    for item in args.set or []:
        key, separator, raw = item.partition("=")
        parts = key.strip().split(".")
        if not separator or not all(parts):
            raise HelperError(f"OVERRIDES entries look like dotted.key=value, got {item!r}.")
        node = config
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise HelperError(f"OVERRIDES {item!r}: {part!r} is not a section of the config.")
        node[parts[-1]] = yaml.safe_load(raw)


def cmd_settings(args: argparse.Namespace) -> None:
    config = load_base_config(args.base_config)
    apply_overrides(config, args)
    training, data = config["training"], config["data"]
    batch = training.get("per_device_train_batch_size", 2)
    accumulation = training.get("gradient_accumulation_steps", 4)
    seq = training.get("max_seq_length", 1024)
    early = (
        f"early stopping on eval_loss, patience {training.get('early_stopping_patience', 1)}"
        if training.get("early_stopping")
        else "no early stopping"
    )
    note(f"Training settings -- {args.base_config} plus overrides, identical for every iteration:")
    note(f"  max_steps          {training.get('max_steps', -1)}  ({early})")
    note(f"  learning_rate      {training.get('learning_rate', 2e-4)}")
    note(f"  batch              {batch} x {accumulation} grad-accum = {batch * accumulation} per step")
    note(f"  max_seq_length     {seq}")
    note(f"  max_samples        {data.get('max_samples')} train / {data.get('max_eval_samples')} eval")
    note(f"  save_steps         {training.get('save_steps', 200)}, keeping {training.get('save_total_limit')} checkpoints")
    if args.expect_min_seq_length and seq < args.expect_min_seq_length:
        note(
            f"  WARNING: max_seq_length {seq} is below the {args.expect_min_seq_length} these datasets "
            "need -- longer records are cut off mid-answer. Raise it with OVERRIDES (and lower "
            "training.per_device_train_batch_size to fit memory), see the header of "
            "layer_grow_multiple_run.sh."
        )


# --------------------------------------------------------------------------------------
# prefetch
# --------------------------------------------------------------------------------------


def normalise_zip_id(value: str | None) -> str | None:
    """A Drive file id from either a bare id or a pasted share link (`.../d/<id>/view`,
    `...?id=<id>`) -- gdown's `id=` argument takes only the former."""
    if not value:
        return None
    value = value.strip()
    if "google.com" in value:
        match = re.search(r"/d/([\w-]+)", value) or re.search(r"[?&]id=([\w-]+)", value)
        if not match:
            raise HelperError(f"cannot find a Google Drive file id in {value!r}.")
        return match.group(1)
    return value


def dataset_patterns(config: dict) -> tuple[str, str | None]:
    """The train-shard glob and eval file name the base config reads, relative to the cache dir."""
    data = config["data"]
    train = Path(str(data.get("dataset_name") or "sft_train-*.jsonl")).name
    evaluation = data.get("eval_dataset_name")
    return train, (Path(str(evaluation)).name if evaluation else None)


def dataset_problem(directory: Path, train_pattern: str, eval_name: str | None) -> str | None:
    """Why `directory` is not a usable dataset, or None if it is.

    Checks the files the trainer will glob for and that the first record is something the loader
    can turn into text -- enough to catch a zip of the wrong thing (a GRPO set, a model
    checkpoint) in seconds instead of at the first training step.
    """
    shards = sorted(directory.glob(train_pattern))
    if not shards:
        return f"no files matching {train_pattern}"
    if eval_name and not (directory / eval_name).is_file():
        return f"no {eval_name} (early stopping and eval loss need it)"
    with shards[0].open("rb") as handle:
        first = next((line for line in handle if line.strip()), b"")
    try:
        record = json.loads(first)
    except ValueError:
        return f"{shards[0].name} does not start with a JSON record"
    if not isinstance(record, dict) or not ("messages" in record or "text" in record):
        return f"{shards[0].name}: first record has neither 'messages' nor 'text'"
    return None


def count_rows(shards: list[Path], stop_at: int) -> int:
    """Non-blank lines across `shards`, counting no further than `stop_at` -- enough to know
    whether a dataset holds `max_samples` rows without reading a multi-gigabyte set end to end."""
    total = 0
    for shard in shards:
        with shard.open("rb") as handle:
            for line in handle:
                if line.strip():
                    total += 1
                    if total >= stop_at:
                        return total
    return total


def cmd_prefetch(args: argparse.Namespace) -> None:
    """Download one dataset zip into `--cache-dir`, validate it, and only then put it in place.

    Extracted into `<cache-dir>.partial` first and renamed on success, because the trainer's own
    rule is "skip the download if the cache dir is non-empty" -- an interrupted extraction
    straight into the real directory would otherwise be mistaken for a finished dataset and
    train on whatever shards happened to land.
    """
    cache = Path(args.cache_dir)
    train_pattern, eval_name = dataset_patterns(load_base_config(args.base_config))

    if cache.is_dir() and any(cache.iterdir()):
        problem = dataset_problem(cache, train_pattern, eval_name)
        if problem:
            raise HelperError(
                f"{cache} already exists but is not a usable dataset ({problem}). Remove it to "
                "have it downloaded again, or fix its contents."
            )
        note(f"Dataset {args.name}: already in {cache}.")
        return

    zip_id = normalise_zip_id(args.zip_id)
    mount_path = (args.mount_path or "").strip() or None
    if not zip_id and not mount_path:
        raise HelperError(f"dataset {args.name} has neither a zip file id nor a Drive mount path.")

    staging = cache.with_name(cache.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    if cache.is_dir():
        cache.rmdir()  # empty; the rename below needs the name free
    cache.parent.mkdir(parents=True, exist_ok=True)

    note(f"Dataset {args.name}: fetching into {staging} ...")
    try:
        from tiny_lora.data import ensure_gdrive_dataset

        ensure_gdrive_dataset(staging, zip_id, mount_path)
    except Exception as exc:  # gdown/zipfile/IO errors all deserve the same advice
        raise HelperError(
            f"could not fetch dataset {args.name}: {type(exc).__name__}: {exc}. Google Drive "
            "throttles anonymous downloads of large files; if that is what happened, mount Drive "
            "(Colab) and give the dataset a mount path in layer_grow_multiple_run.sh, or put the "
            f"extracted files in {cache} yourself."
        ) from exc

    problem = dataset_problem(staging, train_pattern, eval_name)
    if problem:
        raise HelperError(
            f"dataset {args.name} downloaded, but {problem}. Left in {staging} for inspection; "
            "check that the zip holds the sft_train-*.jsonl shards and sft_eval.jsonl."
        )
    staging.rename(cache)
    shards = sorted(cache.glob(train_pattern))
    gigabytes = sum(p.stat().st_size for p in shards) / 1e9
    note(f"Dataset {args.name}: ready -- {len(shards)} train shard(s), {gigabytes:.1f} GB, in {cache}.")


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------


def cmd_config(args: argparse.Namespace) -> None:
    """Write one iteration's config: the base config with this layer, this dataset and the local
    model patched in. The base file itself is never modified."""
    import yaml

    model_dir = Path(args.model_dir)
    info = describe_model(model_dir, args.base_layers)
    if not args.base_layers <= args.layer_index < info["total"]:
        raise HelperError(
            f"layer index {args.layer_index} is not one of the extra layers "
            f"({args.base_layers}-{info['total'] - 1})."
        )
    layer = info["layers"][args.layer_index]
    dims = {key: layer[key] for key in DIM_KEYS}

    config = load_base_config(args.base_config)
    train_pattern, eval_name = dataset_patterns(config)
    cache = args.cache_dir.rstrip("/")

    config["model"]["model_name_or_path"] = info["spec"]["base_model"]

    grow = config["layer_grow"]
    grow["previous_checkpoint"] = str(model_dir)
    # The model is already on this machine, so the Drive fallback the base config carries for a
    # fresh one has nothing left to do -- and its zip is a different checkpoint.
    grow.pop("gdrive", None)
    grow["layers"] = [args.layer_index]
    grow.update(dims)
    # "none", not null: the config loader ignores null, which would leave the default "auto".
    # Every iteration starts from the model as it stands now; an interrupted one is resumed by the
    # Trainer from its own checkpoint-N, which is a different mechanism.
    grow["init_from_checkpoint"] = "none"
    grow["train_only_listed_layers"] = bool(args.single_layer)

    data = config["data"]
    data["reader"] = "gdrive"
    data["dataset_name"] = f"{cache}/{train_pattern}"
    if eval_name:
        data["eval_dataset_name"] = f"{cache}/{eval_name}"
    # Replaces the whole block: a mount_path/local_zip_path in the base config would otherwise
    # leak into every dataset and point them all at one file.
    gdrive = {"cache_dir": cache}
    if normalise_zip_id(args.zip_id):
        gdrive["zip_file_id"] = normalise_zip_id(args.zip_id)
    if (args.mount_path or "").strip():
        gdrive["mount_path"] = args.mount_path.strip()
    data["gdrive"] = gdrive

    training = config["training"]
    training["output_dir"] = args.output_dir
    # Same reason as above, for training's own Drive fallback: it would drop another run's
    # checkpoint into this iteration's fresh output_dir and resume from it.
    training.pop("gdrive", None)
    # Each iteration must end on its own when eval loss stops improving, not run to max_steps --
    # the Trainer then returns normally with the best checkpoint loaded, and the loop pastes it
    # back and moves on. Forced on here so a base config that turns it off cannot make 30
    # iterations each run the full max_steps; OVERRIDES (applied next) can still change it.
    training["early_stopping"] = True
    training.setdefault("early_stopping_patience", 5)
    training.setdefault("early_stopping_threshold", 0.002)

    apply_overrides(config, args)

    capped = ""
    cap = config["data"].get("max_samples")
    shards = sorted(Path(cache).glob(train_pattern))
    if cap and shards:
        rows = count_rows(shards, int(cap))
        if rows < int(cap):
            # The multi-shard reader raises when asked for more rows than exist.
            config["data"]["max_samples"] = rows
            capped = f"; max_samples capped {cap} -> {rows} (all this dataset has)"

    header = (
        "# GENERATED by layer_grow_multiple_run.sh from "
        f"{args.base_config} -- do not edit; the next iteration writes its own.\n"
        f"# The base config is never touched, so layer_grow_install.sh runs exactly as before.\n"
        f"#   iteration : layer {layer['number']} of {info['total']} (layer_grow.layers: "
        f"[{args.layer_index}]) on {args.dataset}\n"
        f"#   dimensions: {dims['hidden_size']}d / {dims['intermediate_size']} MLP / "
        f"{dims['num_attention_heads']} heads / {dims['num_key_value_heads']} KV heads, read from "
        f"{layer['dims_from']} of {model_dir}\n"
        "#               (layer_grow only uses them to build a NEW block; a round that refines an\n"
        "#               existing layer, like this one, prints a note and ignores them)\n"
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + yaml.safe_dump(config, sort_keys=False, default_flow_style=False))

    mode = "this layer only" if args.single_layer else "this layer + every grown block"
    note(
        f"Config {out}: layer {layer['number']} (index {args.layer_index}) x {args.dataset}, "
        f"{dims['hidden_size']}d/{dims['intermediate_size']}/{dims['num_attention_heads']}h/"
        f"{dims['num_key_value_heads']}kv, trains {mode}{capped}"
    )


# --------------------------------------------------------------------------------------
# paste
# --------------------------------------------------------------------------------------


def move_into_place(source: Path, target: Path) -> None:
    """Rename `source` over `target` atomically; across filesystems, copy beside it and rename."""
    try:
        os.replace(source, target)
    except OSError:
        staged = target.with_name(target.name + ".incoming")
        shutil.copy2(source, staged)
        os.replace(staged, target)
        source.unlink()


def compare_layer_bytes(old: Path, new: Path, layer: int) -> tuple[list[str], int, int]:
    """Compare two single-file models tensor by tensor, by their raw bytes.

    Returns (names outside `layer` whose bytes differ, tensors in `layer` that changed, tensors in
    `layer`). With only that layer unfrozen nothing outside it can have moved, so a non-empty first
    list means the run touched something it should not have, or the save changed a tensor.
    """
    old_table, old_start = read_safetensors_header(old)
    new_table, new_start = read_safetensors_header(new)
    prefix = f"model.layers.{layer}."
    outside_changed: list[str] = []
    inside_changed = inside_total = 0
    with old.open("rb") as old_handle, new.open("rb") as new_handle:
        for name in sorted(old_table, key=lambda n: old_table[n]["data_offsets"][0]):
            lo, hi = old_table[name]["data_offsets"]
            new_lo, new_hi = new_table[name]["data_offsets"]
            old_handle.seek(old_start + lo)
            new_handle.seek(new_start + new_lo)
            remaining, same = hi - lo, (hi - lo) == (new_hi - new_lo)
            while same and remaining:
                size = min(16 * 1024 * 1024, remaining)
                same = old_handle.read(size) == new_handle.read(size)
                remaining -= size
            if name.startswith(prefix):
                inside_total += 1
                inside_changed += 0 if same else 1
            elif not same:
                outside_changed.append(name)
    return outside_changed, inside_changed, inside_total


def cmd_paste(args: argparse.Namespace) -> None:
    """Put a finished run's weights and sidecar into the local model, replacing what was there.

    The refused cases are the point of this function, since it overwrites the only local copy of
    the model: the run must extend the model's own sidecar by exactly one refine round of this
    layer (so it really grew from this model, not another), its tensors must match the model's in
    name, dtype and shape, and -- when only this layer was meant to train -- every other tensor
    must be byte-identical to what it replaces. Files are swapped with renames, never rewritten
    in place, so an interruption cannot leave a half-written model.
    """
    model_dir, run_dir = Path(args.model_dir), Path(args.run_dir)
    final, layer = run_dir / "model", args.layer_index

    old_spec = read_json(model_dir / SIDECAR_NAME, "the model's sidecar")
    new_spec = read_json(final / SIDECAR_NAME, "the finished run's sidecar")
    old_rounds, new_rounds = old_spec["rounds"], new_spec["rounds"]
    if new_rounds[:-1] != old_rounds or len(new_rounds) != len(old_rounds) + 1:
        raise HelperError(
            f"{final / SIDECAR_NAME} does not extend {model_dir / SIDECAR_NAME} by exactly one "
            "round, so the run did not grow from this model. Nothing was pasted."
        )
    added = new_rounds[-1]
    if added.get("layers") or added.get("refine_layers") != [layer]:
        raise HelperError(
            f"the finished run's last round is {added.get('layers')!r}/refine "
            f"{added.get('refine_layers')!r}, expected a refine of just [{layer}]. Nothing was pasted."
        )

    new_files = weight_artifacts(final)
    if not weight_files(final):
        raise HelperError(f"no weights in {final} -- the run did not finish saving. Nothing was pasted.")

    old_index, new_index = tensor_index(weight_files(model_dir)), tensor_index(weight_files(final))
    if old_index != new_index:
        gone = sorted(set(old_index) - set(new_index))[:3]
        added_names = sorted(set(new_index) - set(old_index))[:3]
        changed = sorted(n for n in set(old_index) & set(new_index) if old_index[n] != new_index[n])[:3]
        raise HelperError(
            "the run's tensors do not match the model's (dtype/shape/name). Nothing was pasted. "
            f"Missing: {gone}; new: {added_names}; different: {changed}."
        )

    unchanged_note = ""
    old_files, new_weight_files = weight_files(model_dir), weight_files(final)
    if len(old_files) == len(new_weight_files) == 1:
        outside, inside_changed, inside_total = compare_layer_bytes(
            old_files[0], new_weight_files[0], layer
        )
        if args.single_layer and args.verify and outside:
            raise HelperError(
                f"{len(outside)} tensor(s) outside layer index {layer} differ from the model "
                f"being replaced (e.g. {outside[0]}), though only that layer was meant to train. "
                f"Nothing was pasted; the run is kept in {run_dir}. Set VERIFY_PASTE=0 to paste anyway."
            )
        if inside_changed == 0:
            unchanged_note = (
                f" WARNING: layer index {layer} is byte-identical to before -- the run did not "
                "change it (learning rate too small for bf16 to register, or no training steps)."
            )
        else:
            unchanged_note = f" {inside_changed}/{inside_total} of its tensors changed."
    elif args.single_layer and args.verify:
        note("  (weights are sharded: skipped the byte-level check that nothing else changed)")

    # Listed before the swap: once a new shard index is in place `weight_artifacts` would follow
    # it and no longer see an old single model.safetensors (or the reverse), leaving a stale file
    # that a later load could prefer over the new weights.
    old_artifacts = weight_artifacts(model_dir)
    for source in new_files:
        move_into_place(source, model_dir / source.name)
    kept = {source.name for source in new_files}
    for stale in old_artifacts:
        if stale.name not in kept:
            stale.unlink()
    staged = model_dir / (SIDECAR_NAME + ".incoming")
    shutil.copyfile(final / SIDECAR_NAME, staged)
    os.replace(staged, model_dir / SIDECAR_NAME)

    if args.progress_file and args.progress_key:
        progress = Path(args.progress_file)
        progress.parent.mkdir(parents=True, exist_ok=True)
        with progress.open("a") as handle:
            handle.write(args.progress_key + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    note(
        f"Pasted layer index {layer} into {model_dir} "
        f"({len(new_rounds)} growth rounds recorded).{unchanged_note}"
    )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-config", default="configs/sft_layer.yaml")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--set", action="append", metavar="DOTTED.KEY=VALUE")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    sub = commands.add_parser("download", help="fetch the Hugging Face model once")
    sub.add_argument("--repo-id", required=True)
    sub.add_argument("--revision", default="main")
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--marker", required=True)
    sub.add_argument("--metadata-only", action="store_true")
    sub.set_defaults(run=cmd_download)

    sub = commands.add_parser("layers", help="list the extra layers and their dimensions")
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--base-layers", type=int, required=True)
    sub.add_argument("--numbers", default="")
    sub.set_defaults(run=cmd_layers)

    sub = commands.add_parser("settings", help="print the effective training settings")
    _overrides(sub)
    sub.add_argument("--expect-min-seq-length", type=int, default=0)
    sub.set_defaults(run=cmd_settings)

    sub = commands.add_parser("prefetch", help="download and validate one dataset")
    sub.add_argument("--name", required=True)
    sub.add_argument("--cache-dir", required=True)
    sub.add_argument("--zip-id", default="")
    sub.add_argument("--mount-path", default="")
    sub.add_argument("--base-config", default="configs/sft_layer.yaml")
    sub.set_defaults(run=cmd_prefetch)

    sub = commands.add_parser("config", help="write one iteration's training config")
    _overrides(sub)
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--base-layers", type=int, required=True)
    sub.add_argument("--layer-index", type=int, required=True)
    sub.add_argument("--dataset", required=True)
    sub.add_argument("--cache-dir", required=True)
    sub.add_argument("--zip-id", default="")
    sub.add_argument("--mount-path", default="")
    sub.add_argument("--output-dir", required=True)
    sub.add_argument("--out", required=True)
    mode = sub.add_mutually_exclusive_group()
    mode.add_argument("--single-layer", dest="single_layer", action="store_true", default=True)
    mode.add_argument("--joint", dest="single_layer", action="store_false")
    sub.set_defaults(run=cmd_config)

    sub = commands.add_parser("paste", help="put a finished run back into the local model")
    sub.add_argument("--model-dir", required=True)
    sub.add_argument("--run-dir", required=True)
    sub.add_argument("--layer-index", type=int, required=True)
    sub.add_argument("--progress-file", default="")
    sub.add_argument("--progress-key", default="")
    mode = sub.add_mutually_exclusive_group()
    mode.add_argument("--single-layer", dest="single_layer", action="store_true", default=True)
    mode.add_argument("--joint", dest="single_layer", action="store_false")
    sub.add_argument("--no-verify", dest="verify", action="store_false", default=True)
    sub.set_defaults(run=cmd_paste)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.run(args)
    except HelperError as exc:
        print(f"layer_grow_multi: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
