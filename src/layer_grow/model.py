"""Grow an already-expanded stack by one more round, keeping every added block trainable.

The starting point is always a finished `layer_expand` or `layer_grow` checkpoint -- there is no
"grow the stock base model" case, unlike `layer_expand.base_adapters` which can be empty. Reuses
`layer_expand.model`'s geometry/splicing/freezing/weight-loading functions wherever the logic is
round-agnostic (it mostly is: `expand_model` and `freeze_except` do not care how many blocks came
before, only which indices to touch); what is new here is a sidecar format that accumulates one
entry per round instead of describing a single one, and freezing the *union* of every round's
layers instead of only the newest.
"""

from __future__ import annotations

import json
from pathlib import Path

from layer_expand.model import (
    FINAL_DIR_NAME as _EXPAND_FINAL_DIR_NAME,
    SIDECAR_NAME as _EXPAND_SIDECAR_NAME,
    base_model_geometry,
    block_parameter_count,
    expand_model,
    freeze_except,
    latest_checkpoint_in,
    load_expanded_weights,
    resolve_shape,
    validate_layers,
)
from layer_grow.config import LayerGrowConfig
from tiny_lora.config import ModelConfig
from tiny_lora.gdrive import download_and_extract_zip, flatten_single_wrapper_dir
from tiny_lora.model import load_base_model

SIDECAR_NAME = "layer_grow.json"
FINAL_DIR_NAME = _EXPAND_FINAL_DIR_NAME  # "model" -- same convention, same reason

# Written into a grown stack's config.json in place of "qwen2" / "layer_expand", for the same
# reason layer_expand retypes its own: a stock (or layer_expand-only) loader must fail loudly on
# a shape it cannot rebuild, not silently reinitialise the extra blocks and hand back a model
# that loads fine and produces garbage.
GROWN_MODEL_TYPE = "layer_grow"


# --------------------------------------------------------------------------------------
# Reading either sidecar format, normalised to one shape: a list of rounds
# --------------------------------------------------------------------------------------


def _normalise_expand_sidecar(spec: dict) -> dict:
    """A `layer_expand.json` sidecar, wrapped as a one-round growth spec."""
    return {
        "base_model": spec["base_model"],
        "base_num_hidden_layers": spec["base_num_hidden_layers"],
        "rounds": [
            {
                "layers": spec["layers"],
                "shape": spec["shape"],
                "uniform": spec["uniform"],
                "wrapped": spec["wrapped"],
                "init": spec["init"],
                "block_parameters": spec["block_parameters"],
                "source": "layer_expand",
            }
        ],
        "base_adapters": spec.get("base_adapters", []),
    }


def read_growth_spec(path: str | Path) -> dict | None:
    """The normalised, multi-round growth spec for `path`, or None if it holds neither sidecar.

    Checks this module's own `layer_grow.json` first, then falls back to a plain
    `layer_expand.json` (a checkpoint that has only ever been through one round) -- either way
    the caller gets the same `{base_model, base_num_hidden_layers, rounds, base_adapters}` shape.
    """
    path = Path(path)
    own = path / SIDECAR_NAME
    if own.is_file():
        return json.loads(own.read_text())
    legacy = path / _EXPAND_SIDECAR_NAME
    if legacy.is_file():
        return _normalise_expand_sidecar(json.loads(legacy.read_text()))
    return None


def is_growable_checkpoint_dir(path: str | Path) -> bool:
    """True when `path` holds something `layer_grow` can start another round from."""
    return read_growth_spec(path) is not None


def is_grown_model_dir(path: str | Path) -> bool:
    """True when `path` holds a `layer_grow` checkpoint specifically (two-or-more rounds, or a
    single round that has already been through this module once). Distinct from
    `layer_expand.model.is_expanded_model_dir`, which only ever sees a `layer_expand.json` --
    `tiny_lora.model` checks both so `chat`/`eval`/`serve` accept either kind of checkpoint."""
    return (Path(path) / SIDECAR_NAME).is_file()


def _shape_dict(round_spec: dict) -> dict:
    """The `shape` dict `expand_model` wants, rebuilt from one round entry."""
    return {**round_spec["shape"], "uniform": round_spec["uniform"], "wrapped": round_spec["wrapped"]}


def all_grown_layers(spec: dict) -> list[int]:
    """Every layer index any round added, in the order the rounds happened."""
    return [index for round_spec in spec["rounds"] for index in round_spec["layers"]]


# --------------------------------------------------------------------------------------
# Rebuilding the architecture from a spec, and loading weights over it
# --------------------------------------------------------------------------------------


def build_from_rounds(model, rounds: list[dict]):
    """Splice every round's block(s) into `model`, in order. Weights are freshly built (identity
    or random per round); the caller is expected to load the real, trained weights immediately
    after -- same division of labour as `layer_expand.model.expand_model` itself."""
    for round_spec in rounds:
        model = expand_model(
            model, round_spec["layers"], _shape_dict(round_spec), round_spec["init"], 0.0
        )
    return model


def resolve_growth_shape(grow_cfg: LayerGrowConfig, geometry: dict, previous_shape: dict) -> dict:
    """This round's block shape: the previous round's own shape by default, or an explicit
    override.

    Different from `layer_expand.model.resolve_shape`, which inherits unset fields from the
    *base model's* shape -- here the natural default is "the same as whatever I'm growing", so an
    all-null config keeps widening the same wide block rather than silently dropping back to
    base width.
    """
    shape_fields = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")
    if all(getattr(grow_cfg, field) is None for field in shape_fields):
        shape = dict(previous_shape)
        shape["base_hidden_size"] = geometry["hidden_size"]
        shape["wrapped"] = shape["hidden_size"] != geometry["hidden_size"]
        shape["uniform"] = all(
            shape[key] == geometry[key]
            for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")
        )
        return shape
    return resolve_shape(grow_cfg, geometry)


def freeze_all_growth(model, spec: dict) -> tuple[int, int]:
    """Freeze the base layers only; every block any round has ever added stays trainable.

    This is the one behavioural difference from `layer_expand.freeze_except`, which freezes
    everything but the block(s) *this* run adds. Reuses that same function underneath -- it just
    freezes-then-unfreezes a wider set of indices.
    """
    return freeze_except(model, all_grown_layers(spec))


# --------------------------------------------------------------------------------------
# Resolving `layer_grow.previous_checkpoint`
# --------------------------------------------------------------------------------------


def _latest_growable(directory: Path) -> Path | None:
    """The checkpoint `directory` can grow from, or None if it holds no usable one.

    `directory` may be the checkpoint itself (its own layer_expand.json/layer_grow.json) or a run
    directory containing one, in which case the latest -- the finished `model/` if there is one,
    else the highest `checkpoint-N` -- is taken.
    """
    if is_growable_checkpoint_dir(directory):
        return directory
    if not directory.is_dir():
        return None
    found = latest_checkpoint_in(directory, FINAL_DIR_NAME)
    if found is not None and is_growable_checkpoint_dir(found):
        return found
    return None


def resolve_previous_checkpoint(
    spec: str,
    gdrive_zip_file_id: str | None = None,
    gdrive_cache_dir: str | None = None,
) -> Path:
    """`layer_grow.previous_checkpoint` resolved to an exact checkpoint directory.

    Accepts either the checkpoint itself (a directory holding `layer_expand.json` or
    `layer_grow.json` directly) or the run directory containing one, in which case the latest
    weights are taken -- the finished `model/` if there is one, else the highest `checkpoint-N`.
    Same resolution rule `layer_expand.base_adapters` uses for adapter directories.

    Falls back to Google Drive when nothing usable is found locally and `gdrive_zip_file_id` is
    set: downloads and extracts that zip into `gdrive_cache_dir` (defaulting to `spec` itself, so
    a zip of one checkpoint's contents lands exactly where that path expects it) and resolves
    again from there -- the same "fresh machine picks up a run checkpointed elsewhere" case
    `tiny_lora.train_sft.resolve_resume_checkpoint` handles for `training.gdrive`. Skipped
    entirely if `spec` already resolves locally, so this costs nothing on a machine that already
    has the checkpoint.
    """
    candidate = Path(spec)
    found = _latest_growable(candidate)
    if found is not None:
        return found

    if not gdrive_zip_file_id:
        raise FileNotFoundError(
            f"layer_grow.previous_checkpoint: no layer_expand/layer_grow checkpoint at "
            f"{candidate} (no layer_expand.json/layer_grow.json there, and no checkpoint-N under "
            "it carrying one), and layer_grow.gdrive.zip_file_id is not set to fall back to. "
            "Point at a finished run or a specific checkpoint-N inside one, e.g. "
            "outputs/sft-layer-expand-wide or outputs/sft-layer-expand-wide/checkpoint-16000, or "
            "set layer_grow.gdrive.zip_file_id to a zipped copy of one."
        )

    cache_dir = Path(gdrive_cache_dir) if gdrive_cache_dir else candidate
    found = _latest_growable(cache_dir)
    if found is not None:
        print(
            f"{cache_dir} already has a layer_expand/layer_grow checkpoint; skipping the Google "
            "Drive download."
        )
        return found

    print(
        f"No checkpoint at {candidate}; downloading layer_grow.previous_checkpoint from Google "
        f"Drive ({gdrive_zip_file_id}) into {cache_dir}."
    )
    download_and_extract_zip(cache_dir, gdrive_zip_file_id, "_gdrive_previous_checkpoint.zip")
    flatten_single_wrapper_dir(cache_dir)

    found = _latest_growable(cache_dir)
    if found is None:
        raise FileNotFoundError(
            "layer_grow.previous_checkpoint: still no usable layer_expand/layer_grow checkpoint "
            f"after downloading and extracting {gdrive_zip_file_id} into {cache_dir}. Check that "
            "the zip holds a checkpoint-N directory (or its contents directly) carrying "
            "layer_expand.json or layer_grow.json."
        )
    return found


# --------------------------------------------------------------------------------------
# Persisting this round's sidecar
# --------------------------------------------------------------------------------------


def growth_spec(
    model_cfg: ModelConfig,
    previous_spec: dict,
    new_layers: list[int],
    shape: dict,
    init: str,
) -> dict:
    """The full, accumulated sidecar this run writes: every earlier round plus this one."""
    new_round = {
        "layers": new_layers,
        "shape": {k: v for k, v in shape.items() if k not in ("uniform", "wrapped")},
        "uniform": shape["uniform"],
        "wrapped": shape["wrapped"],
        "init": init,
        "block_parameters": block_parameter_count(shape),
        "source": "layer_grow",
    }
    return {
        "base_model": previous_spec["base_model"],
        "base_num_hidden_layers": previous_spec["base_num_hidden_layers"],
        "rounds": [*previous_spec["rounds"], new_round],
        "base_adapters": previous_spec.get("base_adapters", []),
    }


def check_growth_matches(output_dir: Path, spec: dict) -> None:
    """Refuse to write a differently-grown run into a directory that already holds one.

    Mirrors `layer_expand.model.check_sidecar_matches`: the checkpoints in `output_dir` were
    written by a stack of a particular shape, and the Trainer resumes from them by name without
    checking, so a shape change here would either crash deep inside `load_state_dict` or -- worse
    -- silently train the wrong architecture while reporting the old run's step count.
    """
    previous = read_growth_spec(output_dir)
    if previous is None:
        return
    for key in ("rounds", "base_model"):
        if previous.get(key) != spec.get(key):
            raise ValueError(
                f"{output_dir} holds a run with {key}={previous.get(key)!r}, but this config "
                f"asks for {key}={spec.get(key)!r}. Checkpoints in that directory belong to the "
                "old shape and cannot be resumed into the new one. Point training.output_dir "
                "somewhere else, or delete the old run."
            )


def write_sidecar(directory: Path, spec: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SIDECAR_NAME).write_text(json.dumps(spec, indent=2) + "\n")


def stamp_config(directory: Path, wrapped: bool) -> None:
    """Retype a wrapped grown stack's saved config.json, the same way layer_expand does -- just
    to this module's own model_type string, so a stock or layer_expand-only loader fails loudly
    instead of silently reinitialising the newest block(s)."""
    if not wrapped:
        return
    config_path = directory / "config.json"
    if not config_path.is_file():
        return
    data = json.loads(config_path.read_text())
    if data.get("model_type") == GROWN_MODEL_TYPE:
        return
    data["base_model_type"] = data.get("base_model_type", data.get("model_type"))
    data["model_type"] = GROWN_MODEL_TYPE
    data["architectures"] = ["LayerGrowForCausalLM"]
    config_path.write_text(json.dumps(data, indent=2) + "\n")


def describe_growth(spec: dict) -> str:
    rounds = spec["rounds"]
    pieces = [
        f"round {i + 1}: layer(s) {r['layers']} ({r['shape']['hidden_size']}d/"
        f"{r['shape']['intermediate_size']}, {r['block_parameters']:,} params, {r['source']})"
        for i, r in enumerate(rounds)
    ]
    return "; ".join(pieces)


# --------------------------------------------------------------------------------------
# Entry point: building the model to train
# --------------------------------------------------------------------------------------


def load_layer_grow_model(model_cfg: ModelConfig, grow_cfg: LayerGrowConfig, output_dir: Path):
    """Build the grown model with every added block -- old and new -- trainable.

    Returns `(model, init_checkpoint)`, matching `layer_expand.model.load_layer_expand_model`'s
    contract: `init_checkpoint` is where *this round's* weights were loaded from, or None when a
    fresh block was built for it, and the caller needs it for the same resume-vs-fresh-block
    decision layer_expand makes.

    The order:
      1. resolve `previous_checkpoint` and load it -- this is the whole earlier stack, frozen or
         not, exactly as that run left it,
      2. splice in one more block, identity-initialised,
      3. load this round's own previous weights over the top if there is one,
      4. freeze the base only; every block from every round stays trainable.
    """
    if not grow_cfg.previous_checkpoint:
        raise ValueError(
            "layer_grow.previous_checkpoint is required -- point it at a finished layer_expand "
            "or layer_grow checkpoint to grow from, e.g. outputs/sft-layer-expand-wide/"
            "checkpoint-13000."
        )

    previous_checkpoint = resolve_previous_checkpoint(
        grow_cfg.previous_checkpoint,
        grow_cfg.gdrive_zip_file_id,
        grow_cfg.gdrive_cache_dir,
    )
    previous_spec = read_growth_spec(previous_checkpoint)

    geometry = base_model_geometry(model_cfg)
    stack_so_far = previous_spec["base_num_hidden_layers"] + len(all_grown_layers(previous_spec))
    new_layers = validate_layers(grow_cfg.layers or [stack_so_far], stack_so_far)

    previous_shape = _shape_dict(previous_spec["rounds"][-1])
    shape = resolve_growth_shape(grow_cfg, geometry, previous_shape)

    spec = growth_spec(model_cfg, previous_spec, new_layers, shape, grow_cfg.init)
    check_growth_matches(output_dir, spec)

    init_checkpoint = None
    if grow_cfg.init_from_checkpoint and grow_cfg.init_from_checkpoint.strip().lower() not in (
        "none",
        "off",
        "",
    ):
        if grow_cfg.init_from_checkpoint.strip().lower() == "auto":
            candidate = output_dir / FINAL_DIR_NAME
            from layer_expand.model import _WEIGHT_FILES, _INDEX_FILES  # noqa: PLC0415

            if any((candidate / name).is_file() for name in _WEIGHT_FILES + _INDEX_FILES):
                init_checkpoint = candidate
        else:
            init_checkpoint = Path(grow_cfg.init_from_checkpoint)

    print(f"Growing from {previous_checkpoint}: {describe_growth(previous_spec)}.")
    print(f"Loading previous stack ({model_cfg.model_name_or_path}) ...")
    base_model = load_base_model(model_cfg)
    if model_cfg.load_in_4bit:
        from tiny_lora.model import _bitsandbytes_available  # noqa: PLC0415

        if _bitsandbytes_available():
            from peft import prepare_model_for_kbit_training  # noqa: PLC0415

            base_model = prepare_model_for_kbit_training(base_model)

    model = build_from_rounds(base_model, previous_spec["rounds"])
    load_expanded_weights(model, previous_checkpoint, all_grown_layers(previous_spec))

    model = expand_model(model, new_layers, shape, grow_cfg.init, grow_cfg.init_std)
    if init_checkpoint is not None:
        load_expanded_weights(model, init_checkpoint, new_layers)

    trainable, total = freeze_all_growth(model, spec)
    write_sidecar(output_dir, spec)

    print(f"This round adds layer(s) {new_layers} ({shape['hidden_size']}d/{shape['intermediate_size']}).")
    print(
        f"Trainable {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%) -- "
        f"every block from every round ({all_grown_layers(spec)}), base frozen."
    )
    return model, init_checkpoint


# --------------------------------------------------------------------------------------
# Inference: loading a grown model back
# --------------------------------------------------------------------------------------


def load_grown_model(
    path: str | Path,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    eval_mode: bool = True,
):
    """Load a trained grown model for inference, from one self-contained directory.

    Rebuilds every round's block from `layer_grow.json` (or a plain `layer_expand.json`, if this
    checkpoint has only been through one round so far), then loads this checkpoint's own full
    state dict over the whole thing -- no earlier checkpoint needs to be present alongside it.
    """
    path = Path(path)
    spec = read_growth_spec(path)
    if spec is None:
        raise FileNotFoundError(
            f"No {SIDECAR_NAME} (or {_EXPAND_SIDECAR_NAME}) in {path}. Inference on a grown "
            "model needs one of these sidecars written next to its weights."
        )

    if load_in_4bit:
        print("layer_grow: 4-bit loading is not supported for grown models; using bf16.")
        load_in_4bit = False

    base_model = load_base_model(
        ModelConfig(
            model_name_or_path=spec["base_model"],
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
    )
    model = build_from_rounds(base_model, spec["rounds"])
    load_expanded_weights(model, path, all_grown_layers(spec))
    if eval_mode:
        model.eval()
    print(
        f"Loaded grown model from {path}: {model.config.num_hidden_layers} layers, "
        f"{describe_growth(spec)}."
    )
    return model


def expanded_base_model(path: str | Path) -> str:
    """The base model id a grown checkpoint was built from, out of its sidecar."""
    spec = read_growth_spec(path)
    if spec is None:
        raise FileNotFoundError(f"No {SIDECAR_NAME} in {path}.")
    return spec["base_model"]
