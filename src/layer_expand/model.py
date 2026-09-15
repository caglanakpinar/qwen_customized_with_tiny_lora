"""Build a decoder stack with extra transformer blocks, and train only those.

The method
----------
A base model's layers are frozen and one or more freshly built blocks are spliced into the
stack. Only the new blocks carry gradients. This is depth expansion (LLaMA Pro's "block
expansion"), not an adapter: the new block is a full `Qwen2DecoderLayer` and every one of its
~14.9M parameters trains.

Two properties make it cheap, and both depend on the new block being *last*:

* Nothing before it needs a backward pass. The embedding output has `requires_grad=False` and
  so does every frozen layer's output, so autograd builds no graph at all until the new block.
  Layers 0..N-1 cost a forward pass and nothing else -- no stored activations, no weight
  gradients. Appending at index `num_layers` gets this; inserting in the middle does not.
* Optimizer state is sized by the trainable block alone. AdamW over 14.9M parameters is ~170MB
  against ~7GB for the whole model.

Identity initialisation
-----------------------
A randomly initialised block dropped in front of `lm_head` destroys the model's output, and
training starts from noise rather than from a working model. Zeroing the block's *output*
projections -- `self_attn.o_proj` and `mlp.down_proj` -- makes both residual branches contribute
exactly zero, so the block is an identity function on its first forward pass and the expanded
model reproduces the base model bit for bit. Training then learns a delta from a working
starting point. This is what `init: "identity"` does and it is why it is the default.

Shape, and what stays loadable
------------------------------
`Qwen2Config` carries one scalar `hidden_size` / `intermediate_size` / `num_attention_heads` for
the entire stack, so a block shaped differently from its neighbours is *not expressible in
config.json*. Two cases follow:

* Uniform (no shape overrides): plain `Qwen2DecoderLayer`s are appended and
  `config.num_hidden_layers` is bumped. The saved model is an ordinary 25-layer Qwen2 that
  `AutoModelForCausalLM.from_pretrained` loads, that exports to GGUF, and that needs nothing
  from this package.
* Non-uniform (any shape override): the block is wrapped in `WrappedDecoderLayer`, which
  projects the 896-wide residual stream up to the block's own width and back down. The weights
  still save as safetensors, but rebuilding the architecture needs the shapes, so a
  `layer_expand.json` sidecar is written next to them and `load_layer_expand_model` reads it
  back. A stock `from_pretrained` cannot reconstruct this.

`head_dim` is deliberately not configurable. Rotary embeddings are computed once per forward
for the whole stack and are sized by `head_dim`, and every layer's KV cache entry is
`(batch, kv_heads, seq, head_dim)`. A block with a different `head_dim` would need its own
rotary table and would break cache uniformity; widening via `num_attention_heads` gets the same
capacity without either problem.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig

from layer_expand.config import DISABLED, LayerExpandConfig
from tiny_lora.config import ModelConfig
from tiny_lora.model import _bitsandbytes_available, load_base_model

# Written next to the weights whenever the stack is non-uniform, and read back to rebuild it.
SIDECAR_NAME = "layer_expand.json"
# The directory `run_sft_core` writes the finished model to. The adapter modules call theirs
# "adapter"; this one holds a whole model, so calling it that would be actively misleading.
FINAL_DIR_NAME = "model"

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


# --------------------------------------------------------------------------------------
# The wrapped block, for stacks whose new layer is shaped differently from the base
# --------------------------------------------------------------------------------------

try:  # transformers >= 4.53 gives decoder layers a gradient-checkpointing base class.
    from transformers.modeling_layers import GradientCheckpointingLayer as _LayerBase
except ImportError:  # pragma: no cover - older transformers
    _LayerBase = nn.Module


class WrappedDecoderLayer(_LayerBase):
    """A decoder block of width `d` living inside a residual stream of width `hidden_size`.

    `in_proj` lifts the residual stream to the block's width, the inner block runs its own
    attention/MLP residuals at that width, and `out_proj` projects back down to be added to the
    untouched residual. Zeroing `out_proj` makes the whole wrapper an identity, exactly as
    zeroing `o_proj`/`down_proj` does for an unwrapped block -- which is why identity init stays
    available here.

    The projections are the cost of the approach: 2 * hidden_size * d parameters that do no work
    beyond changing width, and an information bottleneck at the exit no matter how wide `d` is.
    That is the trade against simply appending more base-width blocks.
    """

    def __init__(self, inner: nn.Module, hidden_size: int, inner_size: int):
        super().__init__()
        self.inner = inner
        self.in_proj = nn.Linear(hidden_size, inner_size, bias=False)
        self.out_proj = nn.Linear(inner_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        widened = self.inner(self.in_proj(hidden_states), **kwargs)
        return residual + self.out_proj(widened)


# --------------------------------------------------------------------------------------
# Reading the base model's geometry
# --------------------------------------------------------------------------------------


def base_model_geometry(model_cfg: ModelConfig) -> dict:
    """The base model's layer count and block shape, read from config.json without weights.

    Up front so a bad `layers:` index or an unsupported shape override fails in seconds rather
    than after the minutes it takes to pull and materialise the model.
    """
    hf_config = AutoConfig.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            f"{model_cfg.model_name_or_path} reports no num_hidden_layers, so there is no "
            "decoder stack to expand. This module targets standard decoder stacks "
            "(Qwen, Llama, Mistral, ...)."
        )
    heads = hf_config.num_attention_heads
    return {
        "num_hidden_layers": int(num_layers),
        "hidden_size": int(hf_config.hidden_size),
        "intermediate_size": int(hf_config.intermediate_size),
        "num_attention_heads": int(heads),
        "num_key_value_heads": int(getattr(hf_config, "num_key_value_heads", heads)),
        "head_dim": int(getattr(hf_config, "head_dim", hf_config.hidden_size // heads)),
    }


def validate_layers(layers: list[int], base_count: int) -> list[int]:
    """Normalise the requested positions, or raise if they cannot be satisfied.

    Indices name positions in the *final* stack, which has `base_count + len(layers)` layers.
    A 24-layer base with `layers: [24]` therefore appends; `[24, 25]` appends two; `[12]`
    inserts one in the middle. Out-of-range is the important case to catch: unlike the adapter
    modules, where a bad index silently matches no modules, here it would mean building a stack
    with a hole in it.
    """
    if not layers:
        raise ValueError(
            "layer_expand.layers is empty -- name at least one position for a new block, "
            f"e.g. `layers: [{base_count}]` to append one to the end of this model."
        )

    unique = sorted(set(layers))
    if len(unique) != len(layers):
        raise ValueError(
            f"layer_expand.layers has repeated indices: {layers}. Each new block needs its own "
            "position; to add two blocks at the end of a "
            f"{base_count}-layer model write `layers: [{base_count}, {base_count + 1}]`."
        )

    final_count = base_count + len(unique)
    out_of_range = [i for i in unique if not 0 <= i < final_count]
    if out_of_range:
        raise ValueError(
            f"layer_expand.layers contains {out_of_range}, but expanding a {base_count}-layer "
            f"model with {len(unique)} new block(s) gives {final_count} layers, numbered "
            f"0-{final_count - 1}. Use `layers: [{base_count}]` to append one to the end."
        )
    return unique


def resolve_shape(expand_cfg: LayerExpandConfig, geometry: dict) -> dict:
    """The new block's shape, with unset fields inherited from the base model.

    Also reports whether the result is uniform, which is what decides between a plain appended
    `Qwen2DecoderLayer` and a `WrappedDecoderLayer` -- and therefore whether the saved model is
    loadable without this package.
    """
    shape = {
        "hidden_size": expand_cfg.hidden_size or geometry["hidden_size"],
        "intermediate_size": expand_cfg.intermediate_size or geometry["intermediate_size"],
        "num_attention_heads": expand_cfg.num_attention_heads or geometry["num_attention_heads"],
        "num_key_value_heads": (
            expand_cfg.num_key_value_heads or geometry["num_key_value_heads"]
        ),
        "head_dim": geometry["head_dim"],
        # The residual stream the block has to plug into, which is the base model's width
        # whatever the block's own width is. Only the wrapper uses it.
        "base_hidden_size": geometry["hidden_size"],
    }

    head_dim = shape["head_dim"]
    heads, kv_heads = shape["num_attention_heads"], shape["num_key_value_heads"]
    if heads % kv_heads != 0:
        raise ValueError(
            f"layer_expand: num_attention_heads ({heads}) must be a multiple of "
            f"num_key_value_heads ({kv_heads}) -- grouped-query attention splits the query "
            "heads evenly across the KV heads."
        )
    if shape["hidden_size"] != heads * head_dim:
        raise ValueError(
            f"layer_expand: hidden_size ({shape['hidden_size']}) must equal "
            f"num_attention_heads * head_dim ({heads} * {head_dim} = {heads * head_dim}). "
            f"head_dim is fixed at the base model's {head_dim} because the rotary tables and "
            "the KV cache are shared across the stack, so widen the block by raising "
            f"num_attention_heads: {shape['hidden_size'] // head_dim} would fit this "
            "hidden_size."
        )

    # Two different questions, and they have different answers.
    #
    # `wrapped` -- does the block need 896 -> d -> 896 projections to sit in the residual
    # stream? Only when its width actually differs. Widening the MLP alone leaves the block's
    # input and output at the base width, so it plugs straight in; wrapping it there would add
    # two useless 896x896 projections (1.6M parameters) and an information bottleneck for
    # nothing.
    #
    # `uniform` -- is the resulting stack describable by config.json, and therefore loadable by
    # a stock `AutoModelForCausalLM.from_pretrained`? Only when *nothing* differs, since
    # Qwen2Config carries one scalar per dimension for the whole stack. A block that is merely
    # MLP-widened is unwrapped but still non-uniform: it needs the sidecar to be rebuilt.
    shape["wrapped"] = shape["hidden_size"] != geometry["hidden_size"]
    shape["uniform"] = all(
        shape[key] == geometry[key]
        for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads")
    )
    return shape


def block_parameter_count(shape: dict) -> int:
    """Parameters one new block adds, including the wrapper projections when it is not uniform.

    Reported before training so the cost of a shape override is visible at launch rather than
    inferred from a checkpoint's file size afterwards.
    """
    d, inter = shape["hidden_size"], shape["intermediate_size"]
    head_dim = shape["head_dim"]
    q_dim = shape["num_attention_heads"] * head_dim
    kv_dim = shape["num_key_value_heads"] * head_dim

    attention = (d * q_dim + q_dim) + 2 * (d * kv_dim + kv_dim) + q_dim * d
    mlp = 3 * d * inter
    norms = 2 * d
    total = attention + mlp + norms
    if shape.get("wrapped"):
        total += 2 * shape["base_hidden_size"] * d  # in_proj + out_proj, both bias-free
    return total


# --------------------------------------------------------------------------------------
# Start points: the layer_lora adapter to build on, and the expanded run to continue
# --------------------------------------------------------------------------------------


def _is_disabled(spec: str | None) -> bool:
    return spec is None or spec.strip().lower() in DISABLED


def _checkpoint_number(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except ValueError:
        return -1


def latest_checkpoint_in(directory: Path, final_dir_name: str) -> Path | None:
    """The most advanced set of weights in a run directory, or None if there are none.

    `final_dir_name` (layer_lora's "adapter", this module's "model") wins over any numbered
    checkpoint when it exists: it is what a *completed* run writes, and with
    `training.early_stopping` on it holds the best checkpoint by eval loss rather than merely
    the last one. Only when a run never finished does the highest `checkpoint-N` stand in.
    """
    if not directory.is_dir():
        return None

    final = directory / final_dir_name
    if (final / "adapter_config.json").is_file() or any(
        (final / name).is_file() for name in _WEIGHT_FILES + _INDEX_FILES
    ):
        return final

    checkpoints = [
        path
        for path in directory.glob("checkpoint-*")
        if path.is_dir() and _checkpoint_number(path) >= 0
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_number)


def resolve_one_adapter(spec: str) -> Path:
    """Resolve one `base_adapters` entry to an adapter directory.

    An entry may be either the adapter itself (a directory holding adapter_config.json) or the
    run directory it lives in, in which case its latest weights are taken. Both spellings are
    accepted because the two runs in this repo are laid out differently: sft-ds-assistant has a
    finished `adapter/`, sft-layer-lora has only `checkpoint-N` directories.
    """
    candidate = Path(spec)
    if (candidate / "adapter_config.json").is_file():
        return candidate

    if not candidate.is_dir():
        raise FileNotFoundError(
            f"layer_expand.base_adapters: {candidate} does not exist. Each entry is an adapter "
            "directory (one with adapter_config.json in it) or the run directory holding one, "
            "e.g. outputs/sft-ds-assistant or outputs/sft-layer-lora/checkpoint-16000."
        )

    found = latest_checkpoint_in(candidate, "adapter")
    if found is None or not (found / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"layer_expand.base_adapters: {candidate} is a directory but holds no adapter -- "
            "no adapter/ and no checkpoint-N with an adapter_config.json in it. Point at a "
            "finished run, or drop the entry."
        )
    return found


def resolve_base_adapters(expand_cfg: LayerExpandConfig) -> list[Path]:
    """Resolve every `base_adapters` entry, in order. Empty means expand the stock base model.

    Order is load order and it matters: each adapter is merged on top of the result of the
    previous one, so the broad adapter goes first and the layer-scoped one that refines it goes
    last.
    """
    specs = [s for s in (expand_cfg.base_adapters or []) if not _is_disabled(s)]
    if not specs:
        print("base_adapters: empty -- expanding the stock base model.")
        return []

    resolved = [resolve_one_adapter(spec) for spec in specs]
    for position, (spec, path) in enumerate(zip(specs, resolved), start=1):
        described = describe_adapter(path)
        suffix = f" (from {spec})" if str(path) != spec else ""
        print(f"base_adapters[{position}]: {path}{suffix} -- {described}")
    return resolved


def describe_adapter(path: Path) -> str:
    """One line saying what an adapter actually covers, read from its config.

    Printed at launch because "which layers does this run really carry" is the thing that is
    easy to get wrong when chaining: a layer-scoped adapter names one layer and leaves the other
    23 at whatever was underneath it.
    """
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return "no adapter_config.json"
    data = json.loads(config_path.read_text())
    layers = data.get("layers_to_transform")
    if layers is None:
        scope = "all layers"
    elif isinstance(layers, int):
        scope = f"layer {layers}"
    else:
        scope = f"layer{'s' if len(layers) > 1 else ''} {layers}"
    modules = sorted(data.get("target_modules") or [])
    return f"{data.get('peft_type', 'unknown')} on {scope}, {len(modules)} projection(s)"


def resolve_init_checkpoint(expand_cfg: LayerExpandConfig, output_dir: Path) -> Path | None:
    """The expanded model to continue from, or None to build a fresh block.

    "auto" looks only at `<output_dir>/model` -- the weights a *completed* run wrote. It
    deliberately ignores `checkpoint-N` in the same directory, because those belong to an
    interrupted run that `run_sft_core` restores properly on its own, bringing optimizer state,
    LR schedule position and step count with it. Reloading the weights alone here would throw
    all three away and silently restart the schedule from step 0.
    """
    spec = expand_cfg.init_from_checkpoint
    if _is_disabled(spec):
        return None

    if spec.strip().lower() == "auto":
        candidate = output_dir / FINAL_DIR_NAME
        if not any((candidate / name).is_file() for name in _WEIGHT_FILES + _INDEX_FILES):
            print(
                f"init_from_checkpoint: auto -- no finished run at {candidate}. "
                "Building a fresh block (an interrupted run in this directory, if any, "
                "still resumes normally)."
            )
            return None
    else:
        candidate = Path(spec)
        if not candidate.is_dir():
            raise FileNotFoundError(
                f"layer_expand.init_from_checkpoint: {candidate} is not a directory."
            )
        if not any((candidate / name).is_file() for name in _WEIGHT_FILES + _INDEX_FILES):
            raise FileNotFoundError(
                f"layer_expand.init_from_checkpoint: no model weights in {candidate}. This "
                "path wants an expanded *model* directory (safetensors), not a LoRA adapter -- "
                "adapters to merge into the base go in layer_expand.base_adapters instead."
            )

    print(f"Continuing the expanded model from {candidate}.")
    return candidate


def merge_base_adapters(base_model, adapter_dirs: list[Path]):
    """Fold each adapter into the base weights in turn, returning a plain transformer.

    Merged rather than attached, and merged in sequence. The adapters' contribution belongs to
    the *frozen* part of this run -- it is the tuned model the new block is trained on top of --
    so baking it into the weights keeps it out of the optimizer, keeps `named_parameters()` free
    of PEFT's wrappers, and lets the expanded model save as an ordinary transformer.

    Both adapter types in this repo merge: LoRA folds `B @ A` into the projection weight, and
    TinyLoRA's layers implement `merge`/`get_delta_weight` so `BaseTuner.merge_and_unload`
    handles it too. After each step the result is a plain model again, which is what lets the
    next adapter be loaded over it.
    """
    from peft import PeftModel

    model = base_model
    for adapter_dir in adapter_dirs:
        peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
        try:
            model = peft_model.merge_and_unload()
        except (AttributeError, NotImplementedError, TypeError) as exc:
            peft_type = "unknown"
            config_path = adapter_dir / "adapter_config.json"
            if config_path.is_file():
                peft_type = json.loads(config_path.read_text()).get("peft_type", "unknown")
            raise ValueError(
                f"The {peft_type} adapter at {adapter_dir} cannot be merged into the base "
                f"weights ({exc}). Layer expansion needs a plain transformer to splice blocks "
                "into, so every adapter in base_adapters has to fold in first. Drop this entry, "
                "or replace it with a mergeable adapter."
            ) from exc
        print(f"Merged {adapter_dir}; frozen from here on.")

    return model


# --------------------------------------------------------------------------------------
# Building the expanded stack
# --------------------------------------------------------------------------------------


def _decoder_stack(model) -> nn.ModuleList:
    """The `nn.ModuleList` of decoder blocks, wherever this architecture keeps it."""
    for path in (("model", "layers"), ("transformer", "h"), ("model", "decoder", "layers")):
        node = model
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList):
            return node
    raise ValueError(
        f"Could not find the decoder stack on {type(model).__name__}. Layer expansion needs a "
        "model whose blocks live in an nn.ModuleList (model.layers for Qwen/Llama/Mistral)."
    )


def _attention_of(block: nn.Module):
    """The attention submodule of a block, seeing through the width wrapper."""
    inner = getattr(block, "inner", block)
    return getattr(inner, "self_attn", None)


def _build_inner_config(model_config, shape: dict, layer_index: int):
    """A throwaway config describing one block's shape, for constructing it.

    Never serialised -- it exists only so the stock `Qwen2DecoderLayer.__init__` reads the new
    block's dimensions out of it instead of the model's. `head_dim` is pinned explicitly because
    Qwen2Config has no such field and the constructor otherwise derives it as
    `hidden_size // num_attention_heads`, which would silently change it along with the width.
    """
    inner = copy.deepcopy(model_config)
    inner.hidden_size = shape["hidden_size"]
    inner.intermediate_size = shape["intermediate_size"]
    inner.num_attention_heads = shape["num_attention_heads"]
    inner.num_key_value_heads = shape["num_key_value_heads"]
    inner.head_dim = shape["head_dim"]

    # `Qwen2Attention.__init__` indexes config.layer_types[layer_idx], so it has to reach.
    layer_types = list(getattr(inner, "layer_types", None) or [])
    fill = layer_types[-1] if layer_types else "full_attention"
    while len(layer_types) <= layer_index:
        layer_types.append(fill)
    inner.layer_types = layer_types
    return inner


def _init_new_block(block: nn.Module, init: str, init_std: float) -> None:
    """Initialise a freshly constructed block, identity by default.

    Every `nn.Linear` starts from N(0, init_std) -- `nn.Linear`'s own kaiming-uniform default is
    tuned for a different scale than transformer blocks expect, and the block is not being built
    through `from_pretrained`, so HF's `_init_weights` never runs on it. RMSNorm weights are
    left at the ones their constructor set.

    Then, for "identity", the output projections are zeroed. Both residual branches
    (`residual + self_attn(...)` and `residual + mlp(...)`) contribute exactly zero, the block
    returns its input unchanged, and the expanded model's logits are identical to the base
    model's on the first forward pass. Training moves away from there. Without this the block
    injects noise directly into the layer feeding `lm_head`, and the run starts by repairing
    damage it caused itself.
    """
    if init not in ("identity", "random"):
        raise ValueError(
            f"layer_expand.init: {init!r} is not a known initialisation. Use \"identity\" "
            "(zeroed output projections, so the new block starts as a residual passthrough) "
            "or \"random\"."
        )

    for module in block.modules():
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=init_std)
            if module.bias is not None:
                module.bias.data.zero_()

    if init == "random":
        return

    zeroed = []
    for name, module in block.named_modules():
        # o_proj and down_proj are the exits of the attention and MLP branches; out_proj is the
        # width wrapper's exit. Zeroing whichever are present makes the block an identity.
        if isinstance(module, nn.Linear) and name.split(".")[-1] in ("o_proj", "down_proj", "out_proj"):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
            zeroed.append(name or "out_proj")
    if not zeroed:
        raise ValueError(
            "layer_expand.init: \"identity\" found no output projection to zero on the new "
            "block, so it cannot be made a residual passthrough. This architecture does not "
            "name its projections o_proj/down_proj; use init: \"random\" and a low learning "
            "rate instead."
        )


def expand_model(model, layers: list[int], shape: dict, init: str, init_std: float):
    """Splice new blocks into `model` at `layers` and return it.

    Mutates in place: the stack grows, `config.num_hidden_layers` and `config.layer_types` grow
    with it, and every block's `layer_idx` is renumbered. That last step is what makes insertion
    anywhere other than the end safe -- `layer_idx` is the block's index into the KV cache, so
    leaving the blocks after an insertion point with their old indices would have two blocks
    reading and writing the same cache slot.
    """
    stack = _decoder_stack(model)
    reference = stack[0]
    layer_cls = type(reference)
    template = next(reference.parameters())
    device, dtype = template.device, template.dtype

    for index in layers:  # ascending, so each insertion lands at its final position
        inner_config = _build_inner_config(model.config, shape, index)
        block = layer_cls(inner_config, index)
        if shape["wrapped"]:
            block = WrappedDecoderLayer(
                block, hidden_size=shape["base_hidden_size"], inner_size=shape["hidden_size"]
            )
        _init_new_block(block, init, init_std)
        stack.insert(index, block.to(device=device, dtype=dtype))

    model.config.num_hidden_layers = len(stack)
    layer_types = list(getattr(model.config, "layer_types", None) or [])
    if layer_types:
        fill = layer_types[-1]
        for index in layers:
            layer_types.insert(index, fill)
        model.config.layer_types = layer_types

    # Renumber after all insertions, not during: an index is only final once the stack is.
    for position, block in enumerate(stack):
        attention = _attention_of(block)
        if attention is None:
            continue
        attention.layer_idx = position
        if layer_types:
            attention.layer_type = layer_types[position]

    decoder = getattr(model, "model", model)
    if hasattr(decoder, "has_sliding_layers"):
        decoder.has_sliding_layers = "sliding_attention" in (layer_types or [])
    return model


def freeze_except(model, layers: list[int]) -> tuple[int, int]:
    """Freeze the whole model, then unfreeze the new blocks. Returns (trainable, total).

    Freezing is what turns the frozen stack into a pure forward pass: with the embeddings and
    every earlier block frozen, their outputs carry `requires_grad=False`, so autograd builds no
    graph and stores no activations until the first trainable block. When that block is last,
    the backward pass touches it, the final norm, and `lm_head` -- nothing else.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    stack = _decoder_stack(model)
    for index in layers:
        for parameter in stack[index].parameters():
            parameter.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable == 0:
        raise ValueError(
            f"Nothing to train: blocks {layers} hold no parameters after freezing. "
            "Check layer_expand.layers against the expanded stack."
        )
    return trainable, total


# --------------------------------------------------------------------------------------
# Persisting the shape, and reloading trained weights
# --------------------------------------------------------------------------------------


def expansion_spec(model_cfg: ModelConfig, expand_cfg: LayerExpandConfig, layers, shape, geometry):
    """What a later run (or `push_to_hub`, or an eval) needs in order to rebuild this stack."""
    return {
        "base_model": model_cfg.model_name_or_path,
        "base_num_hidden_layers": geometry["num_hidden_layers"],
        "layers": layers,
        "shape": {
            key: value for key, value in shape.items() if key not in ("uniform", "wrapped")
        },
        "uniform": shape["uniform"],
        "wrapped": shape["wrapped"],
        "init": expand_cfg.init,
        "block_parameters": block_parameter_count(shape),
    }


def write_sidecar(directory: Path, spec: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SIDECAR_NAME).write_text(json.dumps(spec, indent=2) + "\n")


def read_sidecar(directory: Path) -> dict | None:
    path = directory / SIDECAR_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def check_sidecar_matches(output_dir: Path, spec: dict) -> None:
    """Refuse to write a differently-shaped run into a directory that already holds one.

    The checkpoints in `output_dir` were written by a stack of a particular shape, and the
    Trainer resumes from them by name without checking. Changing `layers` or any dimension and
    re-running the same config would either fail deep inside `load_state_dict` with a shape
    mismatch, or -- where the shapes happen to line up -- quietly train a different architecture
    while reporting the old run's step count.
    """
    previous = read_sidecar(output_dir)
    if previous is None:
        return

    for key in ("layers", "shape", "base_model"):
        if previous.get(key) != spec.get(key):
            raise ValueError(
                f"{output_dir} holds a run with {key}={previous.get(key)!r}, but this config "
                f"asks for {key}={spec.get(key)!r}. Checkpoints in that directory belong to the "
                "old shape and cannot be resumed into the new one. Point "
                "training.output_dir somewhere else, or delete the old run."
            )


def _load_one_weight_file(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path))
    return torch.load(str(path), map_location="cpu", weights_only=True)


def read_state_dict(directory: Path) -> dict:
    """Read a saved model's weights, sharded or not."""
    for name in _INDEX_FILES:
        index = directory / name
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            state: dict = {}
            for shard in sorted(set(weight_map.values())):
                state.update(_load_one_weight_file(directory / shard))
            return state

    for name in _WEIGHT_FILES:
        path = directory / name
        if path.is_file():
            return _load_one_weight_file(path)

    raise FileNotFoundError(f"No model weights in {directory}.")


def load_expanded_weights(model, checkpoint: Path, layers: list[int]) -> None:
    """Load a previous expanded run's weights over the freshly built stack.

    `strict=False` because tied embeddings mean `lm_head.weight` is normally absent from the
    saved state dict -- it shares storage with `embed_tokens.weight`, which is present. Any
    *other* missing key on a new block is an error, though: it would mean the block silently
    kept its freshly initialised weights (all-zero output projections, under identity init)
    while the run reported that it had continued from a checkpoint.
    """
    state = read_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)

    new_prefixes = tuple(f"model.layers.{index}." for index in layers)
    missing_new = [key for key in missing if key.startswith(new_prefixes)]
    if missing_new:
        raise ValueError(
            f"{checkpoint} is missing {len(missing_new)} weight(s) for the new block(s), "
            f"starting with {missing_new[0]!r}. It was written by a differently-shaped "
            "expansion. Check layer_expand.layers and the dimension overrides against "
            f"{checkpoint / SIDECAR_NAME}, or drop init_from_checkpoint to start fresh."
        )
    if unexpected:
        print(f"Ignored {len(unexpected)} unexpected key(s) in {checkpoint}, e.g. {unexpected[0]}.")

    model.tie_weights()
    print(f"Loaded {len(state)} tensor(s) from {checkpoint}.")


def describe_expansion(model, layers: list[int], shape: dict) -> str:
    stack = _decoder_stack(model)
    pieces = []
    for index in layers:
        count = sum(p.numel() for p in stack[index].parameters())
        kind = "wrapped " if isinstance(stack[index], WrappedDecoderLayer) else ""
        pieces.append(f"layer {index} ({kind}{shape['hidden_size']}d/{shape['intermediate_size']}, {count:,} params)")
    return ", ".join(pieces)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def load_layer_expand_model(
    model_cfg: ModelConfig,
    expand_cfg: LayerExpandConfig,
    output_dir: Path,
):
    """Build the expanded model with exactly `expand_cfg.layers` trainable.

    Returns `(model, init_checkpoint)`. `init_checkpoint` is the directory the expanded weights
    were loaded from, or None when a fresh block was built -- the caller needs it because
    starting from a named checkpoint and letting the Trainer resume an interrupted run in
    `output_dir` are mutually exclusive, and the explicit setting has to win.

    The order is the point of this function:
      1. resolve and merge every `base_adapters` entry into the base weights, in order,
      2. splice in the new block(s), identity-initialised,
      3. load a previous expanded run's weights over the top if there is one,
      4. freeze everything except the new block(s).

    Step 1 is a chain rather than a single adapter because the tuning in this repo is split
    across two runs: the TinyLoRA one covers all 24 layers and the layer_lora one refines a
    single layer on top of it. Merging only the second would silently reset the other 23 layers
    to the stock base weights.
    """
    geometry = base_model_geometry(model_cfg)
    layers = validate_layers(expand_cfg.layers, geometry["num_hidden_layers"])
    shape = resolve_shape(expand_cfg, geometry)
    spec = expansion_spec(model_cfg, expand_cfg, layers, shape, geometry)
    check_sidecar_matches(output_dir, spec)

    init_checkpoint = resolve_init_checkpoint(expand_cfg, output_dir)
    # An interrupted run in output_dir is restored by the Trainer, which overwrites every weight
    # in the model -- so the base adapter would be merged only to be thrown away. Detect that
    # case here and skip the merge rather than paying for it.
    from tiny_lora.train_sft import resolve_resume_checkpoint

    will_resume = (
        init_checkpoint is None
        and output_dir.is_dir()
        and resolve_resume_checkpoint(output_dir) is not None
    )
    if init_checkpoint or will_resume:
        adapter_dirs: list[Path] = []
        if expand_cfg.base_adapters:
            print(
                "Skipping the base_adapters merge: this run continues from expanded weights "
                "that already contain it."
            )
    else:
        adapter_dirs = resolve_base_adapters(expand_cfg)

    base_model = load_base_model(model_cfg)
    if model_cfg.load_in_4bit and _bitsandbytes_available():
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(base_model)

    if adapter_dirs:
        base_model = merge_base_adapters(base_model, adapter_dirs)

    model = expand_model(base_model, layers, shape, expand_cfg.init, expand_cfg.init_std)
    if init_checkpoint is not None:
        load_expanded_weights(model, init_checkpoint, layers)

    trainable, total = freeze_except(model, layers)
    write_sidecar(output_dir, {**spec, "base_adapters": [str(d) for d in adapter_dirs]})

    print(
        f"Expanded {geometry['num_hidden_layers']} -> {model.config.num_hidden_layers} layers: "
        f"{describe_expansion(model, layers, shape)}."
    )
    print(
        f"Trainable {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%) -- "
        f"the other {total - trainable:,} are frozen."
    )
    if not shape["uniform"]:
        print(
            "This stack is non-uniform, so config.json alone cannot describe it: reload it "
            f"through layer_expand (it reads {output_dir / SIDECAR_NAME}), not through a plain "
            "AutoModelForCausalLM.from_pretrained."
        )
    return model, init_checkpoint


# --------------------------------------------------------------------------------------
# Inference: loading an expanded model back
# --------------------------------------------------------------------------------------

# Written into a *wrapped* stack's config.json in place of "qwen2". A wrapped block's weights
# are named `model.layers.N.inner.*` plus the two width projections, which a stock
# `AutoModelForCausalLM.from_pretrained` does not recognise -- it reports them as UNEXPECTED,
# finds the plain `model.layers.N.*` names MISSING, randomly initialises the whole block, and
# returns a model that loads cleanly and produces garbage. Retyping the checkpoint turns that
# silent corruption into an immediate "Transformers does not recognize this architecture".
# Only `model_type` has this effect; `architectures` is metadata the dispatcher ignores.
EXPANDED_MODEL_TYPE = "layer_expand"


def shape_from_spec(spec: dict) -> dict:
    """Rebuild the shape dict `expand_model` wants from a sidecar's contents."""
    return {**spec["shape"], "uniform": spec["uniform"], "wrapped": spec["wrapped"]}


def is_expanded_model_dir(path: str | Path) -> bool:
    """True when `path` holds a layer_expand run's weights rather than an adapter."""
    return (Path(path) / SIDECAR_NAME).is_file()


def expanded_base_model(path: str | Path) -> str:
    """The base model id an expanded checkpoint was built from, out of its sidecar."""
    spec = read_sidecar(Path(path))
    if spec is None:
        raise FileNotFoundError(
            f"No {SIDECAR_NAME} in {path}, so the stack it describes cannot be rebuilt. "
            "Point at a layer_expand run's model/ or checkpoint-N/ directory."
        )
    base = spec.get("base_model")
    if not base:
        raise ValueError(f"{Path(path) / SIDECAR_NAME} has no base_model.")
    return base


def stamp_config(directory: Path, wrapped: bool) -> None:
    """Retype a wrapped stack's saved config.json so stock loaders refuse it.

    Only wrapped stacks need this. A uniform stack is a genuine Qwen2 and should stay loadable
    by anything; an MLP-widened one already fails loudly on a shape mismatch. It is the wrapped
    case alone that loads silently and wrongly, so it is the only one retyped -- keeping the
    blast radius of a non-standard model_type as small as possible.

    Applied by rewriting the file rather than by setting `config.model_type`, because
    `PretrainedConfig.to_dict` writes the *class* attribute and ignores an instance override.
    """
    if not wrapped:
        return
    config_path = directory / "config.json"
    if not config_path.is_file():
        return
    data = json.loads(config_path.read_text())
    if data.get("model_type") == EXPANDED_MODEL_TYPE:
        return
    data["base_model_type"] = data.get("model_type")
    data["model_type"] = EXPANDED_MODEL_TYPE
    data["architectures"] = ["LayerExpandForCausalLM"]
    config_path.write_text(json.dumps(data, indent=2) + "\n")


def load_expanded_model(
    path: str | Path,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
    eval_mode: bool = True,
):
    """Load a trained expanded model for inference.

    The architecture is rebuilt from the sidecar rather than read from config.json, which is
    what makes a non-uniform stack loadable at all: the base model supplies the frozen stack's
    shape, the sidecar supplies the new blocks', and the checkpoint's weights are then loaded
    over the whole thing -- including the base layers, which already carry whatever
    `base_adapters` were merged into them at training time.

    The blocks are built identity-initialised and immediately overwritten, so the init setting
    is irrelevant here; `load_expanded_weights` refuses the load if any new-block weight is
    missing, which is what stops a mismatched checkpoint from leaving zeroed projections behind.
    """
    path = Path(path)
    spec = read_sidecar(path)
    if spec is None:
        raise FileNotFoundError(
            f"No {SIDECAR_NAME} in {path}. Inference on an expanded model needs the sidecar "
            "written next to its weights -- it carries the new blocks' shapes, which "
            "config.json cannot express. It is written to the run's output_dir and copied into "
            "model/ and each checkpoint-N/; copy it alongside if you moved the weights."
        )

    if load_in_4bit:
        # The base would load as quantized parameters, and copying a bf16 state dict into them
        # silently does the wrong thing. Expanded models are small enough that this costs little.
        print("layer_expand: 4-bit loading is not supported for expanded models; using bf16.")
        load_in_4bit = False

    layers = spec["layers"]
    shape = shape_from_spec(spec)
    base_model = load_base_model(
        ModelConfig(
            model_name_or_path=spec["base_model"],
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
    )
    model = expand_model(base_model, layers, shape, "identity", 0.0)
    load_expanded_weights(model, path, layers)
    if eval_mode:
        model.eval()
    print(
        f"Loaded expanded model from {path}: {model.config.num_hidden_layers} layers, "
        f"new block(s) at {layers} ({'wrapped ' if shape['wrapped'] else ''}"
        f"{shape['hidden_size']}d/{shape['intermediate_size']})."
    )
    return model
