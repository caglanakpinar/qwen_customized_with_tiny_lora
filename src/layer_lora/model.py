"""Model loading with layer-scoped adapters.

There are two ways to end up training exactly the layers named in the config, and this module
does both:

*Fresh* -- build a new adapter that only exists on those layers. PEFT already knows how to do
this (`LoraConfig`'s `layers_to_transform`/`layers_pattern`); the work here is driving it from
a config file and validating the layer indices against the model actually being loaded, since a
typo'd index otherwise surfaces as an opaque "target modules not found" error deep inside PEFT.

*Continue* -- `init_from_checkpoint` points at an already-trained adapter that covers more
layers than the config names (typically every layer). The whole adapter is loaded, every
adapter parameter outside `layers` is frozen, and training moves only the named layers. The
adapter saved at the end still carries all of its layers -- the named ones newly fine-tuned,
the rest exactly as the checkpoint had them.

The continue path is adapter-type agnostic: it works on a LoRA checkpoint and on a TinyLoRA
one, which matters because TinyLoRA stores its trainable vectors on the *model* rather than in
the layer modules, so "freeze everything but layer N" cannot be done by parameter name alone.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import torch
from peft import get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoConfig

from layer_lora.config import LayerLoraConfig
from tiny_lora.config import ModelConfig
from tiny_lora.model import _bitsandbytes_available, load_base_model

# Adapter types whose weights the continue path knows how to load and freeze per layer.
_RESUMABLE_PEFT_TYPES = ("LORA", "TINYLORA")


def model_num_layers(model_cfg: ModelConfig) -> int:
    """Transformer layer count, read from the model's config without loading any weights.

    Read up front so an out-of-range `layers:` entry fails in seconds, rather than after the
    minutes it takes to pull and materialise the base model.
    """
    hf_config = AutoConfig.from_pretrained(
        model_cfg.model_name_or_path,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    num_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError(
            f"{model_cfg.model_name_or_path} reports no num_hidden_layers, so layer indices "
            "cannot be validated against it. This module targets standard decoder stacks "
            "(Qwen, Llama, Mistral, ...)."
        )
    return int(num_layers)


def validate_layers(layers: list[int], num_layers: int) -> list[int]:
    """Return `layers` normalised (deduplicated, sorted), or raise if it is unusable.

    Out-of-range indices are the important case: PEFT matches layer indices by regex against
    module names, so an index no layer has simply matches nothing. The adapter is then built
    with fewer modules than asked for -- or none at all -- and training "succeeds" having
    learned nothing about the layer the config named.
    """
    if not layers:
        raise ValueError(
            "layer_lora.layers is empty -- name at least one transformer layer to adapt, "
            f"e.g. `layers: [{num_layers - 1}]` for the last layer of this model."
        )

    out_of_range = sorted({index for index in layers if not 0 <= index < num_layers})
    if out_of_range:
        raise ValueError(
            f"layer_lora.layers contains {out_of_range}, but this model has {num_layers} "
            f"layers, numbered 0-{num_layers - 1}."
        )
    return sorted(set(layers))


def _read_adapter_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {checkpoint_dir}. layer_lora.init_from_checkpoint "
            "wants a saved adapter directory, e.g. outputs/sft-layer-lora/adapter or "
            "outputs/sft-layer-lora/checkpoint-500."
        )
    return json.loads(config_path.read_text())


def resolve_init_checkpoint(
    spec: str | None,
    output_dir: Path,
    layers: list[int],
    target_modules: list[str],
) -> str | None:
    """Resolve `init_from_checkpoint` to an adapter directory to start training from.

    "auto" looks for `<output_dir>/adapter` -- the adapter a *previous, completed* run wrote.
    It deliberately ignores `checkpoint-N` directories in the same place: those belong to an
    interrupted run, which `run_sft_core` restores properly on its own (optimizer state, LR
    schedule position and step count included) when no explicit checkpoint is named here.

    The checkpoint has to *cover* what the config asks for, but it does not have to match it:
    continuing from an adapter trained on all 24 layers in order to move only layer 23 is the
    point of this path. What is refused is a checkpoint that is missing a layer or a projection
    the config names, since nothing would then be training for it -- the run would report
    success having left that layer untouched.
    """
    if spec is None:
        return None

    if spec == "auto":
        candidate = output_dir / "adapter"
        if not (candidate / "adapter_config.json").is_file():
            print(
                f"init_from_checkpoint: auto -- no previous adapter at {candidate}, "
                "starting from the base model."
            )
            return None
    else:
        candidate = Path(spec)
        if not candidate.is_dir():
            raise FileNotFoundError(f"layer_lora.init_from_checkpoint: {candidate} is not a directory.")

    saved = _read_adapter_config(candidate)

    peft_type = saved.get("peft_type")
    if peft_type not in _RESUMABLE_PEFT_TYPES:
        raise ValueError(
            f"{candidate} holds a {peft_type} adapter. Continuing is supported for "
            f"{' and '.join(_RESUMABLE_PEFT_TYPES)} adapters; point init_from_checkpoint at "
            "one of those, or drop the setting to train a fresh adapter on `layers`."
        )

    # `layers_to_transform: null` is PEFT's "every layer", which covers any selection.
    saved_layers = saved.get("layers_to_transform")
    if isinstance(saved_layers, int):
        saved_layers = [saved_layers]
    if isinstance(saved_layers, list):
        missing = sorted(set(layers) - set(saved_layers))
        if missing:
            raise ValueError(
                f"{candidate} adapts layers {sorted(saved_layers)}, which does not cover "
                f"{missing}. The saved adapter's own config wins when it is loaded, so there "
                "would be nothing on those layers to train. Narrow layer_lora.layers to the "
                "checkpoint's, or drop init_from_checkpoint to start a fresh adapter."
            )

    saved_modules = set(saved.get("target_modules") or [])
    missing_modules = sorted(set(target_modules) - saved_modules)
    if missing_modules:
        raise ValueError(
            f"{candidate} adapts modules {sorted(saved_modules)}, which does not cover "
            f"{missing_modules}. Narrow layer_lora.target_modules to the checkpoint's, or "
            "drop init_from_checkpoint to start a fresh adapter."
        )

    print(f"Continuing from the {peft_type} adapter at {candidate}.")
    return str(candidate)


def _split_adapted_name(name: str, layers_pattern: str) -> tuple[int | None, str]:
    """Split a module path into (transformer layer index, projection name).

    `base_model.model.model.layers.23.self_attn.q_proj` -> `(23, "q_proj")`. The index is the
    segment immediately after `layers_pattern`, and None when the module sits outside the
    decoder stack (an embedding adapter, say), which the layer filter then leaves alone.
    """
    parts = name.split(".")
    for position, part in enumerate(parts[:-1]):
        if part == layers_pattern and parts[position + 1].isdigit():
            return int(parts[position + 1]), parts[-1]
    return None, parts[-1]


def _tuner_layer_parameters(module) -> list[torch.nn.Parameter]:
    """The trainable adapter parameters belonging to one adapted module.

    TinyLoRA is the reason this is not just `module.parameters()`: its trainable vectors live in
    a single model-level `tinylora_v` ModuleDict that every adapted module holds a reference to,
    so walking the module would hand back every layer's vector rather than this layer's. The
    per-layer `_tinylora_v_ref` is the direct handle, and is a plain dict precisely so PyTorch
    does not register those shared parameters on each layer. LoRA and the rest keep their
    trainable tensors inside the module, where `adapter_layer_names` finds them.
    """
    tinylora_ref = getattr(module, "_tinylora_v_ref", None)
    if isinstance(tinylora_ref, dict):
        return [p for p in tinylora_ref.values() if isinstance(p, torch.nn.Parameter)]

    params: list[torch.nn.Parameter] = []
    for attr in getattr(module, "adapter_layer_names", ()):
        container = getattr(module, attr, None)
        if container is None:
            continue
        params.extend(param for _, param in container.named_parameters())
    return params


def restrict_training_to_layers(
    model,
    layers: list[int],
    target_modules: list[str],
    layers_pattern: str = "layers",
) -> dict[int, set[str]]:
    """Freeze every adapter parameter outside `layers`/`target_modules`; return what is left.

    Used on the continue path, where the loaded adapter spans more layers than the config asks
    for. The frozen parameters keep their trained values and are still written out by
    `save_model` -- freezing only stops the optimizer from moving them, which is what makes the
    saved adapter "all layers, one of them newly fine-tuned".
    """
    wanted_layers = set(layers)
    wanted_modules = set(target_modules)

    # Keyed by id() rather than by the parameter itself: `nn.Parameter` hashes by identity
    # anyway, but the dict also has to survive being read back for the weight-tying check below.
    keep: dict[int, torch.nn.Parameter] = {}
    freeze: dict[int, torch.nn.Parameter] = {}
    trained: dict[int, set[str]] = defaultdict(set)

    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        index, projection = _split_adapted_name(name, layers_pattern)
        if index is None:
            continue
        selected = index in wanted_layers and projection in wanted_modules
        bucket = keep if selected else freeze
        for param in _tuner_layer_parameters(module):
            bucket[id(param)] = param
        if selected:
            trained[index].add(projection)

    # One parameter serving both a selected and an unselected module means the adapter shares
    # weights across layers (TinyLoRA's `weight_tying`), and "train layer N only" is then not
    # expressible: moving it for layer N moves it everywhere.
    shared = set(keep) & set(freeze)
    if shared:
        raise ValueError(
            f"{len(shared)} adapter parameter(s) in this checkpoint are shared between "
            f"layers {sorted(trained)} and other layers, so training the named layers alone is "
            "not possible -- the same weights back the rest of the stack. This is what "
            "TinyLoRA's weight_tying > 0 does; continue from an untied checkpoint instead."
        )

    if not keep:
        raise ValueError(
            f"Nothing to train: no adapted module matched layers {sorted(wanted_layers)} with "
            f"projections {sorted(wanted_modules)}. Check layer_lora.layers, "
            "layer_lora.target_modules and layer_lora.layers_pattern against the checkpoint."
        )

    for param in freeze.values():
        param.requires_grad_(False)
    for param in keep.values():
        param.requires_grad_(True)

    print(
        f"Froze {len(freeze)} adapter tensor(s) outside the selection; "
        f"{len(keep)} remain trainable."
    )
    return dict(trained)


def build_layer_lora_config(cfg: LayerLoraConfig, layers: list[int]):
    from peft import LoraConfig as PeftLoraConfig

    return PeftLoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.target_modules,
        # The whole point of this module: only layers in this list get an adapter. Everything
        # else keeps its base weights and stays frozen.
        layers_to_transform=layers,
        layers_pattern=cfg.layers_pattern,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.bias,
        use_rslora=cfg.use_rslora,
        task_type="CAUSAL_LM",
    )


def describe_adapted_layers(model, layers_pattern: str = "layers") -> str:
    """Summarise which layers carry an adapter that is actually training, off the built model.

    Reported rather than assumed: this is what confirms the layer indices in the config met
    real modules, which is the failure this module is most exposed to. It reads `requires_grad`
    so it describes the same thing on both paths -- the layers a fresh adapter was built on, and
    the layers left unfrozen after continuing from a wider checkpoint.
    """
    adapted: dict[int, set[str]] = defaultdict(set)
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        index, projection = _split_adapted_name(name, layers_pattern)
        if index is None:
            continue
        if any(param.requires_grad for param in _tuner_layer_parameters(module)):
            adapted[index].add(projection)

    if not adapted:
        return "no layers were adapted"
    return ", ".join(
        f"layer {index}: {', '.join(sorted(modules))}" for index, modules in sorted(adapted.items())
    )


def load_layer_lora_model(
    model_cfg: ModelConfig,
    layer_cfg: LayerLoraConfig,
    output_dir: Path,
):
    """Load the base model and arrange for exactly `layer_cfg.layers` to train.

    Returns `(model, init_checkpoint)`. `init_checkpoint` is the adapter directory the weights
    came from, or None when a fresh adapter was built -- the caller needs it because starting
    from a named checkpoint and resuming an interrupted run in `output_dir` are mutually
    exclusive, and the explicit setting has to win.
    """
    layers = validate_layers(layer_cfg.layers, model_num_layers(model_cfg))
    init_checkpoint = resolve_init_checkpoint(
        layer_cfg.init_from_checkpoint, output_dir, layers, layer_cfg.target_modules
    )

    base_model = load_base_model(model_cfg)
    if model_cfg.load_in_4bit and _bitsandbytes_available():
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(base_model)

    if init_checkpoint is not None:
        from peft import PeftModel

        # is_trainable=True is what separates this from an inference load: without it PEFT
        # freezes the adapter it just restored, and training would update nothing at all. It
        # unfreezes *everything* the checkpoint holds, though, which is why the layer selection
        # is re-imposed immediately afterwards.
        model = PeftModel.from_pretrained(base_model, init_checkpoint, is_trainable=True)
        restrict_training_to_layers(
            model, layers, layer_cfg.target_modules, layer_cfg.layers_pattern
        )
    else:
        model = get_peft_model(base_model, build_layer_lora_config(layer_cfg, layers))

    print(f"Training {describe_adapted_layers(model, layer_cfg.layers_pattern)}.")
    return model, init_checkpoint
