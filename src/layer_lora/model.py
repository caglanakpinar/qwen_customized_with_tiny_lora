"""Model loading with layer-scoped LoRA adapters.

PEFT already knows how to restrict an adapter to particular layers -- `LoraConfig`'s
`layers_to_transform`/`layers_pattern`. This module's job is to drive that from a config file,
validate the layer indices against the model actually being loaded (a typo'd index otherwise
surfaces as an opaque "target modules not found" error deep inside PEFT), and optionally start
from an already-trained adapter instead of a zero-initialised one.
"""

from __future__ import annotations

import json
from pathlib import Path

from peft import get_peft_model
from transformers import AutoConfig

from layer_lora.config import LayerLoraConfig
from tiny_lora.config import ModelConfig
from tiny_lora.model import _bitsandbytes_available, load_base_model


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

    "auto" looks for `<output_dir>/adapter` -- the adapter a *previous, completed* run of this
    config wrote. It deliberately ignores `checkpoint-N` directories in the same place: those
    belong to an interrupted run, and `run_sft_core`'s own `resolve_resume_checkpoint` restores
    those properly (optimizer state, LR schedule position and step count included), which
    re-initialising from their weights alone would not.

    A checkpoint that does not match this config is refused rather than loaded, because
    `PeftModel.from_pretrained` takes the adapter's *saved* config as authoritative -- loading a
    mismatched one would train some other set of layers than the yaml asks for, and report
    success doing it.
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
    if peft_type != "LORA":
        raise ValueError(
            f"{candidate} holds a {peft_type} adapter, which cannot be resumed as LoRA. "
            "layer_lora trains LoRA adapters; point init_from_checkpoint at one of its own "
            "outputs, or drop the setting to start from the base model."
        )

    saved_layers = saved.get("layers_to_transform")
    saved_layers = sorted(saved_layers) if isinstance(saved_layers, list) else saved_layers
    if saved_layers != layers:
        raise ValueError(
            f"{candidate} adapts layers {saved_layers}, but this config asks for {layers}. "
            "The saved adapter's own config wins when it is loaded, so continuing from it "
            "would not train the layers configured here. Align layer_lora.layers with the "
            "checkpoint, or drop init_from_checkpoint to start a fresh adapter."
        )

    saved_modules = sorted(saved.get("target_modules") or [])
    if saved_modules != sorted(target_modules):
        raise ValueError(
            f"{candidate} adapts modules {saved_modules}, but this config asks for "
            f"{sorted(target_modules)}. Align layer_lora.target_modules with the checkpoint, "
            "or drop init_from_checkpoint to start a fresh adapter."
        )

    print(f"Continuing from the adapter at {candidate}.")
    return str(candidate)


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


def describe_adapted_layers(model) -> str:
    """Summarise which layers actually received an adapter, read back off the built model.

    Reported rather than assumed: this is what confirms the layer indices in the config met
    real modules, which is the failure this module is most exposed to.
    """
    adapted: dict[int, set[str]] = {}
    for name, _ in model.named_modules():
        if not name.endswith("lora_A"):
            continue
        parts = name.split(".")
        # `...layers.<index>....<module>.lora_A` -- the index follows the layers segment, and
        # the adapted projection is the module the lora_A container hangs off.
        for position, part in enumerate(parts[:-1]):
            if part.isdigit() and position > 0:
                adapted.setdefault(int(part), set()).add(parts[-2])
                break

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
    """Load the base model and attach a LoRA adapter covering only `layer_cfg.layers`."""
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
        # freezes the adapter it just restored, and training would update nothing at all.
        model = PeftModel.from_pretrained(base_model, init_checkpoint, is_trainable=True)
    else:
        model = get_peft_model(base_model, build_layer_lora_config(layer_cfg, layers))

    print(f"Adapted {describe_adapted_layers(model)}.")
    return model
