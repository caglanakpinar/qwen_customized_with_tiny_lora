"""Layer-scoped LoRA adapter configuration.

The only thing that distinguishes this from `standard_lora` is `layers`: standard LoRA adapts
every transformer layer that carries a target module, this adapts exactly the layers named in
the config and leaves every other layer at its base weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayerLoraConfig:
    # Which transformer layers get an adapter, by index (0-based, as they are numbered in
    # `model.layers.N`). Empty means "no layers", which is never what anyone wants, so
    # `validate_layers` rejects it rather than training an adapter with zero parameters.
    layers: list[int] = field(default_factory=list)
    # Defaults to the attention projections only -- q/k/v is the subset this module exists to
    # target. Add o_proj/gate_proj/up_proj/down_proj here to widen it.
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj"])
    # The module-name segment that precedes the layer index, i.e. the `layers` in
    # `model.layers.7.self_attn.q_proj`. PEFT can infer this when it is None, but its fallback
    # regex matches the first numbered segment in *any* name; naming it is exact and costs
    # nothing. Qwen/Llama/Mistral all use "layers".
    layers_pattern: str = "layers"
    r: int = 8                    # LoRA rank -- width of the trainable A/B low-rank update
    lora_alpha: int = 16          # update is scaled by lora_alpha / r (or / sqrt(r) with use_rslora)
    lora_dropout: float = 0.0     # dropout applied to the LoRA path during training
    bias: str = "none"            # "none", "all", or "lora_only" -- which biases also train
    use_rslora: bool = False      # rank-stabilized scaling (alpha / sqrt(r) instead of alpha / r)
    # Where the adapter weights start from, instead of LoRA's usual zero-init:
    #   None     -- fresh adapter, train the chosen layers from the base model
    #   "auto"   -- continue from the newest adapter under training.output_dir if there is one,
    #               otherwise behave as None. Lets the same config be re-run to extend a run.
    #   <path>   -- continue from that specific saved adapter / checkpoint-N directory
    # Only LoRA adapters can be loaded here, and only ones covering the same `layers` -- see
    # `resolve_init_checkpoint`, which refuses the alternatives instead of silently training
    # something other than what the config asks for.
    init_from_checkpoint: str | None = None
