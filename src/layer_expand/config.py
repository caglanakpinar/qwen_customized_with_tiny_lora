"""Layer-expansion configuration.

Two groups of settings, and they answer different questions:

*Shape* (`layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`,
`num_key_value_heads`, `init`) -- what the new block looks like and where it sits.

*Start point* (`base_adapters`, `init_from_checkpoint`) -- which already-trained weights the run
builds on. This is the only thing carried over from the LoRA modules: there is no adapter here to
configure, but "continue from the runs I already finished" is still the normal way to launch one
of these, and it takes a chain rather than one adapter because the tuning is split across two.

Note on `null`: `_merge_dataclass` ignores None-valued yaml keys, so writing
`init_from_checkpoint: null` does not switch a default of "auto" back off. It therefore defaults
to None (disabled) and the shipped config turns it on explicitly, which keeps yaml `null` meaning
what it looks like it means. "none"/"off" also read as disabled, for overriding from the CLI.
`base_adapters` is a list, so an empty one is an unambiguous "merge nothing".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Spellings that turn a start-point setting off. Needed because the CLI cannot pass a yaml null.
DISABLED = frozenset({"none", "off", "null", ""})


@dataclass
class LayerExpandConfig:
    # Positions the new blocks occupy in the *final* stack, 0-based. A 24-layer base plus
    # `layers: [24]` gives a 25-layer model whose new block is last -- the cheapest case to
    # train, because the backward pass then stops at it and never enters the frozen stack.
    # An index inside the existing range inserts there instead, shifting the blocks after it
    # down; that costs a backward pass through everything above the insertion point.
    layers: list[int] = field(default_factory=lambda: [24])

    # Shape of the new block. None means "same as the base model's layers", which is the only
    # setting that stays loadable by a stock `AutoModelForCausalLM.from_pretrained` -- a
    # uniform stack is expressible in config.json, a non-uniform one is not. Overriding any of
    # these builds a wrapped block instead and writes a layer_expand.json sidecar that
    # `load_layer_expand_model` needs in order to rebuild it. See model.py.
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None

    # "identity" zeroes the block's output projections (o_proj, down_proj, and the wrapper's
    # out_proj), so the block is an exact residual passthrough at step 0 and the expanded model
    # starts at the base model's quality rather than at noise. This is what makes training a
    # freshly-added block converge; "random" is here to demonstrate that it does not.
    init: str = "identity"
    init_std: float = 0.02  # only used by init: "random"

    # Adapters folded into the frozen base weights before the new block is added, IN ORDER.
    # This is a chain, not a single adapter, because no one adapter here holds the whole model:
    #
    #   outputs/sft-ds-assistant   TinyLoRA, all 24 layers, all 7 projections  (672 tensors)
    #   outputs/sft-layer-lora     LoRA, layer 21 only, q/k/v                  (6 tensors)
    #
    # Merging only the layer_lora run would leave layers 0-20 and 22-23 at the *stock*
    # Qwen2.5-0.5B weights, throwing away everything the TinyLoRA run learned. Listing both
    # gives a frozen stack of: base + TinyLoRA(all layers) + LoRA(layer 21).
    #
    # Each entry is either an exact adapter directory (one containing adapter_config.json) or a
    # run directory, in which case its latest weights are taken -- the finished `adapter/` if
    # one exists, else the highest `checkpoint-N`. An empty list expands the stock base model.
    #
    # Caveat worth knowing: these adapters were each trained against the stock base, not against
    # one another, so stacking them is an approximation. The layer-21 LoRA delta was learned
    # assuming stock weights underneath it and here lands on TinyLoRA-modified ones. This is
    # normal practice for composing adapters and is usually fine, but it is why the order
    # matters -- broadest first, most specific last.
    base_adapters: list[str] = field(
        default_factory=lambda: ["outputs/sft-ds-assistant", "outputs/sft-layer-lora"]
    )

    # Where the *expanded* model's weights start from:
    #   None/"none" -- always build a fresh block, ignoring any finished run in output_dir
    #   "auto"      -- continue from <training.output_dir>/model if a previous run finished one
    #   <path>      -- continue from that saved model / checkpoint-N directory
    #
    # "auto" deliberately only looks at the finished `model/` directory. A run interrupted
    # partway leaves `checkpoint-N` behind, and the Trainer restores those itself -- optimizer
    # state, LR schedule position and step count included -- which is strictly better than
    # reloading the weights alone. Naming a checkpoint here turns that automatic resume off,
    # since the two would otherwise fight over which weights win.
    init_from_checkpoint: str | None = None
