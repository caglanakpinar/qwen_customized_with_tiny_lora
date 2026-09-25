"""Layer-growth configuration.

Shorter than `layer_expand.config.LayerExpandConfig` because there is only one "start point"
setting here, not two: `previous_checkpoint` names the finished `layer_expand`/`layer_grow`
checkpoint this round continues from, and its own sidecar already records every earlier round's
block(s) -- there is no separate `base_adapters` chain to configure, because whatever adapters the
first round merged are already baked into `previous_checkpoint`'s weights.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerGrowConfig:
    # The finished layer_expand or layer_grow checkpoint to grow from. Required: unlike
    # layer_expand, there is no "expand the stock base model" case here -- growing implies
    # something to grow. Either a directory holding layer_expand.json/layer_grow.json directly,
    # or the run directory containing one, in which case the latest is taken (the finished
    # `model/` if there is one, else the highest `checkpoint-N`) -- same resolution rule
    # layer_expand.base_adapters uses for adapter directories.
    previous_checkpoint: str = ""

    # Position(s) the new block(s) take in the *final* stack, 0-based. If the previous checkpoint
    # is a 25-layer model (base 24 + one grown block), `layers: [25]` appends one more.
    layers: list[int] | None = None

    # Shape of the new block. None on all four inherits the *previous round's own* shape -- the
    # sensible default when growing the same widened block further rather than changing tack.
    # Set any of these to build a differently-shaped block for this round instead.
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None

    # "identity" zeroes this round's block's output projections, so it starts as a residual
    # passthrough and training moves away from the model `previous_checkpoint` already was,
    # rather than from noise. Same reasoning as layer_expand.init.
    init: str = "identity"
    init_std: float = 0.02  # only used by init: "random"

    # Where *this round's* weights come from:
    #   "auto"   -- continue from <training.output_dir>/model if a previous *layer_grow* run at
    #               this same round already finished one, otherwise build a fresh block. Lets
    #               re-running this config extend this round rather than restart it.
    #   None     -- always build a fresh block for this round
    #   <path>   -- continue this round from that grown model directory
    #
    # Distinct from `previous_checkpoint`, which names the *earlier* round to grow from and is
    # always read. This is about *this* round's own progress, mirroring
    # layer_expand.init_from_checkpoint.
    init_from_checkpoint: str | None = "auto"

    # Google Drive fallback for `previous_checkpoint`, used only when nothing usable is found at
    # that path locally -- e.g. a fresh machine that has not trained or downloaded
    # outputs/sft-layer-expand-wide itself yet. Read from a `layer_grow.gdrive.{zip_file_id,
    # cache_dir}` yaml block via the same flattening helper data.gdrive/training.gdrive use
    # elsewhere in this repo, hence the gdrive_-prefixed field names here.
    gdrive_zip_file_id: str | None = None
    # Where the zip is extracted to. None defaults to `previous_checkpoint` itself, so a zip of
    # one checkpoint's contents lands exactly where that path expects to find it -- mirroring
    # training.gdrive_cache_dir's default of output_dir for the same reason.
    gdrive_cache_dir: str | None = None
