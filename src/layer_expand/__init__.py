"""Layer expansion: append whole new transformer blocks and train only those.

Not an adapter method. `tiny_lora`, `standard_lora` and `layer_lora` all inject low-rank
updates into projections that already exist; this grows the decoder stack itself and trains
the new blocks densely, every parameter of them. There is consequently no `r`, no
`lora_alpha`, no `target_modules` -- a new block has no host projection to target.

What it keeps from the adapter modules is where a run *starts*: `base_adapters` folds the
finished TinyLoRA and layer_lora adapters into the base weights, in order, before expanding --
so the new block is trained on top of the fully tuned model rather than the stock one. It is a
chain because neither adapter alone covers the whole model: the TinyLoRA run holds all 24
layers, the layer_lora run refines one of them.
"""

__version__ = "0.1.0"
