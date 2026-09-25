"""Layer growth: append another transformer block on top of an already-expanded stack, and keep
every previously-added block trainable instead of re-freezing it.

`layer_expand` freezes everything except the single block it just added -- the right choice for a
first block, where "everything before it" is the tuned-but-otherwise-untouched base. Once that
first block has itself been trained, though, it is no longer just frozen scaffolding: it is a
second round's worth of starting point, and re-freezing it when a third block is added would waste
the ability to let the whole tail of the stack keep adapting together.

This module starts from a finished `layer_expand` (or an earlier `layer_grow`) checkpoint, appends
one more block, and leaves every non-base block -- old and new -- trainable. The base layers (the
merged TinyLoRA + layer_lora weights the first round was built on) stay frozen throughout; nothing
here ever re-opens those.

Checkpoints are self-contained the same way `layer_expand`'s are: a `layer_grow.json` sidecar next
to the weights lists every round's block(s) and shape, in order, so `load_grown_model` can rebuild
the whole stack from one directory without needing any earlier checkpoint alongside it.
"""

__version__ = "0.1.0"
