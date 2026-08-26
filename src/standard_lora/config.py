"""Standard LoRA adapter configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StandardLoraConfig:
    r: int = 8                    # LoRA rank -- width of the trainable A/B low-rank update
    lora_alpha: int = 16          # update is scaled by lora_alpha / r (or / sqrt(r) with use_rslora)
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.0     # dropout applied to the LoRA path during training
    bias: str = "none"            # "none", "all", or "lora_only" -- which biases also train
    use_rslora: bool = False      # rank-stabilized scaling (alpha / sqrt(r) instead of alpha / r)
