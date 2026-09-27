"""Reward-function sets `layer_grow grpo` can select via `layer_grow.reward_set` (yaml) or
`--reward-set` (CLI, overrides yaml).

Growing a block via GRPO needs reward functions tuned to whatever dataset that round trains
against -- `tiny_lora.rewards`' arithmetic-focused functions (`reasoning_step_reward`,
`calculation_accuracy_reward`, `correctness_reward`) score `#### <number>` GSM8K-shaped answers
and just return 0 against `data/synthetic/data_science_word_problems.py`'s
`#### overfitting`/`#### underfitting` labels, and vice versa `diagnosis_reward` never fires on
arithmetic data (see that module's docstring). Picking one named set here means a training run
doesn't need its own hand-written `reward_funcs` list per round.
"""

from __future__ import annotations

from typing import Callable

from tiny_lora.rewards import (
    calculation_accuracy_reward,
    correctness_reward,
    diagnosis_reward,
    format_reward,
    length_reward,
    reasoning_step_reward,
    repetition_penalty_reward,
)

# "math": GSM8K-shaped arithmetic word problems -- openai/gsm8k, or
#         data/synthetic/math_word_problems.py's grpo_math_{train,eval}*.jsonl.
# "diagnosis": data/synthetic/data_science_word_problems.py's grpo_diagnosis_{train,eval}*.jsonl.
#              format_reward requires a digit right after '####' and would always score 0 against
#              a word label, so it's left out rather than kept as dead weight.
REWARD_SETS: dict[str, list[Callable]] = {
    "math": [
        correctness_reward,
        format_reward,
        length_reward,
        reasoning_step_reward,
        calculation_accuracy_reward,
        repetition_penalty_reward,
    ],
    "diagnosis": [
        diagnosis_reward,
        length_reward,
        repetition_penalty_reward,
    ],
}

DEFAULT_REWARD_SET = "math"


def resolve_reward_funcs(name: str) -> list[Callable]:
    try:
        return REWARD_SETS[name]
    except KeyError:
        raise ValueError(f"Unknown reward_set {name!r}; choose from {sorted(REWARD_SETS)}") from None
