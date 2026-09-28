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

# The math prompts only ask for the answer after '####'; nothing asks for the `a op b = c` steps
# reasoning_step_reward/calculation_accuracy_reward score, and a stack SFT'd as a data-science tutor
# never produced one on its own (both scored 0 for a whole run). GRPO can only reinforce what
# some sampled completion already does, so the format is spelled out here, with one worked example.
# The example itself is 37 words, under length_reward's 50-word floor (-0.1): that's small next to
# correctness/format/step rewards (up to +1.85), so it's left as the tradeoff rather than padded.
# "diagnosis" prompts already end with their own '####' instruction and have no step rewards.
MATH_SYSTEM_PROMPT = """\
You solve arithmetic word problems step by step. For each step, write one short sentence saying \
what you compute, then the calculation on its own line as `a op b = c`, using plain numbers and \
one of + - * /. After the last step, write the final number on its own line after '####'.

Example problem: Mia has 12 apples. She buys 6 more, then shares all of them equally among 3 \
friends. How many apples does each friend get?
Example answer:
Mia starts with 12 apples and buys 6 more, so she has:
12 + 6 = 18
She shares the 18 apples equally among 3 friends, so each friend gets:
18 / 3 = 6
#### 6"""

SYSTEM_PROMPTS: dict[str, str | None] = {
    "math": MATH_SYSTEM_PROMPT,
    "diagnosis": None,
}


def resolve_reward_funcs(name: str) -> list[Callable]:
    try:
        return REWARD_SETS[name]
    except KeyError:
        raise ValueError(f"Unknown reward_set {name!r}; choose from {sorted(REWARD_SETS)}") from None
