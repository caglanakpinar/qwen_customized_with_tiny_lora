"""Reward functions for GRPO training."""

from __future__ import annotations

import re
from typing import Callable


def _completion_text(completion: str | list[dict]) -> str:
    """Normalize one entry of `completions` to plain text.

    Our math/diagnosis datasets store `prompt` as a chat-message list (see
    data/synthetic/math_word_problems.py's own docstring), so trl.GRPOTrainer treats the whole
    dataset as conversational and generates completions in that same shape: `list[list[dict]]`
    (one assistant-role message dict per completion), not `list[str]`. Every reward function
    below is written against plain text, so this is the one place that shape gets collapsed.
    """
    if isinstance(completion, list):
        return completion[-1]["content"]
    return completion


def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        pred = extract_gsm8k_answer(_completion_text(completion))
        gold_val = extract_gsm8k_answer(gold)
        rewards.append(1.0 if pred is not None and pred == gold_val else 0.0)
    return rewards


def extract_label_answer(text: str) -> str | None:
    """The word after '####', e.g. "#### overfitting" -> "overfitting".

    Separate from `extract_gsm8k_answer`, whose pattern requires a digit right after '####' and so
    never matches a categorical label like this -- and vice versa, this never matches a number.
    """
    match = re.search(r"####\s*([a-zA-Z][a-zA-Z_\-]*)", text)
    return match.group(1).strip().lower() if match else None


def diagnosis_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    """Like `correctness_reward`, but for word-labeled answers (e.g. data_science_word_problems.py's
    "#### overfitting" / "#### underfitting") instead of GSM8K's numeric ones.
    """
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        pred = extract_label_answer(_completion_text(completion))
        gold_label = extract_label_answer(gold)
        rewards.append(1.0 if pred is not None and pred == gold_label else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    pattern = r".*?\s*.*?####\s*-?\d"
    return [
        0.25 if re.search(pattern, _completion_text(c), re.DOTALL) else 0.0 for c in completions
    ]


def length_reward(
    completions: list[str],
    min_len: int = 50,
    max_len: int = 800,
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        length = len(_completion_text(completion).split())
        if length < min_len:
            rewards.append(-0.1)
        elif length > max_len:
            rewards.append(-0.1)
        else:
            rewards.append(0.1)
    return rewards


# Every spelling of an operator a model writes in practice, mapped to what it computes. Qwen
# instruct models often write multiplication as `×` or LaTeX `\times`/`\cdot` rather than `*`, and
# an ASCII-only pattern scores those steps as if they were never shown.
_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "−": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "×": lambda a, b: a * b,
    "·": lambda a, b: a * b,
    "x": lambda a, b: a * b,
    r"\times": lambda a, b: a * b,
    r"\cdot": lambda a, b: a * b,
    "/": lambda a, b: a / b,
    "÷": lambda a, b: a / b,
    r"\div": lambda a, b: a / b,
}

# `a op b = c`, e.g. "12 + 5 = 17" -- how GSM8K-style solutions work a chain of arithmetic.
# Matched literally rather than via prose connectives ("then", "so") since those don't reliably
# mark a real reasoning step and vary too much in phrasing to count on.
_EQUATION_PATTERN = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*("
    + "|".join(re.escape(op) for op in sorted(_OPERATORS, key=len, reverse=True))
    + r")\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)"
)


def _parse_number(token: str) -> float:
    return float(token.replace(",", ""))


def _eval_step(left: str, symbol: str, right: str) -> float:
    return _OPERATORS[symbol](_parse_number(left), _parse_number(right))


def reasoning_step_reward(
    completions: list[str],
    reward_per_step: float = 0.1,
    max_steps: int = 3,
    **kwargs,
) -> list[float]:
    """Reward completions that show their work as explicit arithmetic steps.

    Counts `a op b = c` equations rather than the final answer, so a model is rewarded for
    working the problem in visible steps instead of only stating a number after '####'.
    """
    rewards = []
    for completion in completions:
        steps = len(_EQUATION_PATTERN.findall(_completion_text(completion)))
        rewards.append(reward_per_step * min(steps, max_steps))
    return rewards


def calculation_accuracy_reward(
    completions: list[str],
    scale: float = 0.3,
    **kwargs,
) -> list[float]:
    """Reward the fraction of a completion's shown arithmetic steps that are actually correct.

    This scores the reasoning itself rather than only the final answer: a completion can get
    `correctness_reward` right by luck while its intermediate arithmetic is broken, or get it
    wrong while every step it worked along the way was correct. Completions with no equations
    to check score 0, same as `reasoning_step_reward`.
    """
    rewards = []
    for completion in completions:
        matches = _EQUATION_PATTERN.findall(_completion_text(completion))
        if not matches:
            rewards.append(0.0)
            continue
        correct = 0
        for left, symbol, right, result in matches:
            try:
                expected = _eval_step(left, symbol, right)
            except ZeroDivisionError:
                continue
            if abs(expected - _parse_number(result)) < 1e-2:
                correct += 1
        rewards.append(scale * correct / len(matches))
    return rewards


def repetition_penalty_reward(
    completions: list[str],
    penalty: float = -0.2,
    **kwargs,
) -> list[float]:
    """Penalize completions that pad length by repeating the same line instead of reasoning.

    Once `length_reward` rewards longer completions, repeating a sentence becomes a cheap way
    to farm that reward without adding any new reasoning -- this counters that shortcut.
    """
    rewards = []
    for completion in completions:
        lines = [line.strip() for line in _completion_text(completion).splitlines() if line.strip()]
        if len(lines) < 2:
            rewards.append(0.0)
            continue
        duplicate_ratio = 1 - len(set(lines)) / len(lines)
        rewards.append(penalty * duplicate_ratio)
    return rewards


def make_sample_printer(num_samples: int, every: int) -> Callable:
    """Build a reward function that scores nothing but prints what the policy is generating.

    trl==0.14.0's GRPOTrainer has no option to show completions, and the per-function reward
    means it logs can say a reward never fires without saying why. This rides along in
    `reward_funcs`, always returning 0.0 so it never moves the summed reward or the advantages,
    and prints the first prompt of a batch, its gold answer, and `num_samples` of that prompt's
    completions (trl keeps one prompt's generations adjacent) on the first call and every
    `every`-th call after it. It shows up in the logs as `rewards/print_samples`, always 0.
    """
    calls = 0

    def print_samples(
        completions: list, prompts: list | None = None, answer: list[str] | None = None, **kwargs
    ) -> list[float]:
        nonlocal calls
        calls += 1
        if (calls - 1) % every == 0:
            lines = [f"\n===== sample completions (reward call {calls}) ====="]
            if prompts:
                lines.append(f"PROMPT: {_completion_text(prompts[0])}")
            if answer:
                lines.append(f"GOLD:   {answer[0]!r}")
            for i, completion in enumerate(completions[:num_samples], start=1):
                text = _completion_text(completion)
                lines.append(f"--- completion {i}/{num_samples} ({len(text.split())} words) ---")
                lines.append(text)
            lines.append("=" * 50)
            print("\n".join(lines), flush=True)
        return [0.0] * len(completions)

    return print_samples
