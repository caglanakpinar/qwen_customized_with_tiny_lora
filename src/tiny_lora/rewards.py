"""Reward functions for GRPO training."""

from __future__ import annotations

import re


def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "") if numbers else None


def correctness_reward(completions: list[str], answer: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        pred = extract_gsm8k_answer(completion)
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
        pred = extract_label_answer(completion)
        gold_label = extract_label_answer(gold)
        rewards.append(1.0 if pred is not None and pred == gold_label else 0.0)
    return rewards


def format_reward(completions: list[str], **kwargs) -> list[float]:
    pattern = r".*?\s*.*?####\s*-?\d"
    return [0.25 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]


def length_reward(
    completions: list[str],
    min_len: int = 50,
    max_len: int = 800,
    **kwargs,
) -> list[float]:
    rewards = []
    for completion in completions:
        length = len(completion.split())
        if length < min_len:
            rewards.append(-0.1)
        elif length > max_len:
            rewards.append(-0.1)
        else:
            rewards.append(0.1)
    return rewards


# `a op b = c`, e.g. "12 + 5 = 17" -- how GSM8K-style solutions work a chain of arithmetic.
# Matched literally rather than via prose connectives ("then", "so") since those don't reliably
# mark a real reasoning step and vary too much in phrasing to count on.
_EQUATION_PATTERN = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*([+\-*/])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)"
)


def _parse_number(token: str) -> float:
    return float(token.replace(",", ""))


def _eval_step(left: str, symbol: str, right: str) -> float:
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }
    return ops[symbol](_parse_number(left), _parse_number(right))


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
        steps = len(_EQUATION_PATTERN.findall(completion))
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
        matches = _EQUATION_PATTERN.findall(completion)
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
        lines = [line.strip() for line in completion.splitlines() if line.strip()]
        if len(lines) < 2:
            rewards.append(0.0)
            continue
        duplicate_ratio = 1 - len(set(lines)) / len(lines)
        rewards.append(penalty * duplicate_ratio)
    return rewards
