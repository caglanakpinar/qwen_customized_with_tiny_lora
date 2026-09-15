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
