"""Compare base-model vs TinyLoRA-checkpoint eval metrics on the held-out split.

Two families of metric, both scored on the same held-out split:

- Teacher-forced (`_eval_loss`): eval_loss/perplexity from scoring the reference tokens directly --
  cheap, but only measures next-token prediction, not what the model says unprompted.
- Generation-based (`_generation_metrics`): actually generate a reply from the prompt and compare it
  to the reference assistant turn with ROUGE-L / token-F1 (text overlap), plus, for the code tasks
  in this dataset, whether the generated ```python block parses at all.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import click
import torch
from trl import SFTConfig, SFTTrainer

from tiny_lora.config import PipelineConfig, SFTTrainingConfig, build_pipeline_config, load_yaml_config
from tiny_lora.data import load_raw_dataset, prepare_sft_eval_dataset
from tiny_lora.model import (
    load_adapter_tokenizer,
    load_base_model,
    load_peft_adapter,
    load_tokenizer,
    resolve_adapter_base_model,
)

_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _token_f1(reference: str, hypothesis: str) -> float:
    """Unigram bag-of-words F1 -- the SQuAD-style overlap metric, tolerant of paraphrase."""
    ref_tokens, hyp_tokens = _tokenize(reference), _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return float(ref_tokens == hyp_tokens)
    overlap = sum((Counter(ref_tokens) & Counter(hyp_tokens)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(hyp_tokens), overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            curr[j] = prev[j - 1] + 1 if token_a == token_b else max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l_f1(reference: str, hypothesis: str) -> float:
    """Longest-common-subsequence F1 -- rewards matching word order, not just word choice."""
    ref_tokens, hyp_tokens = _tokenize(reference), _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return float(ref_tokens == hyp_tokens)
    lcs = _lcs_len(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(hyp_tokens), lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_python_block(text: str) -> str | None:
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1) if match else None


def _is_valid_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _generate_reply(model, tokenizer, prompt_messages: list[dict], max_new_tokens: int) -> str:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    reply_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def _generation_metrics(
    model,
    tokenizer,
    raw_messages: list[list[dict]],
    max_new_tokens: int,
) -> dict[str, float]:
    """Generate a reply per example (prompt = all but the last message, reference = the last) and
    score it against the reference assistant turn. Greedy decoding, so results are reproducible."""
    rouge_scores: list[float] = []
    token_f1_scores: list[float] = []
    code_valid_flags: list[bool] = []
    for messages in raw_messages:
        reference = messages[-1]["content"]
        hypothesis = _generate_reply(model, tokenizer, messages[:-1], max_new_tokens)
        rouge_scores.append(_rouge_l_f1(reference, hypothesis))
        token_f1_scores.append(_token_f1(reference, hypothesis))
        if _extract_python_block(reference) is not None:
            hypothesis_code = _extract_python_block(hypothesis)
            code_valid_flags.append(bool(hypothesis_code) and _is_valid_python(hypothesis_code))

    metrics = {
        "rouge_l_f1": sum(rouge_scores) / len(rouge_scores),
        "token_f1": sum(token_f1_scores) / len(token_f1_scores),
        "num_generation_samples": len(raw_messages),
    }
    if code_valid_flags:
        metrics["code_valid_rate"] = sum(code_valid_flags) / len(code_valid_flags)
        metrics["num_code_samples"] = len(code_valid_flags)
    return metrics


def _length_kwarg() -> str:
    # Same TRL 0.20 rename `train_sft.py` works around: `max_seq_length` -> `max_length`.
    return (
        "max_length"
        if "max_length" in inspect.signature(SFTConfig.__init__).parameters
        else "max_seq_length"
    )


def _eval_loss(model, tokenizer, eval_dataset, train_cfg: SFTTrainingConfig) -> dict[str, float]:
    with tempfile.TemporaryDirectory(prefix="tiny-lora-eval-") as scratch_dir:
        args = SFTConfig(
            output_dir=scratch_dir,
            per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
            report_to="none",
            dataset_text_field="text",
            **{_length_kwarg(): train_cfg.max_seq_length},
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=eval_dataset,  # unused: only `.evaluate()` is called below
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
        click.echo(f"Evaluating {model.__class__.__name__} on {len(eval_dataset)} samples ...")
        metrics = trainer.evaluate()
    eval_loss = metrics["eval_loss"]
    return {"eval_loss": eval_loss, "perplexity": math.exp(eval_loss)}


def _write_results(evals_dir: Path, payload: dict) -> Path:
    """Write one timestamped JSON record into `evals_dir` and return its path.

    Timestamped rather than a single fixed filename so re-evaluating the same checkpoint -- against
    a different split, sample cap or config -- accumulates a history instead of silently replacing
    the previous number. Sorting the directory by name gives them back in chronological order.
    """
    evals_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = evals_dir / f"eval-{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def run_eval(
    config_path: str | Path,
    adapter_path: str | Path,
    overrides: dict | None = None,
    evals_dir: str | Path | None = None,
    save: bool = True,
    generation_samples: int = 30,
    max_new_tokens: int = 256,
) -> dict[str, dict[str, float]]:
    """Evaluate the base model and a TinyLoRA checkpoint/adapter on the configured eval split.

    Reports teacher-forced eval_loss/perplexity plus, unless `generation_samples` is 0,
    generation-based metrics (ROUGE-L, token-F1, and a code-valid rate on the dataset's code-task
    rows) computed by actually generating a reply for the first `generation_samples` eval examples.
    Generation is far slower than teacher-forced scoring (one autoregressive pass per example, per
    model), which is why it runs on a small subset rather than the full eval split.

    Unless `save` is False, the results are also written as JSON under `evals_dir`, which defaults
    to an `evals/` folder inside `adapter_path` -- so a checkpoint carries its own scores.
    """
    raw = load_yaml_config(config_path)
    if overrides:
        for section, values in overrides.items():
            raw.setdefault(section, {}).update(values)
    config: PipelineConfig = build_pipeline_config(raw, SFTTrainingConfig)
    train_cfg: SFTTrainingConfig = config.training  # type: ignore[assignment]
    adapter_path = Path(adapter_path)

    tokenizer = load_tokenizer(
        config.model.model_name_or_path, trust_remote_code=config.model.trust_remote_code
    )
    eval_dataset = prepare_sft_eval_dataset(config.data, tokenizer)
    if eval_dataset is None:
        raise ValueError(
            f"{config_path} has no data.eval_dataset_name set -- an eval split is required "
            "to compare eval loss."
        )

    raw_messages: list[list[dict]] = []
    if generation_samples > 0:
        raw_eval_dataset = load_raw_dataset(config.data, dataset_name=config.data.eval_dataset_name)
        n = min(generation_samples, len(raw_eval_dataset))
        raw_messages = raw_eval_dataset.select(range(n))["messages"]

    click.echo(f"Evaluating base model {config.model.model_name_or_path} ...")
    base_model = load_base_model(config.model)
    base_metrics = _eval_loss(base_model, tokenizer, eval_dataset, train_cfg)
    if raw_messages:
        click.echo(f"Generating {len(raw_messages)} replies from the base model ...")
        base_metrics.update(_generation_metrics(base_model, tokenizer, raw_messages, max_new_tokens))
    del base_model

    base_model_name = resolve_adapter_base_model(adapter_path)
    adapter_tokenizer = load_adapter_tokenizer(
        adapter_path, base_model_name, config.model.trust_remote_code
    )
    click.echo(f"Evaluating checkpoint {adapter_path} ...")
    adapter_model = load_peft_adapter(
        base_model_name,
        str(adapter_path),
        load_in_4bit=config.model.load_in_4bit,
        trust_remote_code=config.model.trust_remote_code,
    )
    adapter_metrics = _eval_loss(adapter_model, adapter_tokenizer, eval_dataset, train_cfg)
    if raw_messages:
        click.echo(f"Generating {len(raw_messages)} replies from the checkpoint ...")
        adapter_metrics.update(
            _generation_metrics(adapter_model, adapter_tokenizer, raw_messages, max_new_tokens)
        )

    results = {"base": base_metrics, "checkpoint": adapter_metrics}

    click.echo("\nResults on the held-out eval split (lower is better, except *_f1/*_rate):")
    headers = ["eval_loss", "perplexity"]
    if raw_messages:
        headers += ["rouge_l_f1", "token_f1"]
        if "code_valid_rate" in adapter_metrics:
            headers.append("code_valid_rate")
    click.echo(f"{'':12}" + "".join(f"{h:>14}" for h in headers))
    for name, metrics in results.items():
        row = "".join(f"{metrics[h]:>14.4f}" for h in headers)
        click.echo(f"{name:<12}{row}")

    if save:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "adapter": str(adapter_path),
            "base_model": base_model_name,
            "config": str(config_path),
            "eval_dataset": config.data.eval_dataset_name,
            "num_eval_samples": len(eval_dataset),
            "max_seq_length": train_cfg.max_seq_length,
            "results": results,
            # The comparison is the point of the command, so record it rather than making a
            # reader re-derive it from the two rows above.
            "delta": {
                "eval_loss": adapter_metrics["eval_loss"] - base_metrics["eval_loss"],
                "perplexity": adapter_metrics["perplexity"] - base_metrics["perplexity"],
                "perplexity_ratio": adapter_metrics["perplexity"] / base_metrics["perplexity"],
                **(
                    {
                        "rouge_l_f1": adapter_metrics["rouge_l_f1"] - base_metrics["rouge_l_f1"],
                        "token_f1": adapter_metrics["token_f1"] - base_metrics["token_f1"],
                    }
                    if raw_messages
                    else {}
                ),
            },
        }
        out_dir = Path(evals_dir) if evals_dir is not None else adapter_path / "evals"
        click.echo(f"\nSaved to {_write_results(out_dir, payload)}")

    return results
