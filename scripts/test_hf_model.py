"""Smoke-test the adapter as it is published on the Hugging Face Hub.

This is the after-the-push counterpart to `scripts/push_to_hub.py`: it downloads the repo the way
any user would, loads it on top of its declared base model, and puts 100 data-science questions to
it -- the mix the corpus trains on, so the pass rates say something about the fine-tune rather than
about one lucky prompt.

    poetry run python scripts/test_hf_model.py                       # all 100 cases on `main`
    poetry run python scripts/test_hf_model.py --limit 20            # balanced 20-case quick run
    poetry run python scripts/test_hf_model.py --categories pandas,sql
    poetry run python scripts/test_hf_model.py --revision v3-layer-lora
    poetry run python scripts/test_hf_model.py --metadata-only       # no weights, no generation
    poetry run python scripts/test_hf_model.py --json outputs/hf_smoke_test.json

Checks, in order (each one prints PASS/FAIL and the run exits non-zero if any failed):

    metadata        the repo resolves, carries the files an adapter needs, and its weights match
                    the method/parameter count you expect
    model_card      the parameter count the README claims matches the weights actually published
                    -- catches a card regenerated from a different checkpoint than the one uploaded
    load            base model + adapter load together through `PeftModel.from_pretrained`
    generation      every prompt gets a non-empty reply, and prompts that ask for code get a fenced
                    block that parses -- reported as rates per category, checked against
                    --min-answer-rate / --min-code-rate
    adapter_effect  the adapter changes the output: a sample of prompts is re-generated with the
                    adapter disabled and the replies must differ somewhere -- with --json, each
                    sampled case's record carries both the fine-tuned reply ("reply") and this
                    base-model reply ("base_reply"); cases outside the sample keep "base_reply": null

Nothing here reads the local `outputs/` tree -- that is the point. It tests the published artifact.
A full 100-case run generates 100 replies (plus the comparison sample), so on CPU/MPS it is minutes,
not seconds; `--limit` exists for when you just want to know the push landed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# The parameter-count helpers are shared with the pusher so that the card, the push and this test
# all count the same way (LoRA counts every saved tensor; TinyLoRA counts only its `v` vectors).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from push_to_hub import _module_count, _safetensors_header, _trainable_params  # noqa: E402

_DEFAULT_REPO = "Caglana/qwen0.5b-tinylora-ds-assistant"

_METHOD_TITLES = {"TINYLORA": "TinyLoRA", "LORA": "LoRA"}

# Files a downstream `PeftModel.from_pretrained` + `AutoTokenizer.from_pretrained` needs. The card
# is listed too: this repo's README carries the benchmark table, so a push that dropped it is a bug.
_REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "README.md",
]

# A fence the model labelled (```python / ```sql) and a fence it did not. Both are collected: a
# reply that writes correct code inside a bare ``` fence is a formatting miss, while a reply with no
# fence at all did not write code, and the two failures call for different fixes. `eval.py` scores
# only the labelled form, so the labelled rate here is the one comparable to the benchmark.
_LABELLED_FENCE_RE = r"```{lang}\n(.*?)```"
_BARE_FENCE_RE = re.compile(r"```\n(.*?)```", re.S)
_SQL_KEYWORD_RE = re.compile(r"\bselect\b.*\bfrom\b", re.I | re.S)

# "This version trains **30,720 parameters**" -- what build_model_card writes into the README.
_CARD_PARAMS_RE = re.compile(r"trains \*\*([\d,]+) parameters\*\*")


@dataclass
class Prompt:
    """One test question: what to ask, what it exercises, and what a good answer must contain."""

    question: str
    category: str
    code_lang: str | None  # None for prose questions

    @property
    def expects_code(self) -> bool:
        return self.code_lang is not None

    @property
    def label(self) -> str:
        return self.question if len(self.question) <= 62 else self.question[:59] + "..."


# The 100-case suite, grouped by what it exercises. The mix mirrors the training corpus: concept
# Q&A from the knowledge-base generator, pandas/plotting/scikit-learn/statistics/SQL tasks from
# `code_tasks.py`, and pipeline-shaped questions from the Kaggle-grounded generator. Categories
# named in `_CODE_LANGS` expect a fenced code block in the reply; the rest are prose.
_CODE_LANGS = {
    "pandas": "python",
    "viz": "python",
    "sklearn": "python",
    "tuning": "python",
    "evaluation": "python",
    "statistics": "python",
    "kaggle": "python",
    "sql": "sql",
}

_PROMPT_SUITE: dict[str, tuple[str, ...]] = {
    "pandas": (
        "How do I compute a rolling 7-day average in pandas?",
        "Group a DataFrame by customer and compute the mean and max of the order_total column.",
        "Merge two DataFrames on customer_id, keeping every row from the left one.",
        "Pivot a long DataFrame of date/metric/value into one column per metric.",
        "Fill missing values in a numeric column with the median of its group.",
        "Drop duplicate rows, keeping the most recent one per user by timestamp.",
        "Convert a string column of dates into datetimes and set it as the index.",
        "Compute a month-over-month percentage change per product category.",
        "Select the top 3 rows per group by revenue in a pandas DataFrame.",
        "Explode a column that holds lists of tags into one row per tag.",
        "Reduce the memory a DataFrame uses by downcasting numeric and categorical columns.",
        "Read a 10GB CSV in chunks and aggregate a running total per region.",
        "Reindex a daily time series so missing dates appear with NaN instead of being absent.",
        "Rank customers by spend within each country using a window-style pandas operation.",
        "Write a vectorised replacement for a row-wise apply that computes a weighted score.",
    ),
    "viz": (
        "Plot a bar chart of mean revenue per category with matplotlib.",
        "Draw a histogram of a skewed variable with a log-scaled x axis.",
        "Make a scatter plot of two variables coloured by a categorical group.",
        "Plot two time series on the same axes with a shared y scale and a legend.",
        "Draw a boxplot comparing a metric across four experiment arms.",
        "Make a small-multiples grid of one line chart per region with shared axes.",
        "Plot a correlation heatmap of a numeric DataFrame with a perceptually uniform colormap.",
        "Chart a cumulative distribution function of response times.",
        "Annotate the maximum point of a line chart with its value.",
        "Save a matplotlib figure at 300 dpi with tight bounding box for a report.",
    ),
    "sklearn": (
        "Write a scikit-learn pipeline that imputes and scales numeric columns, one-hot encodes "
        "categorical ones, and cross-validates a gradient boosting classifier.",
        "Build a ColumnTransformer that selects numeric and categorical columns by dtype.",
        "Fit a logistic regression with class weights on an imbalanced binary target.",
        "Cross-validate a random forest with StratifiedKFold and report the fold spread.",
        "Encode a high-cardinality categorical column without leaking the target.",
        "Train a ridge regression and pull out the coefficients with their feature names.",
        "Use TimeSeriesSplit to validate a model on data with a time ordering.",
        "Wrap a custom feature transformation in a FunctionTransformer inside a Pipeline.",
        "Handle unseen categories at predict time in a OneHotEncoder.",
        "Persist a fitted pipeline to disk and load it back for inference.",
        "Use GroupKFold so that rows from the same customer never span train and validation.",
        "Calibrate a classifier's probabilities and check the calibration curve.",
    ),
    "tuning": (
        "Run a randomised hyperparameter search over a full preprocessing + model pipeline.",
        "Grid-search the regularisation strength of a logistic regression with 5-fold CV.",
        "Use a nested cross-validation to get an unbiased estimate of a tuned model.",
        "Early-stop a gradient boosting model on a validation set.",
        "Search over both the imputation strategy and the model's depth in one pass.",
        "Tune a model for average precision instead of accuracy on an imbalanced problem.",
        "Set a random_state everywhere in a search so the result reproduces.",
        "Compare two candidate models on the same CV folds rather than on separate splits.",
    ),
    "evaluation": (
        "Compute precision, recall, F1 and ROC AUC for a binary classifier.",
        "Choose a probability threshold that maximises F1 on a validation set.",
        "Plot a precision-recall curve and mark the chosen operating point.",
        "Compute RMSE, MAE and R2 for a regression model and print them together.",
        "Build a confusion matrix and report per-class recall.",
        "Evaluate a model's error broken down by a slice column such as region.",
        "Compute a bootstrap confidence interval around a test-set AUC.",
        "Score a multiclass model with macro and weighted F1 and explain the code.",
    ),
    "statistics": (
        "Run a two-sample t-test and report the effect size alongside the p-value.",
        "Estimate a confidence interval for a conversion rate with a bootstrap.",
        "Test whether two categorical variables are independent with a chi-square test.",
        "Fit an OLS regression with statsmodels and read off the coefficient p-values.",
        "Check whether a sample is approximately normal before choosing a test.",
        "Compute the sample size needed to detect a 2% lift in an A/B test.",
        "Correct a set of p-values for multiple comparisons.",
        "Compute a Spearman correlation and say when to prefer it over Pearson.",
    ),
    "sql": (
        "Write SQL for a 7-day rolling average of daily revenue per store.",
        "Rank customers by lifetime spend within each country using a window function.",
        "Find the second-highest order value per customer in SQL.",
        "Write a query that joins orders to customers and counts orders per signup month.",
        "Deduplicate a table keeping the latest row per id in SQL.",
        "Compute a month-over-month growth rate in SQL using LAG.",
        "Write a query that buckets users into quartiles by spend.",
    ),
    "kaggle": (
        "For the Titanic dataset, write a fold-safe preprocessing pipeline and a baseline model.",
        "Frame the House Prices competition: is it classification or regression, and which metric?",
        "Write feature engineering for the Telco Customer Churn dataset with a ColumnTransformer.",
        "Handle the extreme class imbalance in the Credit Card Fraud dataset in scikit-learn.",
        "Build a validation split for Store Sales forecasting that respects the time ordering.",
        "Write a leakage-safe pipeline for the Home Credit Default Risk dataset.",
        "Engineer datetime features for the Bike Sharing Demand dataset.",
        "Choose and code an evaluation metric for the Porto Seguro competition.",
        "Write a grouped validation scheme for IEEE-CIS Fraud Detection.",
        "Prepare the Adult Census Income dataset, dropping leakage columns by name.",
        "Write a baseline for NYC Taxi Trip Duration including the log-target transform.",
        "Set up cross-validation for Rossmann Store Sales without leaking future weeks.",
    ),
    "concepts": (
        "What is data leakage, and how would I know it happened?",
        "When should I use boosting instead of bagging?",
        "Explain the bias-variance tradeoff in practical terms.",
        "When is deep learning the wrong tool for a tabular problem?",
        "What does regularisation actually do to a linear model?",
        "Why is accuracy a bad metric on an imbalanced dataset?",
        "What is the difference between cross-validation and a single holdout split?",
        "How do I decide between L1 and L2 regularisation?",
        "What is the difference between bagging, boosting and stacking?",
        "Why does CatBoost handle high-cardinality categorical features well?",
        "What goes wrong if I scale my features before splitting the data?",
        "How do I tell whether my model is overfitting?",
        "What is a p-value, and what is it not?",
        "When should I use a stratified split?",
        "What is multicollinearity and when does it actually matter?",
        "How should I handle missing data that is not missing at random?",
        "What is the difference between a validation set and a test set?",
        "Why can a model with better AUC be worse in production?",
        "What is concept drift and how would I detect it?",
        "How do SHAP values differ from feature importances?",
    ),
}

PROMPTS = [
    Prompt(question, category, _CODE_LANGS.get(category))
    for category, questions in _PROMPT_SUITE.items()
    for question in questions
]


def select_prompts(categories: list[str] | None, limit: int | None) -> list[Prompt]:
    """Pick the cases to run.

    `--limit` takes a round-robin slice across categories rather than the first N: the suite is
    stored category-major, so a head slice would be 20 pandas questions and nothing else, and a
    quick run should still touch prose, code and SQL.
    """
    pool = [p for p in PROMPTS if not categories or p.category in categories]
    if limit is None or limit >= len(pool):
        return pool

    by_category: dict[str, list[Prompt]] = {}
    for prompt in pool:
        by_category.setdefault(prompt.category, []).append(prompt)

    picked: list[Prompt] = []
    round_index = 0
    while len(picked) < limit:
        added = False
        for questions in by_category.values():
            if round_index < len(questions):
                picked.append(questions[round_index])
                added = True
                if len(picked) == limit:
                    break
        if not added:
            break
        round_index += 1
    # Back to suite order so two runs with the same --limit report in the same order.
    return [p for p in pool if p in picked]


@dataclass
class Result:
    """One check's outcome, plus whatever detail is worth printing under it."""

    name: str
    passed: bool
    summary: str
    details: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def report(self) -> None:
        print(f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.summary}")
        for line in self.details:
            print(f"       {line}")


def _extract_code(reply: str, lang: str) -> tuple[str | None, bool]:
    """(code, was_the_fence_labelled) for the first fenced block in the reply."""
    labelled = re.search(_LABELLED_FENCE_RE.format(lang=lang), reply, re.S)
    if labelled:
        return labelled.group(1), True
    bare = _BARE_FENCE_RE.search(reply)
    if bare:
        return bare.group(1), False
    return None, False


def _code_is_valid(code: str, lang: str) -> bool:
    """Python has to parse; SQL only gets a shape check, since there is no parser here."""
    if lang == "sql":
        return bool(_SQL_KEYWORD_RE.search(code))
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _resolve_device(choice: str):
    import torch

    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def check_metadata(repo_id: str, revision: str | None, expect_method: str | None) -> Result:
    """Download the small files and describe what is actually published."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = set(list_repo_files(repo_id, revision=revision))
    missing = [name for name in _REQUIRED_FILES if name not in files]
    if missing:
        return Result(
            "metadata",
            False,
            f"{repo_id} is missing {len(missing)} required file(s)",
            [f"missing: {', '.join(missing)}"],
        )

    def _get(name: str) -> Path:
        return Path(hf_hub_download(repo_id, name, revision=revision))

    config = json.loads(_get("adapter_config.json").read_text())
    header = _safetensors_header(_get("adapter_model.safetensors"))
    peft_type = config.get("peft_type", "LORA")
    params = _trainable_params(peft_type, config, header)
    method = _METHOD_TITLES.get(peft_type, peft_type)
    layers = config.get("layers_to_transform")

    details = [
        f"base model: {config['base_model_name_or_path']}",
        f"method: {method} (r={config.get('r')}"
        + (f", u={config['u']}" if peft_type == "TINYLORA" else f", alpha={config.get('lora_alpha')}")
        + ")",
        f"adapted layers: {', '.join(map(str, layers)) if layers else 'all'}",
        f"target modules: {', '.join(sorted(config.get('target_modules', [])))}"
        f" ({_module_count(header)} adapted modules)",
        f"trainable parameters: {params:,}",
    ]
    data = {
        "peft_type": peft_type,
        "base_model": config["base_model_name_or_path"],
        "trainable_params": params,
        "layers": layers,
    }

    if expect_method and peft_type != expect_method.upper():
        return Result(
            "metadata",
            False,
            f"published method is {method}, expected {expect_method}",
            details,
            data,
        )
    return Result("metadata", True, f"{repo_id} publishes a {method} adapter", details, data)


def check_model_card(repo_id: str, revision: str | None, published_params: int) -> Result:
    """The card is generated from a checkpoint; make sure it was generated from *this* one."""
    from huggingface_hub import hf_hub_download

    card = Path(hf_hub_download(repo_id, "README.md", revision=revision)).read_text()
    match = _CARD_PARAMS_RE.search(card)
    if not match:
        return Result(
            "model_card",
            False,
            "the README does not state a trainable-parameter count",
            ["expected a line like: This version trains **30,720 parameters**"],
        )
    claimed = int(match.group(1).replace(",", ""))
    if claimed != published_params:
        return Result(
            "model_card",
            False,
            f"the README claims {claimed:,} parameters but the weights hold {published_params:,}",
            ["the card was probably generated from a different checkpoint than the one uploaded"],
        )
    has_results = "## Results" in card
    return Result(
        "model_card",
        True,
        f"card matches the weights ({claimed:,} parameters)",
        [] if has_results else ["note: the card has no '## Results' section"],
    )


def load_model(repo_id: str, revision: str | None, base_model_id: str, device_choice: str):
    """Load exactly the way the card tells a user to, and return (model, tokenizer, device)."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(device_choice)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, repo_id, revision=revision)
    model = model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, tokenizer, device


def _generate(model, tokenizer, device, question: str, max_new_tokens: int) -> str:
    """Greedy decoding, so two runs of this script on the same revision agree."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    reply_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


def _rate_table(records: list[dict]) -> list[str]:
    """Per-category answered / code-block / valid-code rates, as a fixed-width table."""
    categories: dict[str, list[dict]] = {}
    for record in records:
        categories.setdefault(record["category"], []).append(record)

    lines = [f"{'category':<12} {'n':>4}  {'answered':>8}  {'code block':>10}  {'valid code':>10}"]
    for category, rows in categories.items():
        code_rows = [r for r in rows if r["expects_code"]]
        answered = sum(1 for r in rows if r["answered"]) / len(rows)
        if code_rows:
            blocks = sum(1 for r in code_rows if r["has_code_block"]) / len(code_rows)
            valid = sum(1 for r in code_rows if r["code_valid"]) / len(code_rows)
            block_cell, valid_cell = f"{blocks:>9.0%}", f"{valid:>9.0%}"
        else:
            block_cell = valid_cell = f"{'--':>9}"
        lines.append(f"{category:<12} {len(rows):>4}  {answered:>7.0%}   {block_cell}  {valid_cell}")
    return lines


def check_generation(
    model,
    tokenizer,
    device,
    prompts: list[Prompt],
    max_new_tokens: int,
    show: bool,
    min_answer_rate: float,
    min_code_rate: float,
) -> tuple[Result, list[dict]]:
    """Answer every prompt; a prompt that asked for code must come back with code that is valid."""
    records: list[dict] = []
    failures: list[str] = []

    for index, prompt in enumerate(prompts, start=1):
        reply = _generate(model, tokenizer, device, prompt.question, max_new_tokens)
        record = {
            "category": prompt.category,
            "question": prompt.question,
            "expects_code": prompt.expects_code,
            "answered": bool(reply),
            "reply_chars": len(reply),
            "has_code_block": False,
            "labelled_fence": False,
            "code_valid": False,
            "reply": reply,
            "base_reply": None,
        }
        if prompt.expects_code and reply:
            code, labelled = _extract_code(reply, prompt.code_lang)
            record["has_code_block"] = code is not None
            record["labelled_fence"] = labelled
            record["code_valid"] = bool(code) and _code_is_valid(code, prompt.code_lang)
        records.append(record)

        if not reply:
            failures.append(f"✗ [{prompt.category}] {prompt.label} -- empty reply")
        elif prompt.expects_code and not record["has_code_block"]:
            failures.append(
                f"✗ [{prompt.category}] {prompt.label} -- no code block ({len(reply)} chars)"
            )
        elif prompt.expects_code and not record["code_valid"]:
            failures.append(f"✗ [{prompt.category}] {prompt.label} -- code block is not valid")

        if show:
            print(f"\n--- [{prompt.category}] {prompt.question}\n{reply}\n")
        elif index % 10 == 0 or index == len(prompts):
            print(f"       ... {index}/{len(prompts)} answered", flush=True)

    code_records = [r for r in records if r["expects_code"]]
    answer_rate = sum(1 for r in records if r["answered"]) / len(records)
    code_rate = (
        sum(1 for r in code_records if r["code_valid"]) / len(code_records) if code_records else 1.0
    )
    labelled_rate = (
        sum(1 for r in code_records if r["labelled_fence"]) / len(code_records)
        if code_records
        else 1.0
    )

    details = _rate_table(records) + [
        "",
        f"answered: {answer_rate:.0%} (threshold {min_answer_rate:.0%})",
        f"valid code on {len(code_records)} code prompts: {code_rate:.0%} "
        f"(threshold {min_code_rate:.0%})",
        f"of those, {labelled_rate:.0%} used a labelled fence (```python / ```sql) -- eval.py "
        "counts only labelled blocks, so this is the number comparable to the benchmark",
    ]
    # Cap the failure list: at 100 cases a broken run would otherwise print a hundred lines.
    if failures:
        details += ["", f"{len(failures)} failing case(s):"] + failures[:15]
        if len(failures) > 15:
            details.append(f"... and {len(failures) - 15} more (see the --json output)")

    passed = answer_rate >= min_answer_rate and code_rate >= min_code_rate
    summary = (
        f"{len(prompts)} prompts: {answer_rate:.0%} answered, {code_rate:.0%} valid code"
        if passed
        else f"{len(prompts)} prompts below threshold: {answer_rate:.0%} answered, "
        f"{code_rate:.0%} valid code"
    )
    result = Result("generation", passed, summary, details)
    result.data = {
        "answer_rate": answer_rate,
        "code_valid_rate": code_rate,
        "labelled_fence_rate": labelled_rate,
        "num_prompts": len(records),
        "num_code_prompts": len(code_records),
    }
    return result, records


def check_adapter_effect(
    model, tokenizer, device, records: list[dict], max_new_tokens: int, sample_size: int
) -> Result:
    """Re-answer a sample with the adapter switched off; identical replies mean it was never applied.

    `disable_adapter()` is a context manager on the PeftModel, so this compares against the true base
    model without loading a second copy of the weights. Only a sample is re-generated -- doubling a
    100-case run to prove a point that one differing reply already proves is not worth the minutes.
    """
    if not records:
        return Result("adapter_effect", False, "no replies to compare against")

    stride = max(1, len(records) // sample_size)
    sample = records[::stride][:sample_size]

    differing = 0
    details: list[str] = []
    for record in sample:
        with model.disable_adapter():
            base_reply = _generate(model, tokenizer, device, record["question"], max_new_tokens)
        record["base_reply"] = base_reply
        question = record["question"]
        label = question if len(question) <= 62 else question[:59] + "..."
        if base_reply.strip() != record["reply"].strip():
            differing += 1
        else:
            details.append(f"· identical with and without the adapter: {label}")

    passed = differing > 0
    summary = (
        f"the adapter changes {differing} of {len(sample)} sampled replies"
        if passed
        else "every sampled reply is identical with the adapter disabled -- it is not being applied"
    )
    return Result("adapter_effect", passed, summary, details)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo-id", default=_DEFAULT_REPO, help="adapter repo on the Hub")
    parser.add_argument("--revision", default=None, help="branch, tag or commit sha (default: main)")
    parser.add_argument(
        "--base-model",
        default=None,
        help="base model to load under the adapter (default: whatever adapter_config.json declares)",
    )
    parser.add_argument(
        "--expect-method",
        default=None,
        choices=["tinylora", "lora"],
        help="fail if the published adapter is not this method",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run this many cases, sampled round-robin across categories (default: all 100)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=f"comma-separated subset of: {', '.join(_PROMPT_SUITE)}",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="extra question to ask (repeatable); added to the suite, treated as prose",
    )
    parser.add_argument(
        "--min-answer-rate",
        type=float,
        default=1.0,
        help="fail the generation check below this fraction of non-empty replies",
    )
    parser.add_argument(
        "--min-code-rate",
        type=float,
        default=0.5,
        help="fail the generation check below this fraction of code prompts answered with valid code",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="check what is published without downloading the base model or generating anything",
    )
    parser.add_argument(
        "--compare-base-limit",
        type=int,
        default=8,
        help="how many replies to re-generate with the adapter disabled (0 skips the check)",
    )
    parser.add_argument("--show-replies", action="store_true", help="print each full reply")
    parser.add_argument("--json", type=Path, default=None, help="also write the results here")
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    if categories:
        unknown = [c for c in categories if c not in _PROMPT_SUITE]
        if unknown:
            parser.error(f"unknown categor(ies): {', '.join(unknown)}")

    print(f"==> Testing {args.repo_id}" + (f" @ {args.revision}" if args.revision else " @ main"))
    results: list[Result] = []

    metadata = check_metadata(args.repo_id, args.revision, args.expect_method)
    results.append(metadata)
    metadata.report()
    if not metadata.passed:
        return _finish(results, [], args.json)

    card = check_model_card(args.repo_id, args.revision, metadata.data["trainable_params"])
    results.append(card)
    card.report()

    if args.metadata_only:
        print("==> --metadata-only: skipping load and generation")
        return _finish(results, [], args.json)

    base_model_id = args.base_model or metadata.data["base_model"]
    print(f"==> Loading {base_model_id} + adapter (this downloads ~1GB the first time)")
    try:
        model, tokenizer, device = load_model(args.repo_id, args.revision, base_model_id, args.device)
    except Exception as exc:  # noqa: BLE001 -- any load failure is the finding, whatever it is
        load = Result("load", False, f"{type(exc).__name__}: {exc}")
        results.append(load)
        load.report()
        return _finish(results, [], args.json)

    load = Result("load", True, f"loaded on {device}", [f"base model: {base_model_id}"])
    results.append(load)
    load.report()

    prompts = select_prompts(categories, args.limit)
    prompts += [Prompt(question, "custom", None) for question in args.prompt]
    print(f"==> Generating {len(prompts)} replies at {args.max_new_tokens} new tokens each")

    generation, records = check_generation(
        model,
        tokenizer,
        device,
        prompts,
        args.max_new_tokens,
        args.show_replies,
        args.min_answer_rate,
        args.min_code_rate,
    )
    results.append(generation)
    generation.report()

    if args.compare_base_limit > 0:
        effect = check_adapter_effect(
            model, tokenizer, device, records, args.max_new_tokens, args.compare_base_limit
        )
        results.append(effect)
        effect.report()

    return _finish(results, records, args.json)


def _finish(results: list[Result], records: list[dict], json_path: Path | None) -> int:
    failed = [r for r in results if not r.passed]
    print()
    print(f"==> {len(results) - len(failed)}/{len(results)} checks passed")
    for result in failed:
        print(f"    FAILED {result.name}: {result.summary}")

    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(
                {
                    "passed": not failed,
                    "checks": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "summary": r.summary,
                            "details": r.details,
                            **({"data": r.data} if r.data else {}),
                        }
                        for r in results
                    ],
                    "cases": records,
                },
                indent=2,
            )
        )
        print(f"==> Wrote {json_path}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
