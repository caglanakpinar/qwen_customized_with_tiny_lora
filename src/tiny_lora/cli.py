"""Click CLI for the TinyLoRA training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click

from tiny_lora import __version__
from tiny_lora.config import (
    GRPOTrainingConfig,
    PipelineConfig,
    SFTTrainingConfig,
    build_pipeline_config,
    config_to_dict,
    load_yaml_config,
)
from tiny_lora.model import build_tinylora_config, load_tinylora_model, load_tokenizer
from tiny_lora.train_sft import run_sft_from_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_CONFIG = PROJECT_ROOT / "configs" / "sft_default.yaml"
DEFAULT_GRPO_CONFIG = PROJECT_ROOT / "configs" / "grpo_default.yaml"


def _apply_common_overrides(
    overrides: dict,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> dict:
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if u is not None:
        overrides.setdefault("tinylora", {})["u"] = u
    if r is not None:
        overrides.setdefault("tinylora", {})["r"] = r
    if weight_tying is not None:
        overrides.setdefault("tinylora", {})["weight_tying"] = weight_tying
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir
    return overrides


@click.group()
@click.version_option(__version__, prog_name="tiny-lora")
def cli() -> None:
    """TinyLoRA training pipeline — SFT and GRPO with PEFT."""


@cli.command("sft")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run supervised fine-tuning with TinyLoRA."""
    overrides = _apply_common_overrides({}, model, u, r, weight_tying, max_samples, output_dir, no_quant)
    adapter_path = run_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Adapter saved to: {adapter_path}")


@cli.command("sft-lora")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (reads its standard_lora: section).",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--r", type=int, default=None, help="LoRA rank.")
@click.option("--lora-alpha", type=int, default=None, help="LoRA alpha (scaling numerator).")
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def sft_lora_cmd(
    config_path: Path,
    model: str | None,
    r: int | None,
    lora_alpha: int | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run supervised fine-tuning with standard LoRA."""
    from standard_lora.train_sft import run_sft_from_yaml as run_standard_lora_sft_from_yaml

    overrides: dict = {}
    if model:
        overrides.setdefault("model", {})["model_name_or_path"] = model
    if no_quant:
        overrides.setdefault("model", {})["load_in_4bit"] = False
    if r is not None:
        overrides.setdefault("standard_lora", {})["r"] = r
    if lora_alpha is not None:
        overrides.setdefault("standard_lora", {})["lora_alpha"] = lora_alpha
    if max_samples is not None:
        overrides.setdefault("data", {})["max_samples"] = max_samples
    if output_dir:
        overrides.setdefault("training", {})["output_dir"] = output_dir

    adapter_path = run_standard_lora_sft_from_yaml(config_path, overrides)
    click.echo(f"SFT complete. Adapter saved to: {adapter_path}")


@cli.command("grpo")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GRPO_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--max-samples", type=int, default=None, help="Limit training samples.")
@click.option("--output-dir", default=None, help="Override output directory.")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
def grpo_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    max_samples: int | None,
    output_dir: str | None,
    no_quant: bool,
) -> None:
    """Run GRPO reinforcement learning with TinyLoRA."""
    from tiny_lora.train_grpo import run_grpo_from_yaml  # local: trl's GRPOTrainer needs vllm

    overrides = _apply_common_overrides({}, model, u, r, weight_tying, max_samples, output_dir, no_quant)
    adapter_path = run_grpo_from_yaml(config_path, overrides)
    click.echo(f"GRPO complete. Adapter saved to: {adapter_path}")


@cli.command("chat")
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option("--model", "base_model", default=None, help="Override base model (default: read from the adapter).")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
@click.option("--max-new-tokens", type=int, default=512, show_default=True, help="Max tokens generated per reply.")
@click.option("--temperature", type=float, default=0.7, show_default=True, help="Sampling temperature; 0 for greedy.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt to prepend.")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for the base model.")
@click.option(
    "--summarize-after",
    "summarize_after_turns",
    type=int,
    default=6,
    show_default=True,
    help="Fold older turns into a running key-points summary once history exceeds this many turns.",
)
@click.option(
    "--keep-recent",
    "keep_recent_turns",
    type=int,
    default=2,
    show_default=True,
    help="Number of most recent turns to keep verbatim when summarizing.",
)
@click.option(
    "--memory-dir",
    "memory_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs/chat_memory"),
    show_default=True,
    help="Where the FAISS index and Chroma collection of persisted running-summaries are written.",
)
@click.option(
    "--db-path",
    "db_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Knowledge base directory, e.g. data/data_science_dbs. Must contain a faiss_index/ and a "
    "chroma_db/; every chat turn is retrieval-augmented with matches from it.",
)
def chat_cmd(
    adapter_path: Path,
    base_model: str | None,
    no_quant: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool,
    summarize_after_turns: int,
    keep_recent_turns: int,
    memory_dir: Path,
    db_path: Path | None,
) -> None:
    """Open an interactive chat REPL with a trained TinyLoRA adapter."""
    from tiny_lora.chat import run_chat

    run_chat(
        adapter_path=adapter_path,
        base_model_override=base_model,
        load_in_4bit=not no_quant,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        summarize_after_turns=summarize_after_turns,
        keep_recent_turns=keep_recent_turns,
        trust_remote_code=trust_remote_code,
        memory_dir=memory_dir,
        db_path=db_path,
    )


@cli.command("serve")
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option("--model", "base_model", default=None, help="Override base model (default: read from the adapter).")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
@click.option("--max-new-tokens", type=int, default=1024, show_default=True, help="Max tokens generated per reply.")
@click.option("--temperature", type=float, default=0.7, show_default=True, help="Sampling temperature; 0 for greedy.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt to prepend.")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for the base model.")
@click.option(
    "--summarize-after",
    "summarize_after_turns",
    type=int,
    default=20,
    show_default=True,
    help="Fold older turns into a running key-points summary once history exceeds this many turns.",
)
@click.option(
    "--keep-recent",
    "keep_recent_turns",
    type=int,
    default=8,
    show_default=True,
    help="Number of most recent turns to keep verbatim when summarizing.",
)
@click.option(
    "--memory-dir",
    "memory_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs/chat_memory"),
    show_default=True,
    help="Where the FAISS index and Chroma collection of persisted running-summaries are written.",
)
@click.option(
    "--db-path",
    "db_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Knowledge base directory, e.g. data/data_science_dbs. Must contain a faiss_index/ and a "
    "chroma_db/; every chat turn is retrieval-augmented with matches from it.",
)
@click.option("--host", default="127.0.0.1", show_default=True, help="Host interface to bind the web server to.")
@click.option("--port", type=int, default=8000, show_default=True, help="Port to serve the chat UI on.")
def serve_cmd(
    adapter_path: Path,
    base_model: str | None,
    no_quant: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool,
    summarize_after_turns: int,
    keep_recent_turns: int,
    memory_dir: Path,
    db_path: Path | None,
    host: str,
    port: int,
) -> None:
    """Serve a browser chat UI (web/) backed by a trained TinyLoRA adapter."""
    import sys

    import uvicorn

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from web.app import create_app

    click.echo(f"Loading adapter {adapter_path} ...")
    if db_path is not None:
        click.echo(f"Loading knowledge base from {db_path} ...")
    app = create_app(
        adapter_path=adapter_path,
        base_model_override=base_model,
        load_in_4bit=not no_quant,
        trust_remote_code=trust_remote_code,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        summarize_after_turns=summarize_after_turns,
        keep_recent_turns=keep_recent_turns,
        memory_dir=memory_dir,
        db_path=db_path,
    )
    click.echo(f"Chat UI ready at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port)


@cli.command("chat-api")
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option("--model", "base_model", default=None, help="Override base model (default: read from the adapter).")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization (use bf16).")
@click.option("--max-new-tokens", type=int, default=512, show_default=True, help="Max tokens generated per reply.")
@click.option("--temperature", type=float, default=0.7, show_default=True, help="Sampling temperature; 0 for greedy.")
@click.option("--system", "system_prompt", default=None, help="Optional system prompt to prepend.")
@click.option("--trust-remote-code", is_flag=True, help="Trust remote code for the base model.")
@click.option(
    "--summarize-after",
    "summarize_after_turns",
    type=int,
    default=6,
    show_default=True,
    help="Fold older turns into a running key-points summary once history exceeds this many turns.",
)
@click.option(
    "--keep-recent",
    "keep_recent_turns",
    type=int,
    default=2,
    show_default=True,
    help="Number of most recent turns to keep verbatim when summarizing.",
)
@click.option(
    "--memory-dir",
    "memory_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("outputs/chat_memory"),
    show_default=True,
    help="Where the FAISS index and Chroma collection of persisted running-summaries are written.",
)
@click.option(
    "--db-path",
    "db_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Knowledge base directory, e.g. data/data_science_dbs. Must contain a faiss_index/ and a "
    "chroma_db/; every chat turn is retrieval-augmented with matches from it.",
)
@click.option("--model-name", "model_name", default="tinylora-chat", show_default=True, help="KServe model name.")
@click.option("--http-port", type=int, default=8080, show_default=True, help="Port KServe serves the API on.")
def chat_api_cmd(
    adapter_path: Path,
    base_model: str | None,
    no_quant: bool,
    max_new_tokens: int,
    temperature: float,
    system_prompt: str | None,
    trust_remote_code: bool,
    summarize_after_turns: int,
    keep_recent_turns: int,
    memory_dir: Path,
    db_path: Path | None,
    model_name: str,
    http_port: int,
) -> None:
    """Serve a payload/response chat inference API via KServe, backed by a trained TinyLoRA adapter.

    POST /v1/models/<model-name>:predict with {"instances": [{"message": "...", "session_id":
    "optional"}]} -> {"predictions": [{"session_id": "...", "reply": "..."}]}. Requires kserve,
    which is not installed by `poetry install` (it conflicts with this project's protobuf pin) --
    install it separately in this environment, e.g. `pip install kserve`.
    """
    from tiny_lora.chat_api import run_chat_api  # local: kserve isn't a project dependency

    click.echo(f"Loading adapter {adapter_path} ...")
    if db_path is not None:
        click.echo(f"Loading knowledge base from {db_path} ...")
    run_chat_api(
        adapter_path=adapter_path,
        base_model_override=base_model,
        load_in_4bit=not no_quant,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        trust_remote_code=trust_remote_code,
        summarize_after_turns=summarize_after_turns,
        keep_recent_turns=keep_recent_turns,
        memory_dir=memory_dir,
        db_path=db_path,
        model_name=model_name,
        http_port=http_port,
    )


@cli.command("eval")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_SFT_CONFIG,
    show_default=True,
    help="YAML config file (supplies the eval split, base model, and eval batch/seq settings).",
)
@click.option(
    "--adapter",
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Saved adapter or checkpoint dir, e.g. outputs/sft-ds-assistant/adapter "
    "or outputs/sft-ds-assistant/checkpoint-500.",
)
@click.option(
    "--max-eval-samples",
    type=int,
    default=None,
    help="Cap the eval split to this many rows (overrides data.max_eval_samples).",
)
@click.option(
    "--evals-dir",
    "evals_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Where to write the JSON results [default: an evals/ folder inside --adapter].",
)
@click.option("--no-save", is_flag=True, help="Print the results without writing them to disk.")
def eval_cmd(
    config_path: Path,
    adapter_path: Path,
    max_eval_samples: int | None,
    evals_dir: Path | None,
    no_save: bool,
) -> None:
    """Compare base-model vs checkpoint eval loss/perplexity on the configured eval split."""
    from tiny_lora.eval import run_eval

    overrides: dict = {}
    if max_eval_samples is not None:
        overrides.setdefault("data", {})["max_eval_samples"] = max_eval_samples
    run_eval(config_path, adapter_path, overrides, evals_dir=evals_dir, save=not no_save)


@cli.command("info")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_GRPO_CONFIG,
    show_default=True,
    help="YAML config file.",
)
@click.option("--model", default=None, help="Override base model name or path.")
@click.option("--u", type=int, default=None, help="TinyLoRA trainable vector size.")
@click.option("--r", type=int, default=None, help="TinyLoRA SVD rank.")
@click.option("--weight-tying", type=float, default=None, help="Weight tying ratio [0, 1].")
@click.option("--no-quant", is_flag=True, help="Disable 4-bit quantization.")
@click.option("--json", "as_json", is_flag=True, help="Print config as JSON.")
def info_cmd(
    config_path: Path,
    model: str | None,
    u: int | None,
    r: int | None,
    weight_tying: float | None,
    no_quant: bool,
    as_json: bool,
) -> None:
    """Print resolved config and trainable parameter count."""
    raw = load_yaml_config(config_path)
    overrides = _apply_common_overrides({}, model, u, r, weight_tying, None, None, no_quant)
    for section, values in overrides.items():
        raw.setdefault(section, {}).update(values)

    training_cls = GRPOTrainingConfig if "grpo" in config_path.name else SFTTrainingConfig
    config: PipelineConfig = build_pipeline_config(raw, training_cls)

    if as_json:
        click.echo(json.dumps(config_to_dict(config), indent=2))
        return

    click.echo("Model config:")
    click.echo(f"  base: {config.model.model_name_or_path}")
    click.echo(f"  4-bit quant: {config.model.load_in_4bit}")
    click.echo("TinyLoRA config:")
    click.echo(f"  r={config.tinylora.r}, u={config.tinylora.u}, weight_tying={config.tinylora.weight_tying}")
    click.echo(f"  targets: {config.tinylora.target_modules}")

    click.echo("Loading model to count trainable parameters...")
    model_obj = load_tinylora_model(config.model, config.tinylora)
    model_obj.print_trainable_parameters()


@cli.command("show-config")
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show_config_cmd(config_path: Path) -> None:
    """Print a YAML config file as formatted JSON."""
    raw = load_yaml_config(config_path)
    click.echo(json.dumps(raw, indent=2))


if __name__ == "__main__":
    cli()
