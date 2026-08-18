"""Model and tokenizer loading with TinyLoRA adapters."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tiny_lora.config import ModelConfig, TinyLoraConfig


def _resolve_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype}")
    return mapping[dtype]


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _bitsandbytes_available() -> bool:
    try:
        import bitsandbytes  # noqa: F401

        return True
    except ImportError:
        return False


def _best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_base_model(model_cfg: ModelConfig):
    model_kwargs: dict = {"trust_remote_code": model_cfg.trust_remote_code}
    if model_cfg.attn_implementation:
        model_kwargs["attn_implementation"] = model_cfg.attn_implementation

    use_4bit = model_cfg.load_in_4bit and _bitsandbytes_available()
    if model_cfg.load_in_4bit and not use_4bit:
        import warnings

        warnings.warn(
            "4-bit quantization requested but bitsandbytes is unavailable. "
            "Falling back to bf16. Install with: poetry install --extras quant",
            stacklevel=2,
        )

    if use_4bit:
        # `device_map="auto"` is only needed to place quantized weights -- accelerate's
        # meta-device dispatch it relies on targets CUDA/disk-offload and, on MPS/CPU, can
        # leave newly-injected adapter parameters (e.g. a later `load_peft_adapter` call)
        # stranded on the meta device instead of materialized. Skip it when unused and just
        # move the plain model to the best available device below.
        model_kwargs["device_map"] = "auto"
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(model_cfg.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(model_cfg.model_name_or_path, **model_kwargs)
    if not use_4bit:
        model = model.to(_best_device())
    return model


def build_tinylora_config(cfg: TinyLoraConfig):
    try:
        from peft import TinyLoraConfig as PeftTinyLoraConfig
    except ImportError as exc:
        raise ImportError(
            "TinyLoraConfig requires peft>=0.15.0. "
            "Upgrade with: poetry add peft@latest"
        ) from exc

    return PeftTinyLoraConfig(
        r=cfg.r,
        u=cfg.u,
        weight_tying=cfg.weight_tying,
        target_modules=cfg.target_modules,
        projection_seed=cfg.projection_seed,
        save_projection=cfg.save_projection,
        init_weights=cfg.init_weights,
        init_v_bound=cfg.init_v_bound,
        tinylora_dropout=cfg.tinylora_dropout,
    )


def load_tinylora_model(model_cfg: ModelConfig, tinylora_cfg: TinyLoraConfig):
    base_model = load_base_model(model_cfg)
    if model_cfg.load_in_4bit and _bitsandbytes_available():
        from peft import prepare_model_for_kbit_training

        base_model = prepare_model_for_kbit_training(base_model)

    peft_config = build_tinylora_config(tinylora_cfg)
    model = get_peft_model(base_model, peft_config)
    return model


def load_peft_adapter(
    base_model_name: str,
    adapter_path: str,
    load_in_4bit: bool = True,
    trust_remote_code: bool = False,
):
    """Load a base model plus a saved PEFT adapter (a trained TinyLoRA checkpoint) for inference.

    Routes through `load_base_model` so a missing bitsandbytes install (e.g. on macOS) falls
    back to bf16 with a warning instead of crashing, exactly as training does.
    """
    base_model = load_base_model(
        ModelConfig(
            model_name_or_path=base_model_name,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
    )
    return PeftModel.from_pretrained(base_model, adapter_path)


def resolve_adapter_base_model(adapter_path: str | Path) -> str:
    """Read the base model id out of a saved adapter's own config.

    Every saved adapter -- the final one under `<output_dir>/adapter` and each periodic
    `checkpoint-N` -- carries an `adapter_config.json` with `base_model_name_or_path`, so the
    base model normally never needs to be typed by hand.
    """
    adapter_config_path = Path(adapter_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(
            f"No adapter_config.json found in {adapter_path}. Point at a saved adapter or "
            "checkpoint directory, e.g. outputs/sft-ds-assistant/adapter or "
            "outputs/sft-ds-assistant/checkpoint-500."
        )
    base_model_name = json.loads(adapter_config_path.read_text()).get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"{adapter_config_path} has no base_model_name_or_path.")
    return base_model_name


def load_adapter_tokenizer(adapter_path: str | Path, base_model_name: str, trust_remote_code: bool = False):
    """Load the tokenizer saved alongside an adapter, falling back to the base model's.

    Periodic `checkpoint-N` dirs may not carry tokenizer files -- only the final `adapter` dir
    (saved explicitly by `train_sft.py`) does.
    """
    try:
        return load_tokenizer(str(adapter_path), trust_remote_code=trust_remote_code)
    except OSError:
        return load_tokenizer(base_model_name, trust_remote_code=trust_remote_code)
