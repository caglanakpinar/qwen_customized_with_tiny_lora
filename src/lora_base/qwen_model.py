"""A hand-written TensorFlow/Keras port of the Qwen2/Qwen3 dense causal-LM architecture.

`transformers` has no `TFQwen*` classes, so this exists to make `TFAutoModelForCausalLM`
unnecessary: `build_qwen_lora_model` reads a checkpoint's own `config.json` + safetensors state
dict, builds the matching graph (RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, and Qwen3's
optional per-head QK-norm -- detected from the checkpoint's keys, not hardcoded by model name),
and assigns every weight by hand. Only `q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/
`up_proj`/`down_proj` in `lora_base.target_modules` become `LoRADense`; everything else, and every
layer not listed, stays a frozen `FrozenDense`/`FrozenEmbedding`.

Out of scope (see the plan): sliding-window attention, quantization, gradient checkpointing, and
any non-text (vision-language) components newer checkpoints may ship alongside the text model.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf

from lora_base.config import LoraBaseConfig, TFModelConfig
from lora_base.hf_weights import fetch_checkpoint_files, has_key, load_hf_config, load_state_dict
from lora_base.layers import (
    FrozenDense,
    FrozenEmbedding,
    LoRADense,
    RMSNorm,
    apply_rotary_pos_emb,
    compute_rope_cos_sin,
    repeat_kv,
)

# Additive mask value for disallowed (future / padding) attention positions. A large finite
# negative rather than -inf: exp(-1e9) still underflows softmax to 0, but avoids -inf + -inf
# overflow bookkeeping once the causal and padding masks are summed.
_NEG_INF = -1e9


def _resolve_tf_dtype(name: str) -> tf.DType:
    mapping = {"bfloat16": tf.bfloat16, "float16": tf.float16, "float32": tf.float32}
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name!r}. Use one of {sorted(mapping)}.")
    return mapping[name]


def _make_projection(
    name: str,
    in_features: int,
    out_features: int,
    use_bias: bool,
    lora_cfg: LoraBaseConfig,
    dtype_: tf.DType,
) -> FrozenDense:
    if name in lora_cfg.target_modules:
        return LoRADense(
            in_features,
            out_features,
            r=lora_cfg.r,
            alpha=lora_cfg.lora_alpha,
            dropout=lora_cfg.lora_dropout,
            use_bias=use_bias,
            dtype_=dtype_,
            seed=lora_cfg.seed,
            name=name,
        )
    return FrozenDense(in_features, out_features, use_bias=use_bias, dtype_=dtype_, name=name)


class QwenAttention(tf.keras.layers.Layer):
    def __init__(
        self,
        hf_cfg: dict,
        lora_cfg: LoraBaseConfig,
        dtype_: tf.DType,
        has_qk_norm: bool,
        has_qkv_bias: bool,
    ):
        super().__init__(name="self_attn")
        hidden_size = hf_cfg["hidden_size"]
        self.num_heads = hf_cfg["num_attention_heads"]
        self.num_kv_heads = hf_cfg.get("num_key_value_heads", self.num_heads)
        self.head_dim = hf_cfg.get("head_dim", hidden_size // self.num_heads)
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = _make_projection("q_proj", hidden_size, self.num_heads * self.head_dim, has_qkv_bias, lora_cfg, dtype_)
        self.k_proj = _make_projection("k_proj", hidden_size, self.num_kv_heads * self.head_dim, has_qkv_bias, lora_cfg, dtype_)
        self.v_proj = _make_projection("v_proj", hidden_size, self.num_kv_heads * self.head_dim, has_qkv_bias, lora_cfg, dtype_)
        self.o_proj = _make_projection("o_proj", self.num_heads * self.head_dim, hidden_size, False, lora_cfg, dtype_)

        eps = hf_cfg.get("rms_norm_eps", 1e-6)
        self.q_norm = RMSNorm(self.head_dim, eps=eps, dtype_=dtype_, name="q_norm") if has_qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, eps=eps, dtype_=dtype_, name="k_norm") if has_qk_norm else None

    def call(self, x: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor, attn_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        q = self.q_proj(x, training=training)
        k = self.k_proj(x, training=training)
        v = self.v_proj(x, training=training)

        q = tf.reshape(q, [batch, seq_len, self.num_heads, self.head_dim])
        k = tf.reshape(k, [batch, seq_len, self.num_kv_heads, self.head_dim])
        v = tf.reshape(v, [batch, seq_len, self.num_kv_heads, self.head_dim])

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = tf.transpose(q, [0, 2, 1, 3])  # [batch, num_heads, seq_len, head_dim]
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = tf.matmul(tf.cast(q, tf.float32), tf.cast(k, tf.float32), transpose_b=True) * scale
        scores = scores + attn_mask
        probs = tf.cast(tf.nn.softmax(scores, axis=-1), v.dtype)

        attn_out = tf.matmul(probs, v)  # [batch, num_heads, seq_len, head_dim]
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])
        attn_out = tf.reshape(attn_out, [batch, seq_len, self.num_heads * self.head_dim])
        return self.o_proj(attn_out, training=training)


class QwenMLP(tf.keras.layers.Layer):
    def __init__(self, hf_cfg: dict, lora_cfg: LoraBaseConfig, dtype_: tf.DType):
        super().__init__(name="mlp")
        hidden_size = hf_cfg["hidden_size"]
        intermediate_size = hf_cfg["intermediate_size"]
        self.gate_proj = _make_projection("gate_proj", hidden_size, intermediate_size, False, lora_cfg, dtype_)
        self.up_proj = _make_projection("up_proj", hidden_size, intermediate_size, False, lora_cfg, dtype_)
        self.down_proj = _make_projection("down_proj", intermediate_size, hidden_size, False, lora_cfg, dtype_)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        gate = tf.nn.silu(self.gate_proj(x, training=training))
        up = self.up_proj(x, training=training)
        return self.down_proj(gate * up, training=training)


class QwenDecoderLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        hf_cfg: dict,
        lora_cfg: LoraBaseConfig,
        layer_idx: int,
        dtype_: tf.DType,
        has_qk_norm: bool,
        has_qkv_bias: bool,
    ):
        super().__init__(name=f"layers.{layer_idx}")
        eps = hf_cfg.get("rms_norm_eps", 1e-6)
        self.input_layernorm = RMSNorm(hf_cfg["hidden_size"], eps=eps, dtype_=dtype_, name="input_layernorm")
        self.self_attn = QwenAttention(hf_cfg, lora_cfg, dtype_, has_qk_norm, has_qkv_bias)
        self.post_attention_layernorm = RMSNorm(hf_cfg["hidden_size"], eps=eps, dtype_=dtype_, name="post_attention_layernorm")
        self.mlp = QwenMLP(hf_cfg, lora_cfg, dtype_)

    def call(self, x: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor, attn_mask: tf.Tensor, training: bool = False) -> tf.Tensor:
        residual = x
        x = self.self_attn(self.input_layernorm(x), cos, sin, attn_mask, training=training)
        x = residual + x

        residual = x
        x = self.mlp(self.post_attention_layernorm(x), training=training)
        return residual + x


class QwenForCausalLM(tf.keras.Model):
    def __init__(
        self,
        hf_cfg: dict,
        lora_cfg: LoraBaseConfig,
        dtype_: tf.DType,
        has_qk_norm: bool,
        has_qkv_bias: bool,
        tie_embeddings: bool,
    ):
        super().__init__()
        self.hf_cfg = hf_cfg
        self.hidden_size = hf_cfg["hidden_size"]
        self.rope_theta = hf_cfg.get("rope_theta", 10000.0)
        self.head_dim = hf_cfg.get("head_dim", self.hidden_size // hf_cfg["num_attention_heads"])
        self.tie_embeddings = tie_embeddings

        self.embed_tokens = FrozenEmbedding(hf_cfg["vocab_size"], self.hidden_size, dtype_=dtype_, name="embed_tokens")
        self.layers_list = [
            QwenDecoderLayer(hf_cfg, lora_cfg, i, dtype_, has_qk_norm, has_qkv_bias)
            for i in range(hf_cfg["num_hidden_layers"])
        ]
        self.norm = RMSNorm(self.hidden_size, eps=hf_cfg.get("rms_norm_eps", 1e-6), dtype_=dtype_, name="norm")
        self.lm_head = (
            None
            if tie_embeddings
            else FrozenDense(self.hidden_size, hf_cfg["vocab_size"], use_bias=False, dtype_=dtype_, name="lm_head")
        )

    def call(self, input_ids: tf.Tensor, attention_mask: tf.Tensor | None = None, training: bool = False) -> tf.Tensor:
        seq_len = tf.shape(input_ids)[1]

        x = self.embed_tokens(input_ids)
        cos, sin = compute_rope_cos_sin(seq_len, self.head_dim, self.rope_theta, dtype=tf.float32)

        causal = tf.linalg.band_part(tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0)
        attn_mask = (1.0 - causal) * _NEG_INF
        attn_mask = attn_mask[tf.newaxis, tf.newaxis, :, :]  # [1, 1, seq_len, seq_len]
        if attention_mask is not None:
            pad_mask = tf.cast(attention_mask, tf.float32)  # 1 = keep, 0 = pad
            attn_mask = attn_mask + (1.0 - pad_mask)[:, tf.newaxis, tf.newaxis, :] * _NEG_INF

        for layer in self.layers_list:
            x = layer(x, cos, sin, attn_mask, training=training)
        x = self.norm(x)

        logits = self.embed_tokens.logits(x) if self.tie_embeddings else self.lm_head(x, training=training)
        return tf.cast(logits, tf.float32)


def build_qwen_lora_model(model_cfg: TFModelConfig, lora_cfg: LoraBaseConfig) -> QwenForCausalLM:
    """Fetch a checkpoint, build the matching TF graph, and load every weight into it.

    Which optional pieces the graph needs (QK-norm, qkv bias, a separate lm_head) is read off
    the checkpoint's own state dict rather than assumed from the model name, so this works for
    both Qwen2-style and Qwen3-style dense checkpoints without a branch per family.
    """
    checkpoint_dir = fetch_checkpoint_files(model_cfg.model_name_or_path)
    hf_cfg = load_hf_config(checkpoint_dir)
    state_dict = load_state_dict(checkpoint_dir)

    dtype_ = _resolve_tf_dtype(model_cfg.dtype)
    has_qk_norm = has_key(state_dict, "self_attn.q_norm.weight")
    has_qkv_bias = has_key(state_dict, "self_attn.q_proj.bias")
    tie_embeddings = "lm_head.weight" not in state_dict

    tf.random.set_seed(lora_cfg.seed)
    model = QwenForCausalLM(
        hf_cfg,
        lora_cfg,
        dtype_=dtype_,
        has_qk_norm=has_qk_norm,
        has_qkv_bias=has_qkv_bias,
        tie_embeddings=tie_embeddings,
    )
    # Every variable is created eagerly in __init__ (no lazy build()), but running one dummy
    # forward pass here surfaces a shape mismatch from a mis-detected architecture flag right
    # away, at a clear call site, instead of confusingly deep inside a later load_weight call.
    model(tf.zeros((1, 4), dtype=tf.int32), attention_mask=tf.ones((1, 4), dtype=tf.int32))

    model.embed_tokens.load_weight(state_dict["model.embed_tokens.weight"])
    model.norm.load_weight(state_dict["model.norm.weight"])
    if not tie_embeddings:
        model.lm_head.load_weight(state_dict["lm_head.weight"])

    for i, layer in enumerate(model.layers_list):
        prefix = f"model.layers.{i}"
        layer.input_layernorm.load_weight(state_dict[f"{prefix}.input_layernorm.weight"])
        layer.post_attention_layernorm.load_weight(state_dict[f"{prefix}.post_attention_layernorm.weight"])

        attn = layer.self_attn
        attn.q_proj.load_weight(state_dict[f"{prefix}.self_attn.q_proj.weight"], state_dict.get(f"{prefix}.self_attn.q_proj.bias"))
        attn.k_proj.load_weight(state_dict[f"{prefix}.self_attn.k_proj.weight"], state_dict.get(f"{prefix}.self_attn.k_proj.bias"))
        attn.v_proj.load_weight(state_dict[f"{prefix}.self_attn.v_proj.weight"], state_dict.get(f"{prefix}.self_attn.v_proj.bias"))
        attn.o_proj.load_weight(state_dict[f"{prefix}.self_attn.o_proj.weight"])
        if has_qk_norm:
            attn.q_norm.load_weight(state_dict[f"{prefix}.self_attn.q_norm.weight"])
            attn.k_norm.load_weight(state_dict[f"{prefix}.self_attn.k_norm.weight"])

        mlp = layer.mlp
        mlp.gate_proj.load_weight(state_dict[f"{prefix}.mlp.gate_proj.weight"])
        mlp.up_proj.load_weight(state_dict[f"{prefix}.mlp.up_proj.weight"])
        mlp.down_proj.load_weight(state_dict[f"{prefix}.mlp.down_proj.weight"])

    return model


def count_trainable_parameters(model: tf.keras.Model) -> tuple[int, int]:
    trainable = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    non_trainable = sum(int(np.prod(v.shape)) for v in model.non_trainable_variables)
    return trainable, trainable + non_trainable


def print_trainable_parameters(model: tf.keras.Model) -> None:
    trainable, total = count_trainable_parameters(model)
    pct = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {pct:.4f}")
