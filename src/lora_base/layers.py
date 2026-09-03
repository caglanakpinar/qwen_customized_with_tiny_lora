"""From-scratch building blocks: RMSNorm, rotary embeddings, GQA's repeat_kv, and LoRA.

No `peft` anywhere here -- `LoRADense` is a plain `tf.keras.layers.Layer` holding its own frozen
base weight plus trainable `lora_A`/`lora_B`, matching the LoRA paper's formulation:
`y = xW + scaling * x @ A @ B`, `A` small-random-init, `B` zero-init so the adapter starts as a
no-op.
"""

from __future__ import annotations

import math

import numpy as np
import tensorflow as tf


class RMSNorm(tf.keras.layers.Layer):
    """Qwen/Llama-style RMSNorm: normalize in float32, scale by a frozen weight."""

    def __init__(self, dim: int, eps: float = 1e-6, dtype_: tf.DType = tf.float32, name: str | None = None):
        super().__init__(name=name)
        self.eps = eps
        self.weight = tf.Variable(tf.ones((dim,), dtype=dtype_), trainable=False, name="weight")

    def load_weight(self, weight: np.ndarray) -> None:
        self.weight.assign(tf.cast(tf.constant(weight), self.weight.dtype))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        input_dtype = x.dtype
        x32 = tf.cast(x, tf.float32)
        variance = tf.reduce_mean(tf.square(x32), axis=-1, keepdims=True)
        x32 = x32 * tf.math.rsqrt(variance + self.eps)
        return tf.cast(self.weight, input_dtype) * tf.cast(x32, input_dtype)


def compute_rope_cos_sin(seq_len: int, head_dim: int, theta: float, dtype: tf.DType = tf.float32):
    """Precompute the cos/sin tables RoPE needs for positions `0..seq_len-1`."""
    inv_freq = 1.0 / (theta ** (tf.range(0, head_dim, 2, dtype=tf.float32) / head_dim))
    positions = tf.range(seq_len, dtype=tf.float32)
    freqs = tf.einsum("i,j->ij", positions, inv_freq)  # [seq_len, head_dim/2]
    emb = tf.concat([freqs, freqs], axis=-1)  # [seq_len, head_dim]
    return tf.cast(tf.cos(emb), dtype), tf.cast(tf.sin(emb), dtype)


def _rotate_half(x: tf.Tensor) -> tf.Tensor:
    x1, x2 = tf.split(x, 2, axis=-1)
    return tf.concat([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q: tf.Tensor, k: tf.Tensor, cos: tf.Tensor, sin: tf.Tensor):
    """Rotate `q`/`k` (`[batch, heads, seq_len, head_dim]`) by the `[seq_len, head_dim]` tables."""
    cos = cos[tf.newaxis, tf.newaxis, :, :]
    sin = sin[tf.newaxis, tf.newaxis, :, :]
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(x: tf.Tensor, n_rep: int) -> tf.Tensor:
    """Tile grouped-query attention's key/value heads up to the query head count.

    `x`: `[batch, num_kv_heads, seq_len, head_dim]` -> `[batch, num_kv_heads * n_rep, seq_len,
    head_dim]`, each kv head repeated `n_rep` times contiguously (matching HF's `repeat_kv`).
    """
    if n_rep == 1:
        return x
    shape = tf.shape(x)
    batch, num_kv_heads, seq_len, head_dim = shape[0], shape[1], shape[2], shape[3]
    x = tf.expand_dims(x, axis=2)
    x = tf.tile(x, [1, 1, n_rep, 1, 1])
    return tf.reshape(x, [batch, num_kv_heads * n_rep, seq_len, head_dim])


class FrozenDense(tf.keras.layers.Layer):
    """A `y = xW (+ b)` linear layer with a non-trainable weight, loaded from a checkpoint.

    Used for every projection not in `lora_base.target_modules` -- the base model stays frozen,
    only `LoRADense` layers below train anything.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = False,
        dtype_: tf.DType = tf.float32,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tf.Variable(
            tf.zeros((in_features, out_features), dtype=dtype_), trainable=False, name="weight"
        )
        self.bias = (
            tf.Variable(tf.zeros((out_features,), dtype=dtype_), trainable=False, name="bias")
            if use_bias
            else None
        )

    def load_weight(self, weight_out_in: np.ndarray, bias: np.ndarray | None = None) -> None:
        """`weight_out_in` is a PyTorch `nn.Linear.weight`, shaped `[out_features, in_features]`

        -- transposed here since this layer computes `x @ weight` with `weight` shaped
        `[in_features, out_features]`.
        """
        self.weight.assign(tf.cast(tf.constant(np.ascontiguousarray(weight_out_in.T)), self.weight.dtype))
        if bias is not None:
            if self.bias is None:
                raise ValueError(f"{self.name}: got a bias array but this layer has use_bias=False")
            self.bias.assign(tf.cast(tf.constant(bias), self.bias.dtype))

    def call(self, x: tf.Tensor) -> tf.Tensor:
        out = tf.matmul(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class FrozenEmbedding(tf.keras.layers.Layer):
    """Token embedding table, frozen, doubling as the tied `lm_head` when the checkpoint ties them.

    Unlike `FrozenDense`, a PyTorch `nn.Embedding.weight` is already `[vocab_size, hidden_size]`
    -- the same orientation this layer stores it in -- so no transpose happens on load.
    """

    def __init__(self, vocab_size: int, hidden_size: int, dtype_: tf.DType = tf.float32, name: str | None = None):
        super().__init__(name=name)
        self.weight = tf.Variable(
            tf.zeros((vocab_size, hidden_size), dtype=dtype_), trainable=False, name="weight"
        )

    def load_weight(self, weight: np.ndarray) -> None:
        self.weight.assign(tf.cast(tf.constant(weight), self.weight.dtype))

    def call(self, ids: tf.Tensor) -> tf.Tensor:
        return tf.gather(self.weight, ids)

    def logits(self, x: tf.Tensor) -> tf.Tensor:
        """`x @ weight^T` -- the tied lm_head projection back to vocab size."""
        return tf.matmul(x, tf.cast(self.weight, x.dtype), transpose_b=True)


class LoRADense(FrozenDense):
    """`FrozenDense` plus a trainable low-rank update: `y = xW (+ b) + scaling * x @ A @ B`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        alpha: int,
        dropout: float = 0.0,
        use_bias: bool = False,
        dtype_: tf.DType = tf.float32,
        seed: int = 42,
        name: str | None = None,
    ):
        super().__init__(in_features, out_features, use_bias=use_bias, dtype_=dtype_, name=name)
        self.r = r
        self.scaling = alpha / r
        # Matches PyTorch's default nn.Linear init (kaiming_uniform_ with a=sqrt(5)), which for
        # a [in_features, r] tensor reduces to Uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        # -- the same init LoRA papers/PEFT use for the A matrix. B is zero so training starts
        # as a no-op: the adapter's initial output exactly matches the frozen base model.
        limit = 1.0 / math.sqrt(in_features)
        self.lora_A = tf.Variable(
            tf.random.uniform((in_features, r), minval=-limit, maxval=limit, seed=seed, dtype=tf.float32),
            trainable=True,
            name="lora_A",
        )
        self.lora_B = tf.Variable(tf.zeros((r, out_features), dtype=tf.float32), trainable=True, name="lora_B")
        self._dropout = tf.keras.layers.Dropout(dropout, seed=seed) if dropout > 0 else None

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        base = super().call(x)
        lora_input = self._dropout(x, training=training) if self._dropout is not None else x
        lora_input = tf.cast(lora_input, tf.float32)
        lora_out = tf.matmul(tf.matmul(lora_input, self.lora_A), self.lora_B) * self.scaling
        return base + tf.cast(lora_out, base.dtype)
