"""Bridges tiny_lora's framework-agnostic dataset prep into `tf.data.Dataset`s.

`tiny_lora.data.prepare_sft_dataset`/`prepare_sft_eval_dataset` already do all the dataset
loading and chat-template rendering -- they hand back a HF `datasets.Dataset` with a plain
`text` column, which is exactly as framework-agnostic on the way in as it needs to be. Only the
tokenize-and-batch step below is TF-specific.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from datasets import Dataset

from tiny_lora.config import DataConfig
from tiny_lora.data import prepare_sft_dataset, prepare_sft_eval_dataset

# Matches HF's `ForCausalLMLoss` convention: labels at this value are excluded from the loss.
IGNORE_INDEX = -100


def tokenize_for_causal_lm(dataset: Dataset, tokenizer, max_seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded, truncated `input_ids`/`attention_mask` for every row's `text` column.

    Forces right-padding regardless of the tokenizer's own default (some chat tokenizers default
    to left-padding for generation) -- training needs pad tokens trailing the real content so the
    causal mask and the label masking below line up.
    """
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        dataset["text"],
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="np",
    )
    return encoded["input_ids"].astype(np.int32), encoded["attention_mask"].astype(np.int32)


def build_tf_dataset(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> tf.data.Dataset:
    """`input_ids`/`attention_mask` -> a batched dataset of `({input_ids, attention_mask}, labels)`."""
    labels = np.where(attention_mask.astype(bool), input_ids, IGNORE_INDEX)
    ds = tf.data.Dataset.from_tensor_slices(
        ({"input_ids": input_ids, "attention_mask": attention_mask}, labels)
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(input_ids), reshuffle_each_iteration=True)
    return ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)


def prepare_tf_sft_datasets(
    data_cfg: DataConfig,
    tokenizer,
    max_seq_length: int,
    train_batch_size: int,
    eval_batch_size: int,
) -> tuple[tf.data.Dataset, tf.data.Dataset | None, int]:
    """Returns `(train_dataset, eval_dataset_or_None, num_train_examples)`."""
    train_dataset = prepare_sft_dataset(data_cfg, tokenizer)
    train_ids, train_mask = tokenize_for_causal_lm(train_dataset, tokenizer, max_seq_length)
    train_tf = build_tf_dataset(train_ids, train_mask, train_batch_size, shuffle=True)

    eval_dataset = prepare_sft_eval_dataset(data_cfg, tokenizer)
    eval_tf = None
    if eval_dataset is not None:
        eval_ids, eval_mask = tokenize_for_causal_lm(eval_dataset, tokenizer, max_seq_length)
        eval_tf = build_tf_dataset(eval_ids, eval_mask, eval_batch_size, shuffle=False)

    return train_tf, eval_tf, len(train_ids)
