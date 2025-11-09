# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Dict

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.data.tokenizers import TokenEncoder


def filter_empty_text(
    builder: DataPipelineBuilder, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `text_selector` field."""

    def is_not_empty(example: Dict[str, Any]) -> bool:
        return len(example[text_selector]) > 0

    return builder.filter(is_not_empty)


def encode_text(
    builder: DataPipelineBuilder, text_encoder: TokenEncoder, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `text_selector` field."""
    return builder.map(text_encoder, selector=text_selector)


def filter_unknown_sequences(
    builder: DataPipelineBuilder, unk_idx: int, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `text_selector` field and that field being a `torch.tensor`"""

    def is_not_unknown_sequence(example: Dict[str, Any]) -> bool:
        return bool((example[text_selector] != unk_idx).sum().item() > 0)

    return builder.filter(is_not_unknown_sequence)


def filter_long_text(
    builder: DataPipelineBuilder, threshold: int, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `text_selector` field and that field being a `torch.tensor`"""

    def is_below_length_threshold(example: Dict[str, Any]) -> bool:
        return example[text_selector].numel() <= threshold

    return builder.filter(is_below_length_threshold)


def filter_unknown_tokens(
    builder: DataPipelineBuilder, unk_idx: int, text_selector: str
) -> DataPipelineBuilder:
    """Expects the data to have a `'text'` field and that field being a `torch.tensor`.
    Modifies the tensor inplace.
    """
    return builder.map(lambda tensor: tensor[tensor != unk_idx], selector=text_selector)


def filter_fast_speech(
    builder: DataPipelineBuilder,
    min_samples_per_char: int,
    text_selector: str,
    audio_length_selector: str,
) -> DataPipelineBuilder:
    """Filters samples that have too few samples per character, indicating too fast speech. Expects the data to have a `text_selector` and `audio_length_selector` field."""

    def is_slow_enough_speech(example: Dict[str, Any]) -> bool:
        return (
            example[audio_length_selector] / len(example[text_selector])
            >= min_samples_per_char
        )

    return builder.filter(is_slow_enough_speech)
