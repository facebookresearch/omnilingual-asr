# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum

from fairseq2.data.data_pipeline import DataPipelineBuilder, create_bucket_sizes
from fairseq2.logging import log


class BatchingStrategy(Enum):
    """Batching strategies"""

    STATIC = "STATIC"
    LENGTH = "LENGTH"


def add_static_batching(
    builder: DataPipelineBuilder, batch_size: int, drop_remainder: bool
) -> DataPipelineBuilder:
    """Add static batching to pipeline."""
    return builder.bucket(batch_size, drop_remainder=drop_remainder)


def add_length_batching(
    builder: DataPipelineBuilder,
    min_audio_len: int,
    max_audio_len: int,
    max_num_elements: int,
    num_seqs_multiple_of: int,
    drop_remainder: bool,
    max_bucket_size: int | None,
    selector: str,
) -> DataPipelineBuilder:
    """Add length-based batching to pipeline."""
    log.info(f"Using length batching with max_num_elements={max_num_elements}!")

    if max_num_elements % max_audio_len != 0:
        max_num_elements = (max_num_elements // max_audio_len) * max_audio_len
        log.warning(f"`max_num_elements` is rounded to {max_num_elements}")

    bucket_sizes = create_bucket_sizes(
        min_seq_len=min_audio_len,
        max_seq_len=max_audio_len,
        max_num_elements=max_num_elements,
        num_seqs_multiple_of=num_seqs_multiple_of,
    )
    # Comply with the max batch size (bucket size) constraint
    final_bucket_sizes = bucket_sizes
    if max_bucket_size is not None:
        final_bucket_sizes = []
        for bucket_size, seq_len in bucket_sizes:
            if bucket_size <= max_bucket_size:
                final_bucket_sizes.append((bucket_size, seq_len))

    return builder.bucket_by_length(
        final_bucket_sizes,
        selector=selector,
        min_data_len=min_audio_len,
        skip_below_min_examples=True,  # this should be neutral
        skip_above_max_examples=True,  # this should be neutral
        drop_remainder=drop_remainder,
    )
