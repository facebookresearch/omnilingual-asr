# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List

import torch
from fairseq2.data.data_pipeline import Collater, DataPipelineBuilder
from fairseq2.datasets import SequenceBatch
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from typing_extensions import override

from omnilingual_asr.datasets.interfaces.task_interface import TaskConfig, TaskInterface
from omnilingual_asr.datasets.utils.audio import (
    add_audio_cropping,
    add_audio_decoding,
    add_fbank_processing,
    add_waveform_processing,
)
from omnilingual_asr.datasets.utils.batching import (
    BatchingStrategy,
    add_length_batching,
    add_static_batching,
)


@dataclass
class SslTaskConfig(TaskConfig):
    """SSL-specific task configuration"""

    min_audio_len: int = 1
    """The minimum audio sequence length."""

    max_audio_len: int = 800_000
    """The maximum audio sequence length."""

    # Batching parameters
    batching_strategy: BatchingStrategy = BatchingStrategy.LENGTH
    """Batching strategy is defined through an enum:
    - BatchingStrategy.LENGTH ("length") = Specifies batching where each batch has a maximum number of elements.
    - BatchingStrategy.STATIC ("static") = Specifies batching where each batch has the same number of examples.
    """

    batch_size: int = 8
    """If `batching_strategy = BatchingStrategy.STATIC`, ignores `max_num_elements` and each batch will have `batch_size` examples.
    """

    num_seqs_multiple_of: int = 8
    """If `batching_strategy = BatchingStrategy.LENGTH, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multiple_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_num_elements: int = 3_200_000
    """If `batching_strategy = BatchingStrategy.LENGTH`, ignores `batch_size` and each batch will have
    `<= max_num_elements` elements with `<= num_seqs_multiple_of` or `num_seqs_multiple_of * N` sequences.
    This is primarily for hardware optimization, but will only work when sufficient enough sequences are available for bucketing.
    """

    max_bucket_size: int | None = None
    """If `batching_strategy = BatchingStrategy.LENGTH`, limits the amount of buckets to `max_bucket_size`. This will essentially limit buckets to consist of shorter rather than longer sequence lengths due to `fairseq2.data.data_pipeline.create_bucket_sizes` starting with bucketing the shortest sequences lengths first.
    """

    drop_remainder: bool = False
    """If ``True``, drops the last set of batches if they have in total fewer examples than requested (for both `BatchingStrategy.[STATIC | LENGTH]`)."""

    # Audio processing
    normalize_audio: bool = False
    """If ``True``, normalizes audio to have zero mean and unit variance."""

    use_fbank: bool = False
    """If ``True``, use fbank features instead of waveform."""

    # SpecAugment
    spec_aug_p: float | None = None
    """Probability of applying SpecAugment per row."""

    spec_aug_freq_mask_param: int = 80
    """Maximum frequency mask length."""

    spec_aug_time_mask_param: int = 80
    """Maximum time mask length."""

    unified_audio_feature_keys: bool = True
    """Unify the feature keys for different audio processing strategies `use_fbank=[True | False]`."""

    # Misc
    example_shuffle_window: int = 500_000
    """The size of the sliding window for shuffling examples (pre-batch). `example_shuffle_window=0` shuffles the entire dataset."""

    npc: int = 10
    """The number of parallel calls to use in the pipeline."""

    seed: int = 2
    """The seed to initialize the random number generators used internally."""

    no_padding: bool = False
    """If ``True``, all elements in the batch will be truncated by batch minimal length.
    Therefore, no padding will be applied to the batch.
    """

    max_num_batches: int | None = None
    """The maximum number of batches to return."""

    num_prefetch: int = 4
    """The number of batches to prefetch in background."""


class SslTask(TaskInterface[SslTaskConfig]):
    """``SslTask`` defines the complete data preprocessing pipeline for self-supervised learning.

    Audio-only processing without text labels, supports both waveform and filterbank features
    with configurable cropping for consistent sequence lengths.

    Steps: shuffling → audio filtering → batching → audio cropping → features → SequenceBatch.
    """

    def __init__(self, config: SslTaskConfig):
        super().__init__(config)

    @override
    def apply_processing_pipeline(  # type: ignore[override]
        self,
        builder: DataPipelineBuilder,
        gangs: Gangs,
        dtype: torch.dtype,
    ) -> DataPipelineBuilder:

        config = self.config

        # Shuffle individual samples
        builder = SslTask.add_example_shuffling(
            builder,
            example_shuffle_window=config.example_shuffle_window,
            seed=config.seed,
        )

        config.seed += 1

        builder = SslTask.shard_across_distributed_procs(builder, gangs=gangs)

        config.seed += gangs.dp.rank

        # Bucket examples by audio length.
        builder = SslTask.add_bucketing_pipeline(
            builder,
            batching=config.batching_strategy,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            max_num_elements=config.max_num_elements,
            num_seqs_multiple_of=config.num_seqs_multiple_of,
            drop_remainder=config.drop_remainder,
            max_bucket_size=config.max_bucket_size,
            length_selector="length",
            batch_size=config.batch_size,
            no_padding=config.no_padding,
        )

        # Task specific audio processing
        builder = SslTask.add_audio_processing_pipeline(
            builder,
            dtype=dtype,
            normalize_audio=config.normalize_audio,
            audio_selector="[*].audio",
            npc=config.npc,
            use_fbank=config.use_fbank,
            spec_aug_p=config.spec_aug_p,
            spec_aug_freq_mask_param=config.spec_aug_freq_mask_param,
            spec_aug_time_mask_param=config.spec_aug_time_mask_param,
            unified_audio_feature_keys=config.unified_audio_feature_keys,
            max_audio_len=config.max_audio_len,
            no_padding=config.no_padding,
            seed=config.seed,
        )

        # Collate, limit batch count, prefetch, convert to SequenceBatch
        return SslTask.add_postprocessing_pipeline(
            builder,
            npc=config.npc,
            max_num_batches=config.max_num_batches,
            num_prefetch=config.num_prefetch,
            no_padding=config.no_padding,
        )

    @staticmethod
    def add_example_shuffling(
        builder: DataPipelineBuilder, example_shuffle_window: int, seed: int
    ) -> DataPipelineBuilder:
        # Shuffle the dataset samples
        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)
        return builder

    @staticmethod
    def shard_across_distributed_procs(
        builder: DataPipelineBuilder, gangs: Gangs
    ) -> DataPipelineBuilder:
        if gangs.dp.size > 1:
            builder.shard(gangs.dp.rank, gangs.dp.size, allow_uneven=True)
        return builder

    @staticmethod
    def add_bucketing_pipeline(
        builder: DataPipelineBuilder,
        batching: BatchingStrategy,
        min_audio_len: int,
        max_audio_len: int,
        max_num_elements: int,
        num_seqs_multiple_of: int,
        drop_remainder: bool,
        max_bucket_size: int | None,
        length_selector: str,
        batch_size: int,
        no_padding: bool,
    ) -> DataPipelineBuilder:

        if batching is BatchingStrategy.LENGTH:
            builder = add_length_batching(
                builder,
                min_audio_len=min_audio_len,
                max_audio_len=max_audio_len,
                max_num_elements=max_num_elements,
                num_seqs_multiple_of=num_seqs_multiple_of,
                drop_remainder=drop_remainder,
                max_bucket_size=max_bucket_size,
                selector=length_selector,
            )
        elif batching is BatchingStrategy.STATIC:
            if no_padding:
                raise NotSupportedError(
                    "no_padding is not supported for static batching"
                )
            builder = add_static_batching(
                builder, batch_size=batch_size, drop_remainder=drop_remainder
            )
        else:
            raise NotSupportedError(
                f"`{batching}` is not supported. Options: {[o.value for o in BatchingStrategy]}"
            )
        return builder

    @staticmethod
    def add_audio_processing_pipeline(
        builder: DataPipelineBuilder,
        use_fbank: bool,
        audio_selector: str,
        dtype: torch.dtype,
        normalize_audio: bool,
        spec_aug_p: float | None,
        spec_aug_freq_mask_param: int,
        spec_aug_time_mask_param: int,
        npc: int,
        unified_audio_feature_keys: bool,
        max_audio_len: int,
        no_padding: bool,
        seed: int,
    ) -> DataPipelineBuilder:

        builder = add_audio_decoding(
            builder,
            dtype=dtype,
            normalize_audio=normalize_audio,
            selector=audio_selector,
            npc=npc,
        )

        if use_fbank:
            builder = add_fbank_processing(
                builder, dtype=dtype, selector=audio_selector, npc=npc
            )
        else:
            builder = add_waveform_processing(
                builder,
                normalize_audio=normalize_audio,
                dtype=dtype,
                selector=audio_selector + ".waveform",
                spec_aug_p=spec_aug_p,
                spec_aug_freq_mask_param=spec_aug_freq_mask_param,
                spec_aug_time_mask_param=spec_aug_time_mask_param,
            )

        builder = SslTask.add_unified_naming(
            builder, unified_audio_feature_keys=unified_audio_feature_keys
        )

        return add_audio_cropping(
            builder,
            max_audio_len=max_audio_len,
            crop_to_batch_minimal_size=no_padding,
            audio_selector="audio_selector",
            seed=seed,
        )

    @staticmethod
    def add_unified_naming(
        builder: DataPipelineBuilder, unified_audio_feature_keys: bool
    ) -> DataPipelineBuilder:
        def unify_audio_features(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            for example in batch:
                if "fbank" in example["audio"]["data"]:
                    example["audio_feature"] = example["audio"]["data"].pop("fbank")
                elif "waveform" in example["audio"]["data"]:
                    example["audio_feature"] = example["audio"]["data"].pop("waveform")
            return batch

        if unified_audio_feature_keys:
            builder.map(unify_audio_features)
        return builder

    @staticmethod
    def add_postprocessing_pipeline(
        builder: DataPipelineBuilder,
        npc: int,
        max_num_batches: int | None,
        num_prefetch: int,
        no_padding: bool,
    ) -> DataPipelineBuilder:

        collater = Collater(pad_value=None if no_padding else 0)

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap in SequenceBatch
        return builder.map(
            partial(SslTask.create_sequence_batch, no_padding=no_padding)
        )

    @staticmethod
    def create_sequence_batch(
        batch_dict: Dict[str, Any], no_padding: bool
    ) -> SequenceBatch:
        """
        Convert batch dictionary to SequenceBatch with proper sequence length handling.
        """
        audio_feature = batch_dict["audio_feature"]

        if no_padding:
            # no_padding=True: All sequences cropped to same length, no padding needed
            # audio_feature is a plain Tensor from Collater(pad_value=None)
            return SequenceBatch(audio_feature, seq_lens=None, example=batch_dict)
        else:
            # no_padding=False: Sequences are padded, need actual sequence lengths
            # audio_feature is SequenceData dict from Collater(pad_value=0)
            if isinstance(audio_feature, dict) and "seq_lens" in audio_feature:
                seqs = audio_feature["seqs"]
                seq_lens = audio_feature["seq_lens"]

                return SequenceBatch(seqs, seq_lens=seq_lens, example=batch_dict)
            else:
                # Fallback: assume uniform lengths (should not happen with proper Collater setup)
                return SequenceBatch(audio_feature, seq_lens=None, example=batch_dict)  # type: ignore
