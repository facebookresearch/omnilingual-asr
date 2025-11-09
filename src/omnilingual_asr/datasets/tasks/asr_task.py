# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from fairseq2.data.data_pipeline import (
    CollateOptionsOverride,
    Collater,
    DataPipelineBuilder,
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.error import NotSupportedError
from fairseq2.gang import Gangs
from fairseq2.logging import log
from typing_extensions import Any, Dict, List, override

from omnilingual_asr.datasets.interfaces.task_interface import TaskConfig, TaskInterface
from omnilingual_asr.datasets.utils.audio import (
    add_audio_decoding,
    add_fbank_processing,
    add_waveform_processing,
    filter_by_audio_length,
)
from omnilingual_asr.datasets.utils.batching import (
    BatchingStrategy,
    add_length_batching,
    add_static_batching,
)
from omnilingual_asr.datasets.utils.text import (
    encode_text,
    filter_empty_text,
    filter_fast_speech,
    filter_long_text,
    filter_unknown_sequences,
    filter_unknown_tokens,
)


@dataclass
class AsrTaskConfig(TaskConfig):
    """ASR-specific task configuration"""

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

    # Text processing
    filter_long_text_threshold: int | None = None
    """Filters samples that have a text longer than `filter_long_text_threshold`."""

    remove_unknown: bool = False
    """Remove unknown tokens from text inplace."""

    min_samples_per_char: int = 160
    """If a sample has more than ``sample_rate / min_samples_per_char`` chars per second, it's filtered out."""

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
    example_shuffle_window: int = 0
    """The size of the sliding window for shuffling samples (pre-batching). If `1`, no shuffling is
    performed. If `0` the complete dataset is loaded and shuffled."""

    batch_shuffle_window: int = 1000
    """The size of the sliding window for shuffling batches."""

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


class AsrTask(TaskInterface[AsrTaskConfig]):
    """``AsrTask`` defines the complete data preprocessing pipeline for ASR training,
    including audio preprocessing, text tokenization and batching strategies.

    Steps: audio filtering → shuffling → text tokenization → batching
         → batch shuffling → audio features → Seq2SeqBatch
    """

    def __init__(self, config: AsrTaskConfig):
        super().__init__(config)

    @override
    def apply_processing_pipeline(  # type: ignore[override]
        self,
        builder: DataPipelineBuilder,
        gangs: Gangs,
        tokenizer: Tokenizer,
        dtype: torch.dtype,
    ) -> DataPipelineBuilder:

        config = self.config

        # Filtering audio to optimize before batching
        builder = filter_by_audio_length(
            builder,
            min_audio_len=config.min_audio_len,
            max_audio_len=config.max_audio_len,
            length_selector="length",
        )

        # Shuffle dataset pre-batching
        builder = AsrTask.add_example_shuffling(
            builder,
            example_shuffle_window=config.example_shuffle_window,
            seed=config.seed,
        )

        config.seed += 1

        # Tokenize, filter unknown tokens, long target text and too fast speech
        builder = AsrTask.add_tokenization_pipeline(
            builder,
            tokenizer,
            filter_long_text_threshold=config.filter_long_text_threshold,
            remove_unknown=config.remove_unknown,
            min_samples_per_char=config.min_samples_per_char,
            text_selector="text",
            audio_length_selector="length",
        )

        # Bucket examples by audio length.
        builder = AsrTask.add_bucketing_pipeline(
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

        # Shuffle inter-batch
        builder = AsrTask.add_batch_shuffling(
            builder, batch_shuffle_window=config.batch_shuffle_window, seed=config.seed
        )

        # Task specific audio processing
        builder = AsrTask.add_audio_processing_pipeline(
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
        )

        # Collate, limit batch count, prefetch, convert to Seq2Seq
        return AsrTask.add_postprocessing_pipeline(
            builder,
            text_selector="text",
            pad_idx=tokenizer.vocab_info.pad_idx,  # type: ignore
            npc=config.npc,
            max_num_batches=config.max_num_batches,
            num_prefetch=config.num_prefetch,
            no_padding=config.no_padding,
        )

    @override
    def get_batch_type(self) -> type:
        return Seq2SeqBatch

    @staticmethod
    def add_tokenization_pipeline(
        builder: DataPipelineBuilder,
        tokenizer: Tokenizer,
        filter_long_text_threshold: int | None,
        remove_unknown: bool,
        min_samples_per_char: int,
        text_selector: str,
        audio_length_selector: str,
    ) -> DataPipelineBuilder:

        builder = filter_empty_text(builder, text_selector=text_selector)

        builder = filter_fast_speech(
            builder,
            min_samples_per_char=min_samples_per_char,
            text_selector=text_selector,
            audio_length_selector=audio_length_selector,
        )

        builder = encode_text(
            builder,
            text_encoder=tokenizer.create_encoder(),
            text_selector=text_selector,
        )

        builder = filter_unknown_sequences(
            builder, unk_idx=tokenizer.vocab_info.unk_idx, text_selector=text_selector  # type: ignore
        )

        if filter_long_text_threshold is not None:
            builder = filter_long_text(
                builder,
                threshold=filter_long_text_threshold,
                text_selector=text_selector,
            )

        if remove_unknown:
            builder = filter_unknown_tokens(
                builder, unk_idx=tokenizer.vocab_info.unk_idx  # type: ignore
            )
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
    def add_example_shuffling(
        builder: DataPipelineBuilder, example_shuffle_window: int, seed: int
    ) -> DataPipelineBuilder:
        """Shuffles samples (pre-batching operation)."""
        assert (
            example_shuffle_window > 0
        ), "Shuffling the entire dataset can result in OOM, set `example_shuffle_window` > 0 to shuffle inside a window."

        if example_shuffle_window != 1:
            builder.shuffle(example_shuffle_window, seed)
        return builder

    @staticmethod
    def add_batch_shuffling(
        builder: DataPipelineBuilder, batch_shuffle_window: int, seed: int
    ) -> DataPipelineBuilder:
        """Shuffles batches, not samples (post-batching operation)."""
        assert (
            batch_shuffle_window > 0
        ), "Shuffling the entire dataset can result in OOM, set `batch_shuffle_window` > 0 to shuffle inside a window."

        if batch_shuffle_window != 1:
            builder.shuffle(batch_shuffle_window, seed)
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

        return AsrTask.add_unified_naming(
            builder, unified_audio_feature_keys=unified_audio_feature_keys
        )

    @staticmethod
    def add_unified_naming(
        builder: DataPipelineBuilder, unified_audio_feature_keys: bool
    ) -> DataPipelineBuilder:
        def unify_audio_features(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            for example in batch:
                if "fbank" in example["audio"]:
                    example["audio_feature"] = example["audio"].pop("fbank")
                elif "waveform" in example["audio"]:
                    example["audio_feature"] = example["audio"].pop("waveform")
            return batch

        if unified_audio_feature_keys:
            builder.map(unify_audio_features)
        return builder

    @staticmethod
    def add_postprocessing_pipeline(
        builder: DataPipelineBuilder,
        text_selector: str,
        pad_idx: int,
        npc: int,
        max_num_batches: int | None,
        num_prefetch: int,
        no_padding: bool,
    ) -> DataPipelineBuilder:

        if no_padding:
            log.warning(
                "Collating without padding is currently not supported, defaulting to padding."
            )
        no_padding = False

        # Collate bucketed examples into a batch.
        text_collate_opts = CollateOptionsOverride(text_selector, pad_value=pad_idx)

        collater = Collater(
            pad_value=None if no_padding else 0, overrides=[text_collate_opts]
        )

        builder.map(collater, num_parallel_calls=npc)

        # Return only the first `max_num_batches`.
        if max_num_batches is not None:
            builder.take(max_num_batches)

        # Prefetch `num_prefetch` batches in background.
        builder.prefetch(num_prefetch)

        # Wrap examples with `Seq2SeqBatch`.
        return builder.map(AsrTask.to_seq2seq_batch)

    @staticmethod
    def to_seq2seq_batch(example: Dict[str, Any]) -> Seq2SeqBatch:
        """Convert collated example to Seq2SeqBatch."""
        audio_data = example["audio_feature"]
        text_data = example["text"]

        return Seq2SeqBatch(
            source_seqs=audio_data["seqs"],
            source_seq_lens=audio_data["seq_lens"],
            target_seqs=text_data["seqs"],
            target_seq_lens=text_data["seq_lens"],
            example=example,
        )
