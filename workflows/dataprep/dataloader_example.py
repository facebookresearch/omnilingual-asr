# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DataLoader example for MixtureParquetAsrDataset.

This module provides a simple utility for testing MixtureParquetAsrDataset
with proper configuration and iteration examples.
"""

from collections import Counter
from pathlib import Path
from typing import Tuple

import fire
import torch
from fairseq2.data.parquet.fragment_loading import FragmentLoadingConfig
from fairseq2.data.parquet.fragment_streaming import FragmentStreamingConfig
from fairseq2.data.tokenizers import load_tokenizer
from fairseq2.datasets import SyncMode
from fairseq2.gang import create_fake_gangs

from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
    MixtureParquetAsrDataset,
)
from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    LangASRSchema,
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig


class DataLoaderExample:
    """Example class for building and testing MixtureParquetAsrDataset."""

    def test_dataset(
        self,
        dataset_path: str,
        split: str = "train",
        tokenizer_name: str = "omniASR_tokenizer",
        num_iterations: int = 10,
        stats_path: str | None = None,
        device: str = "cpu",
    ):
        """
        Build and test MixtureParquetAsrDataset with iterations.

        Args:
            dataset_path: Path to the dataset directory (e.g., "all_asr/version=0")
            split: Split to use (e.g., "train", "dev", "test")
            tokenizer_name: Name of tokenizer to use (e.g., "nllb-200")
            num_iterations: Number of batches to iterate over
            batch_size: Batch size for iterations
            stats_path: Optional path to dataset statistics file for weighting
            device: Device to use for computation ("cpu" or "cuda")
        """
        print(f"🚀 Building MixtureParquetAsrDataset from: {dataset_path}")
        print(f"📊 Split: {split}, Tokenizer: {tokenizer_name}")

        # Create the dataset
        dataset = MixtureParquetAsrDataset.from_path(Path(dataset_path))

        # Load tokenizer
        tokenizer = load_tokenizer(tokenizer_name)
        print(f"🔤 Loaded tokenizer: {tokenizer.vocab_info}")

        # Configure storage
        storage_config = MixtureParquetStorageConfig(
            fragment_streaming=FragmentStreamingConfig(
                parquet_path="",  # Will be set automatically
                name=f"{split}_streaming",
                partition_filters=None,
                seed=42,
                fragment_shuffle_window=-1,  # shuffle all fragments=row groups globally
                nb_epochs=None,  # inifiinte loop
            ),
            fragment_loading=FragmentLoadingConfig(
                columns=LangASRSchema(),
                non_deterministic_read=False,
                add_fragment_traces=False,
                nb_prefetch=0,
                num_parallel_fragments=1,
            ),
            dataset_summary_path=stats_path,
            beta_corpus=0.5 if stats_path else None,
            beta_language=0.5 if stats_path else None,
            sync_mode=SyncMode.UNTIL_FIRST if "train" in split else SyncMode.UNTIL_LAST,
            sync_batches=True,
        )

        # Configure task
        task_config = AsrTaskConfig(
            min_audio_len=8000,  # ~0.5s at 16kHz
            max_audio_len=800_000,  # ~50s at 16kHz
            max_num_elements=1_600_000,  # to avoid OOM
            num_seqs_multiple_of=8,
            normalize_audio=False,
            example_shuffle_window=1000,  # No shuffling for testing
            batch_shuffle_window=20,  # No batch shuffling for testing
            max_num_batches=num_iterations,  # Limit iterations
            num_prefetch=2,
            seed=202510,
        )

        # Create reader
        print("🔧 Creating data reader...")
        reader = dataset.create_reader(
            split=split,
            tokenizer=tokenizer,
            gangs=create_fake_gangs(device=torch.device(device)),
            dtype=torch.float32,
            num_accumulate=1,
            storage_config=storage_config,
            task_config=task_config,
        )

        print(f"🔄 Starting iterations (max {num_iterations} batches)...")

        # Initialize counter for (corpus, language) tuples
        corpus_language_counter: Counter[Tuple[str, str]] = Counter()

        # Iterate and collect corpus/language statistics
        total_samples = 0
        total_audio_elements = 0
        total_text_elements = 0

        for i, batches in enumerate(reader):
            for batch in batches:
                batch_size = len(batch.source_seqs)
                total_samples += batch_size

                # Count audio elements (total audio samples across all sequences)
                batch_audio_elements = sum(batch.source_seq_lens)
                total_audio_elements += batch_audio_elements

                # Count text elements (total tokens across all sequences)
                batch_text_elements = sum(batch.target_seq_lens)
                total_text_elements += batch_text_elements

                print(f"\n📦 Batch {i + 1}:")
                print(f"  🔢 Samples: {batch_size}")
                print(f"  🎵 Audio shape: {batch.source_seqs.shape}")
                print(f"  📝 Text shape: {batch.target_seqs.shape}")
                print(f"  🎵 Audio elements: {batch_audio_elements:,}")
                print(f"  📝 Text elements: {batch_text_elements:,}")

                first_target = batch.target_seqs[0][: batch.target_seq_lens[0]]
                decoded_text = tokenizer.create_decoder()(first_target)
                print(f"  💬 Sample text: '{decoded_text}'")

                assert (
                    hasattr(batch, "example") and batch.example
                ), "Batch must have 'example' attribute with metadata"

                assert isinstance(batch.example, dict), "Batch example must be a dict"
                assert (
                    "lang" in batch.example
                ), "Batch example must contain 'lang' field"
                assert (
                    "corpus" in batch.example
                ), "Batch example must contain 'corpus' field"

                languages = batch.example["lang"]
                corpora = batch.example["corpus"]

                # Assert that we have the same number of languages and corpora
                assert len(languages) == len(
                    corpora
                ), f"Mismatch between number of languages ({len(languages)}) and corpora ({len(corpora)})"

                # Count (corpus, language) tuples
                for corpus, language in zip(corpora, languages):
                    corpus_language_counter[(corpus, language)] += 1

        # Report final statistics
        print(f"\n✅ Completed {i} iterations successfully!")
        print(f"📊 Total samples processed: {total_samples}")
        print(f"🎵 Total audio elements: {total_audio_elements:,}")
        print(f"📝 Total text elements: {total_text_elements:,}")
        print(f"🔢 Total elements: {total_audio_elements + total_text_elements:,}")
        print("\n📈 Final (corpus, language) distribution:")
        for (corpus, language), count in corpus_language_counter.most_common():
            print(f"  📚 {corpus} / 🌍 {language}: {count} samples")
        return reader


def main():
    """
    ```
    python -m omnilingual_asr.dataprep.dataloader_example \
    --dataset_path="root_ds/all_asr/version=0" \
    --split="train" \
    --num_iterations=10
    ```
    """
    fire.Fire(DataLoaderExample().test_dataset)


if __name__ == "__main__":
    main()
