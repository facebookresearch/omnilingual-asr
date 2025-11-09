# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Set, final

import torch
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataPipelineReader, DataReader, Seq2SeqBatch
from fairseq2.gang import Gangs

from omnilingual_asr.datasets.storage.manifest_storage import (
    ManifestStorage,
    ManifestStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTask, AsrTaskConfig

MANIFEST_ASR_DATASET: Final = "manifest_asr_dataset"


@dataclass
class ManifestAsrDatasetConfig:
    """Asset-level card configuration for `ManifestAsrDataset`."""

    data: Path = field(default_factory=Path)
    """Path to the dataset directory containing .tsv and .wrd files."""


@final
class ManifestAsrDataset:
    """Combines both the `ManifestStorage` and `AsrTask` pipelines in a fairseq2 dataset implementation."""

    def __init__(
        self,
        manifest_dir: Path,
        splits: Set[str],
    ):
        self.manifest_dir = manifest_dir
        self.splits = splits

    @classmethod
    def from_path(
        cls,
        path: Path,
    ) -> "ManifestAsrDataset":
        splits, manifest_dir = ManifestStorage.discover_splits(path)
        return cls(manifest_dir, splits)

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gangs: Gangs,
        dtype: torch.dtype,
        num_accumulate: int,
        storage_config: ManifestStorageConfig,
        task_config: AsrTaskConfig,
    ) -> DataReader[Seq2SeqBatch]:

        storage = ManifestStorage(
            splits=self.splits,
            manifest_dir=self.manifest_dir,
            config=storage_config,
        )
        task = AsrTask(config=task_config)

        # Stitch storage and task together via DataPipeline
        builder = storage.create_raw_data_pipeline(split, gangs)
        builder = task.apply_processing_pipeline(
            builder, gangs, tokenizer=tokenizer, dtype=dtype
        )
        pipeline = builder.and_return()

        return DataPipelineReader[Seq2SeqBatch](
            pipeline,
            gangs,
            num_accumulate=num_accumulate,
            sync=storage_config.sync_batches,
            sync_mode=storage_config.sync_mode,
        )


def open_manifest_asr_dataset(
    config: ManifestAsrDatasetConfig,
) -> ManifestAsrDataset:
    return ManifestAsrDataset.from_path(config.data)
