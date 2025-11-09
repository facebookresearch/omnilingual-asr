# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, final

import torch
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.datasets import DataPipelineReader, DataReader, Seq2SeqBatch
from fairseq2.gang import Gangs

from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    MixtureParquetStorage,
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTask, AsrTaskConfig

MIXTURE_PARQUET_ASR_DATASET: Final = "mixture_parquet_asr_dataset"


@dataclass
class MixtureParquetAsrDatasetConfig:
    """Asset-level card configuration for `MixtureParquetAsrDataset`."""

    data: Path = field(default_factory=Path)
    """Path to the dataset directory containing parquet corpora."""


@final
class MixtureParquetAsrDataset:
    """Combines both the `MixtureParquetStorage` and `AsrTask` pipelines in a fairseq2 dataset implementation."""

    def __init__(
        self,
        path: Path,
    ):
        self.path = path

    @classmethod
    def from_path(
        cls,
        path: Path,
    ) -> "MixtureParquetAsrDataset":
        """Splits are defined in the parquet dataset, but we need `config.fragment_streaming.filesystem` and `config.split_name` to instantiate it. Resolved during `create_reader` call instead."""
        return cls(path)

    def create_reader(
        self,
        split: str,
        tokenizer: Tokenizer,
        gangs: Gangs,
        dtype: torch.dtype,
        num_accumulate: int,
        storage_config: MixtureParquetStorageConfig,
        task_config: AsrTaskConfig,
    ) -> DataReader[Seq2SeqBatch]:
        storage = MixtureParquetStorage(
            path=self.path,
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
            strict_state=False,  # required for parquet-based data-reader checkpointing
        )


def open_mixture_parquet_asr_dataset(
    config: MixtureParquetAsrDatasetConfig,
) -> MixtureParquetAsrDataset:
    return MixtureParquetAsrDataset.from_path(path=config.data)
