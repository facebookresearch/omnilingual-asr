# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from fairseq2.recipe.config import (
    CommonSection,
    DatasetSection,
    EvaluatorSection,
    GangSection,
    ReferenceModelSection,
    TokenizerSection,
)

from omnilingual_asr.datasets.storage.manifest_storage import ManifestStorageConfig
from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig

from ..default_config import StorageMode, TaskMode


@dataclass(kw_only=True)
class Wav2Vec2AsrEvalDatasetSection(DatasetSection):
    """Recipe-specific dataset section that supports mixing storage + task interfaces."""

    valid_split: str = "test"
    """The name of the validation data split(s). Format multiple splits interspersed by `,`and without spaces (`'valid,dev_clean,test_clean'`).
    """

    storage_mode: StorageMode = StorageMode.MANIFEST
    """Storage format for the dataset (e.g., MANIFEST, PARQUET)."""

    task_mode: TaskMode = TaskMode.ASR
    """Task type for training (e.g., ASR, SSL)."""

    manifest_storage_config: ManifestStorageConfig = field(
        default_factory=ManifestStorageConfig
    )
    """Configuration for manifest-based dataset storage. Used when storage_mode is MANIFEST."""

    mixture_parquet_storage_config: MixtureParquetStorageConfig = field(
        default_factory=MixtureParquetStorageConfig
    )
    """Configuration for parquet-based dataset storage. Used when storage_mode is MIXTURE_PARQUET."""

    asr_task_config: AsrTaskConfig = field(default_factory=AsrTaskConfig)
    """Configuration for ASR task parameters. Used when task_mode is ASR."""


@dataclass(kw_only=True)
class Wav2Vec2AsrEvalRecipeConfig:
    """wav2vec2 ASR evaluation configuration."""

    # ReferenceModelSection instead of ModelSection because we are
    # loading a checkpoint instead of training the model.
    model: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="omniASR_CTC_300M",
        )
    )

    dataset: Wav2Vec2AsrEvalDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrEvalDatasetSection()
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="omniASR_tokenizer")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    evaluator: EvaluatorSection = field(
        default_factory=lambda: EvaluatorSection(amp=True, amp_dtype=torch.bfloat16)
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
