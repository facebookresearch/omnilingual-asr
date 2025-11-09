# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from enum import Enum

import torch
from fairseq2.recipe.config import (
    ADAMW_OPTIMIZER,
    TRI_STAGE_LR,
    AdamWConfig,
    CommonSection,
    CompileOptions,
    DatasetSection,
    GangSection,
    GradAccumulationConfig,
    LRSchedulerSection,
    MixedPrecisionConfig,
    ModelSection,
    OptimizerSection,
    ReferenceModelSection,
    RegimeSection,
    TokenizerSection,
    TrainerSection,
    TriStageLRConfig,
)

from omnilingual_asr.datasets.storage.manifest_storage import ManifestStorageConfig
from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig


class StorageMode(Enum):
    """Storage backends for Wav2Vec2AsrRecipe"""

    MANIFEST = "MANIFEST"
    MIXTURE_PARQUET = "MIXTURE_PARQUET"


class TaskMode(Enum):
    """Task backends for Wav2Vec2AsrRecipe"""

    ASR = "ASR"


@dataclass(kw_only=True)
class Wav2Vec2AsrDatasetSection(DatasetSection):
    """Recipe-specific dataset section that supports mixing storage + task interfaces."""

    train_split: str | None = "train"
    """The name of the training data split. Only `None` during evaluation.
    """

    valid_split: str | None = "dev"
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
class Wav2Vec2AsrTrainerSection(TrainerSection):
    """
    ASR-specific trainer configuration with encoder freezing.
    Note the inheritance from TrainerSection which can be used for recipe customization.
    """

    freeze_encoder_for_n_steps: int = 10_000
    """The encoder will be frozen for this number of steps."""


@dataclass(kw_only=True)
class Wav2Vec2AsrRecipeConfig:
    """wav2vec2 ASR training recipe configuration."""

    # Model configuration
    model: ModelSection = field(
        default_factory=lambda: ModelSection(
            family="wav2vec2_asr",
            compile=False,
            compile_options=CompileOptions(fullgraph=False, dynamic=False),
        )
    )

    # Optional pretrained encoder
    # 1. Sharing encoder weights if `pretrained_encoder.name != ""`  and `model.name == ("" or None)`
    # 2. Training from model checkpoint if `model.name != ("" or None)`
    pretrained_encoder: ReferenceModelSection = field(
        default_factory=lambda: ReferenceModelSection(
            name="",
            family="wav2vec2_ssl",
            compile=False,
            compile_options=CompileOptions(fullgraph=False, dynamic=False),
        )
    )

    dataset: Wav2Vec2AsrDatasetSection = field(
        default_factory=lambda: Wav2Vec2AsrDatasetSection()
    )

    tokenizer: TokenizerSection = field(
        default_factory=lambda: TokenizerSection(name="omniASR_tokenizer")
    )

    gang: GangSection = field(default_factory=lambda: GangSection())

    trainer: Wav2Vec2AsrTrainerSection = field(
        default_factory=lambda: Wav2Vec2AsrTrainerSection(
            mixed_precision=MixedPrecisionConfig(dtype=torch.bfloat16),
            grad_accumulation=GradAccumulationConfig(num_batches=4),
        )
    )

    optimizer: OptimizerSection = field(
        default_factory=lambda: OptimizerSection(
            name=ADAMW_OPTIMIZER,
            config=AdamWConfig(
                lr=5e-05,
                betas=(0.9, 0.98),
                eps=1e-08,
                weight_decay=0.00,
            ),
        )
    )

    lr_scheduler: LRSchedulerSection = field(
        default_factory=lambda: LRSchedulerSection(
            name=TRI_STAGE_LR,
            config=TriStageLRConfig(
                stage_ratio=(0.1, 0.4, 0.5),
                start_lr_scale=0.01,
                final_lr_scale=0.05,
            ),
        )
    )

    regime: RegimeSection = field(
        default_factory=lambda: RegimeSection(
            num_steps=20_000,
            score_metric="wer",  # defined in wer_calculator.py::WerCalculator::_wer_key
            validate_after_n_steps=10_000,
            validate_every_n_steps=1_000,
            publish_metrics_every_n_steps=100,
            checkpoint_every_n_steps=5_000,
        )
    )

    common: CommonSection = field(default_factory=lambda: CommonSection())
