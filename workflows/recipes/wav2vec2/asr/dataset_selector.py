# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import NoReturn, Tuple, Union

from fairseq2.recipe.base import RecipeContext

from omnilingual_asr.datasets.impl.manifest_asr_dataset import ManifestAsrDataset
from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
    MixtureParquetAsrDataset,
)
from omnilingual_asr.datasets.storage.manifest_storage import ManifestStorageConfig
from omnilingual_asr.datasets.storage.mixture_parquet_storage import (
    MixtureParquetStorageConfig,
)
from omnilingual_asr.datasets.tasks.asr_task import AsrTaskConfig

from .default_config import StorageMode, TaskMode, Wav2Vec2AsrRecipeConfig
from .eval.default_config import Wav2Vec2AsrEvalRecipeConfig

# Both training and eval recipes share the same config structure w.r.t. the dataset
ConfigType = Union[Wav2Vec2AsrRecipeConfig, Wav2Vec2AsrEvalRecipeConfig]


class Wav2Vec2AsrDatasetSelector:
    """Type-safe dataset selection based on storage/task modes."""

    @classmethod
    def get_dataset_and_configs(
        cls, config: ConfigType, context: RecipeContext
    ) -> Union[
        Tuple[ManifestAsrDataset, ManifestStorageConfig, AsrTaskConfig],
        Tuple[MixtureParquetAsrDataset, MixtureParquetStorageConfig, AsrTaskConfig],
    ]:
        combination = (config.dataset.storage_mode, config.dataset.task_mode)

        if combination == (StorageMode.MANIFEST, TaskMode.ASR):
            return cls._get_manifest_asr(config, context)
        elif combination == (StorageMode.MIXTURE_PARQUET, TaskMode.ASR):
            return cls._get_mixture_parquet_asr(config, context)
        else:
            cls._raise_unsupported_combination(combination)

    @classmethod
    def _get_manifest_asr(
        cls, config: ConfigType, context: RecipeContext
    ) -> Tuple[ManifestAsrDataset, ManifestStorageConfig, AsrTaskConfig]:
        dataset = context.default_dataset.as_(ManifestAsrDataset)
        return (
            dataset,
            config.dataset.manifest_storage_config,
            config.dataset.asr_task_config,
        )

    @classmethod
    def _get_mixture_parquet_asr(
        cls, config: ConfigType, context: RecipeContext
    ) -> Tuple[MixtureParquetAsrDataset, MixtureParquetStorageConfig, AsrTaskConfig]:
        dataset = context.default_dataset.as_(MixtureParquetAsrDataset)
        return (
            dataset,
            config.dataset.mixture_parquet_storage_config,
            config.dataset.asr_task_config,
        )

    @classmethod
    def _raise_unsupported_combination(cls, combination) -> NoReturn:
        supported = [
            (StorageMode.MANIFEST, TaskMode.ASR),
            (StorageMode.MIXTURE_PARQUET, TaskMode.ASR),
        ]
        raise ValueError(
            f"Unsupported combination {combination}. Supported: {supported}"
        )
