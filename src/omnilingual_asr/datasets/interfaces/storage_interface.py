# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Set, TypeVar

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.datasets import SyncMode
from fairseq2.gang import Gangs


@dataclass
class StorageConfig:
    """Base storage configuration"""

    sync_batches: bool = True
    """If ``True``, ensures that each process reads the same number of batches."""

    sync_mode: SyncMode = SyncMode.UNTIL_FIRST
    """The data synchronization mode among processes."""


StorageConfigType = TypeVar("StorageConfigType", bound=StorageConfig)


class StorageInterface(ABC, Generic[StorageConfigType]):
    """Base interface for data storage backends"""

    def __init__(self, config: StorageConfigType):
        self._config = config

    @abstractmethod
    def create_raw_data_pipeline(self, split: str, gangs: Gangs) -> DataPipelineBuilder:
        """Create pipeline yielding raw examples: {'audio': bytes, 'text': str, 'audio_size': int}"""

    @property
    @abstractmethod
    def splits(self) -> Set[str]:
        """Return available dataset splits"""

    @property
    def config(self) -> StorageConfigType:
        return self._config
