# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

from fairseq2.data.data_pipeline import DataPipelineBuilder
from fairseq2.gang import Gangs


@dataclass
class TaskConfig:
    """Base task configuration."""

    pass


TaskConfigType = TypeVar("TaskConfigType", bound=TaskConfig)


class TaskInterface(ABC, Generic[TaskConfigType]):
    """Base interface for task-specific processing."""

    def __init__(self, config: TaskConfigType):
        self._config = config

    @abstractmethod
    def apply_processing_pipeline(
        self, builder: DataPipelineBuilder, gangs: Gangs, **kwargs
    ) -> DataPipelineBuilder:
        """Apply task-specific processing to raw examples."""

    @abstractmethod
    def get_batch_type(self) -> type:
        """Return the batch type this task produces."""

    @property
    def config(self) -> TaskConfigType:
        return self._config
