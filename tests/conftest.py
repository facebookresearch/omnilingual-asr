# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Only loading with CUDA as this takes 20+ minutes on CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    pytest.skip("CUDA not available")
