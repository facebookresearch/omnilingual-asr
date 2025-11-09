# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from omnilingual_asr.datasets.utils.audio import (
    apply_audio_normalization,
    convert_to_mono,
)


def test_audio_normalization():
    torch.manual_seed(14)
    waveform = torch.randn(16000) * 11 + 7

    normalized = apply_audio_normalization(waveform)

    assert normalized.shape == waveform.shape
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(normalized.std(), torch.tensor(1.0), atol=3e-5)


def test_convert_to_mono():
    stereo = torch.tensor([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]])
    mono = convert_to_mono(stereo)

    assert mono.dim() == 1
    expected = torch.tensor([2.0, 3.0, 4.0])
    assert torch.allclose(mono, expected)

    mono_input = torch.tensor([1.0, 2.0, 3.0])
    mono_output = convert_to_mono(mono_input)
    assert torch.equal(mono_input, mono_output)
