# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from omnilingual_asr.datasets.utils.audio import AudioCropper


def test_audio_cropper():
    max_len = 1000
    cropper = AudioCropper(
        audio_selector="audio",
        max_audio_len=max_len,
        seed=42,
        crop_to_batch_minimal_size=False,
    )

    long_audio = torch.randn(2000)
    short_audio = torch.randn(500)

    batch = [
        {"audio": long_audio, "id": 1},
        {"audio": short_audio, "id": 2},
    ]

    cropped_batch = cropper.crop_audios_in_batch(batch)

    assert cropped_batch[0]["audio"].shape[0] == max_len
    assert cropped_batch[1]["audio"].shape[0] == 500
    assert cropped_batch[0]["id"] == 1
    assert cropped_batch[1]["id"] == 2

    cropper_same_seed1 = AudioCropper(
        audio_selector="audio",
        max_audio_len=max_len,
        seed=42,
        crop_to_batch_minimal_size=False,
    )
    cropper_same_seed2 = AudioCropper(
        audio_selector="audio",
        max_audio_len=max_len,
        seed=42,
        crop_to_batch_minimal_size=False,
    )

    batch_2 = [{"audio": long_audio.clone(), "id": 1}]
    batch_3 = [{"audio": long_audio.clone(), "id": 1}]

    result_1 = cropper_same_seed1.crop_audios_in_batch(batch_2)
    result_2 = cropper_same_seed2.crop_audios_in_batch(batch_3)

    assert torch.allclose(result_1[0]["audio"], result_2[0]["audio"])


def test_audio_cropper_batch_minimal_size():
    max_len = 2000
    cropper = AudioCropper(
        audio_selector="audio",
        max_audio_len=max_len,
        seed=42,
        crop_to_batch_minimal_size=True,
    )

    batch = [
        {"audio": torch.randn(1500)},
        {"audio": torch.randn(2500)},
        {"audio": torch.randn(1200)},
    ]

    cropped_batch = cropper.crop_audios_in_batch(batch)

    min_len = 1200
    assert cropped_batch[0]["audio"].shape[0] == min_len
    assert cropped_batch[1]["audio"].shape[0] == min_len
    assert cropped_batch[2]["audio"].shape[0] == min_len
