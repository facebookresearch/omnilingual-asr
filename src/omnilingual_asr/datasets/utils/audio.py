# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from functools import partial, reduce
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchaudio
from fairseq2.data.audio import AudioDecoder, WaveformToFbankConverter
from fairseq2.data.data_pipeline import DataPipelineBuilder
from torch import Tensor
from torch.nn.functional import layer_norm


@torch.no_grad()
def apply_audio_normalization(waveform: Tensor) -> Tensor:
    """Normalize audio to zero mean and unit variance."""
    return layer_norm(waveform, waveform.shape)


@torch.no_grad()
def convert_to_mono(waveform: Tensor) -> Tensor:
    """Convert multi-channel audio to mono by averaging channels.
    Warning: inplace mode to avoid reallocate the memory, use `convert_to_mono(waveform.clone())` to be safe!
    """
    if waveform.dim() == 2:
        # reduce channels inplace to save the memory
        size = waveform.size(1)
        result = reduce(
            torch.Tensor.add_, [waveform[:, i] for i in range(1, size)], waveform[:, 0]
        )
        waveform = result
        waveform /= size

    return waveform


@torch.no_grad()
def apply_freq_mask(spec: Tensor, freq_mask_param: int = 80) -> Tensor:
    """Apply frequency masking to the spectrogram."""
    n_freq = spec.size(-2)

    assert freq_mask_param < n_freq
    fmask_len = random.randint(20, freq_mask_param)
    fmask_i = random.randint(0, (n_freq - fmask_len - 1))

    masked_spec = spec.clone()
    masked_spec[:, fmask_i : (fmask_i + fmask_len)] = 0.0
    return masked_spec


@torch.no_grad()
def apply_time_mask(spec: Tensor, time_mask_param: int = 80) -> Tensor:
    """Apply time masking to the spectrogram."""
    n_t = spec.size(-1)

    time_mask_param = min(120, int(n_t / 4))
    assert time_mask_param < n_t
    tmask_len = random.randint(0, time_mask_param)
    tmask_i = random.randint(0, (n_t - tmask_len - 1))

    masked_spec = spec.clone()
    masked_spec[..., tmask_i : (tmask_i + tmask_len)] = 0.0
    return masked_spec


@torch.no_grad()
def apply_spec_augment(
    waveform: Tensor,
    n_fft: int = 400,
    win_len: Optional[int] = None,
    hop_len: Optional[int] = None,
    power: int | None = None,
    freq_mask_param: int = 80,
    time_mask_param: int = 80,
) -> Tensor:
    """Apply SpecAugment with frequency and time masking."""
    spectrogram = torchaudio.transforms.Spectrogram(  # type: ignore
        n_fft=n_fft,
        win_length=win_len,
        hop_length=hop_len,
        center=True,
        pad_mode="reflect",
        power=power,
    )(waveform)

    # augment
    spectrogram_aug = apply_freq_mask(spectrogram, freq_mask_param)
    spectrogram_aug = apply_time_mask(spectrogram_aug, time_mask_param)

    # convert back to waveform
    inverse_spec = torchaudio.transforms.InverseSpectrogram()  # type: ignore
    waveform_aug: Tensor = inverse_spec(spectrogram_aug)
    return waveform_aug


@torch.no_grad()
def postprocess_waveform(
    waveform: Tensor,
    normalize_audio: bool,
    dtype: torch.dtype,
    spec_aug_p: Optional[float],
    spec_aug_freq_mask_param: int,
    spec_aug_time_mask_param: int,
) -> Tensor:
    """Post-process audio waveform with normalization and optional SpecAugment."""
    # Handle multi-channel audio
    waveform = convert_to_mono(waveform)

    # Apply normalization
    if normalize_audio:
        waveform = apply_audio_normalization(waveform)

    # Apply SpecAugment
    if spec_aug_p is not None and random.random() < spec_aug_p:
        waveform = apply_spec_augment(
            waveform,
            freq_mask_param=spec_aug_freq_mask_param,
            time_mask_param=spec_aug_time_mask_param,
        )

    return waveform.to(dtype)


def add_audio_decoding(
    builder: DataPipelineBuilder,
    dtype: torch.dtype,
    normalize_audio: bool,
    selector: str,
    npc: int,
) -> DataPipelineBuilder:
    audio_decoder = AudioDecoder(dtype=torch.float32 if normalize_audio else dtype)
    return builder.map(audio_decoder, selector=selector, num_parallel_calls=npc)


def add_fbank_processing(
    builder: DataPipelineBuilder, dtype: torch.dtype, selector: str, npc: int
) -> DataPipelineBuilder:
    """Add filterbank feature extraction."""
    fbank_converter = WaveformToFbankConverter(
        num_mel_bins=80,
        waveform_scale=2**15,
        channel_last=True,
        standardize=True,
        dtype=dtype,
    )

    return builder.map(
        fbank_converter,
        selector=selector,
        num_parallel_calls=npc,
    )


def add_waveform_processing(
    builder: DataPipelineBuilder,
    normalize_audio: bool,
    dtype: torch.dtype,
    selector: str,
    spec_aug_p: float | None,
    spec_aug_freq_mask_param,
    spec_aug_time_mask_param,
) -> DataPipelineBuilder:
    """Add waveform processing (normalization + SpecAugment)."""
    return builder.map(
        partial(
            postprocess_waveform,
            normalize_audio=normalize_audio,
            dtype=dtype,
            spec_aug_p=spec_aug_p,
            spec_aug_freq_mask_param=spec_aug_freq_mask_param,
            spec_aug_time_mask_param=spec_aug_time_mask_param,
        ),
        selector=selector,
    )


class AudioCropper:
    """Crops audio sequences to maximum length."""

    def __init__(
        self,
        audio_selector: str,
        max_audio_len: int,
        seed: int,
        crop_to_batch_minimal_size: bool = False,
    ) -> None:
        self.audio_selector = audio_selector
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.max_audio_len: int = max_audio_len
        self.crop_to_batch_minimal_size: bool = crop_to_batch_minimal_size

    def crop_audios_in_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Crop audio sequences in a batch."""
        if self.crop_to_batch_minimal_size:
            min_audio_len_batch = min(
                (item[self.audio_selector].size(0) for item in batch)
            )
            crop_size = min(self.max_audio_len, min_audio_len_batch)
        else:
            crop_size = self.max_audio_len

        for item in batch:
            audio = item[self.audio_selector]
            if audio.size(0) > crop_size:
                start = self.rng.randint(0, audio.size(0) - crop_size)
                item[self.audio_selector] = audio[start : start + crop_size]

        return batch


def add_audio_cropping(
    builder: DataPipelineBuilder,
    audio_selector: str,
    max_audio_len: int,
    crop_to_batch_minimal_size: bool,
    seed: int,
) -> DataPipelineBuilder:
    """Crop long audios to `max_audio_len`."""
    audio_cropper = AudioCropper(
        audio_selector=audio_selector,
        max_audio_len=max_audio_len,
        crop_to_batch_minimal_size=crop_to_batch_minimal_size,
        seed=seed,
    )
    return builder.map(audio_cropper.crop_audios_in_batch)


def filter_by_audio_length(
    builder: DataPipelineBuilder,
    min_audio_len: int,
    max_audio_len: int,
    length_selector: str,
) -> DataPipelineBuilder:
    """Filter out too short (< `min_audio_len`) or too long audios (> `max_audio_len`)."""
    return builder.filter(
        lambda x: (x[length_selector] >= min_audio_len)
        and (x[length_selector] <= max_audio_len)
    )
