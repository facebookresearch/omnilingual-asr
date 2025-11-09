# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Audio processing tools for Omnilingual ASR data preparation.

This module provides utilities for processing audio data in PyArrow tables,
including audio decoding, resampling, format conversion, and batch processing.
"""

import io
from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import torch
import torchaudio
from fairseq2.data._memory import MemoryBlock
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import FileMapper, read_sequence
from numpy.typing import NDArray


def map_to_target_schema(batch: pa.Table, split: str, corpus: str) -> pa.Table:
    """
    Maps a batch of data to the target schema by flattening, renaming columns,
    adding audio bytes, split, and corpus columns, and selecting the final set of columns.
    """
    batch = batch.rename_columns({"transcription": "text"})
    batch = batch.append_column(
        "split", pa.array([split] * len(batch), type=pa.string())
    )
    batch = batch.append_column(
        "corpus", pa.array([corpus] * len(batch), type=pa.string())
    )
    return batch.select(
        ["text", "audio_bytes", "language", "split", "corpus", "audio_size"]
    )


class AudioTableProcessor:
    """
    A processor for handling audio data in PyArrow tables.

    This class can read audio from file paths or byte arrays, resample to a target
    sample rate, and convert to compressed audio bytes for storage.

    Args:
        sample_rate (int): Target sample rate for audio processing. Defaults to 16,000 Hz.
        nb_threads (int): Number of parallel threads for audio processing. Defaults to 10.
        audio_column (str): Name of the column containing audio data. Defaults to "audio_bytes".
        audio_format (str): Format for compressing audio bytes. Defaults to "ogg".
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        nb_threads: int = 10,
        audio_column: str = "audio_bytes",
        audio_format: str = "ogg",
    ):
        self.audio_decoder = AudioDecoder(dtype=torch.float32)
        self.file_mapper = FileMapper(cached_fd_count=200)
        self.nb_threads = nb_threads
        self.sample_rate = sample_rate
        self.audio_column = audio_column
        self.audio_format = audio_format

    def _post_process(self, data: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """
        Post-processes decoded audio data by handling multi-channel audio and resampling.

        Args:
            data (Dict[str, Any] | None): Audio data dictionary containing 'sample_rate'
                                         and 'waveform' keys, or None if decoding failed.

        Returns:
            Dict[str, Any] | None: Processed audio data with standardized format, or None.
        """
        if data is None:
            return None

        sr, wav = data["sample_rate"], data["waveform"]

        if len(wav.shape) > 1:
            dim = np.argmin(wav.shape)
            wav = wav.mean(dim=dim, keepdim=True).reshape(1, -1)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            data["sample_rate"] = self.sample_rate

        data["waveform"] = wav
        return data

    def _wav_to_bytes(
        self,
        wav: torch.Tensor | NDArray | None,
    ) -> NDArray[np.int8] | None:
        """
        Converts a waveform (tensor or numpy array) to a byte array in the specified compression audio format.

        Args:
            wav (torch.Tensor | NDArray): Input waveform as a torch tensor or numpy array.
            sample_rate (int, optional): Sample rate of the audio. Defaults to 16,000 Hz.
            format (str, optional): Audio format for encoding (e.g., "flac"). Defaults to "flac".

        Returns:
            NDArray[np.int8]: Numpy array of int8 representing the encoded audio bytes.
        """
        if wav is None:
            return None
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if wav.dtype != torch.float32:
            wav = wav.float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        buffer = io.BytesIO()
        torchaudio.save(
            buffer,
            wav,
            sample_rate=self.sample_rate,
            format=self.audio_format,
            backend="soundfile",  # fs2 compatible
        )
        buffer.seek(0)
        return np.frombuffer(buffer.getvalue(), dtype=np.int8)

    def _bytes_decoded_audio(self, _bytes):
        """
        Decodes audio from a byte array.

        Args:
            _bytes: Raw audio bytes to decode.

        Returns:
            Decoded audio data dictionary or None if decoding fails.
        """
        try:
            return self.audio_decoder(MemoryBlock(_bytes))
        except Exception as e:
            print(f"READING audio error: \n {e}")
            return None

    def _file_audio_decoder(self, data):
        """
        Decodes audio from file data.

        Args:
            data: File data to decode.

        Returns:
            Decoded audio data dictionary or None if decoding fails.
        """
        try:
            return self.audio_decoder(data)
        except Exception as e:
            print(f"READING audio error: \n {e}")
            return None

    def read_audio_files(self, file_seqs: List[str]) -> List[torch.Tensor | None]:
        """
        Reads a list of audio file paths, decodes them, resamples if necessary,
        and returns a list of processed audio data.

        Args:
            file_seqs (List[str]): List of audio file paths to read.

        Returns:
            List[Tensor]: List of dictionaries containing processed audio data with waveforms.
        """

        builder = read_sequence(file_seqs)
        builder = builder.map(self.file_mapper)
        builder = builder.map(
            self._file_audio_decoder,
            selector="data",
            num_parallel_calls=self.nb_threads,
        )
        builder = builder.map(self._post_process, selector="data")
        builder = builder.map(lambda x: x["data"]["waveform"])
        return list(builder.and_return())

    def read_audio_bytes(
        self, bytes_seqs: List[NDArray[np.int8]]
    ) -> List[torch.Tensor | None]:
        """
        Args:
            bytes_seq (List[NDArray[np.int8]]): List of audio binary representations to read.

        Returns:
            List[Tensor]: List of dictionaries containing processed audio data with waveforms.
        """

        builder = read_sequence(bytes_seqs)
        builder = builder.map(self._bytes_decoded_audio)
        builder = builder.map(self._post_process)
        builder = builder.map(lambda x: x["waveform"] if x is not None else None)
        return list(builder.and_return())

    def __call__(self, table: pa.Table) -> pa.Table:
        table = table.flatten()

        col_type = table[self.audio_column].type
        seqs = table[self.audio_column].to_pandas().to_list()

        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            audio_waveforms = self.read_audio_files(seqs)
        elif (
            pa.types.is_binary(col_type)
            or pa.types.is_large_binary(col_type)
            or (
                (pa.types.is_list(col_type) or pa.types.is_large_list(col_type))
                and pa.types.is_int8(col_type.value_type)
            )
        ):
            audio_waveforms = self.read_audio_bytes(seqs)
        else:
            raise ValueError(f"Unsupported column type for audio column: {col_type}.")

        audio_sizes = [max(x.shape) if x is not None else None for x in audio_waveforms]
        audio_bytes = [self._wav_to_bytes(wav) for wav in audio_waveforms]

        for col in ["audio_size", "audio_bytes"]:
            if col in table.column_names:
                table = table.drop([col])

        table = table.append_column(
            "audio_size", pa.array(audio_sizes, type=pa.int64())
        ).append_column("audio_bytes", pa.array(audio_bytes, type=pa.list_(pa.int8())))
        return table


def bytes_to_tensor(
    audio_arr: np.ndarray, target_sample_rate: int = 16_000
) -> np.ndarray:
    """
    Converts a numpy array of audio bytes to a waveform numpy array, resampling if needed.

    Args:
        audio_arr (np.ndarray): Numpy array containing audio bytes.
        target_sample_rate (int, optional): Desired sample rate. Defaults to 16,000 Hz.

    Returns:
        np.ndarray: Flattened numpy array of the waveform at the target sample rate.
    """
    wav, sample_rate = torchaudio.load(io.BytesIO(audio_arr), backend="soundfile")  # type: ignore
    if len(wav.shape) > 1:
        dim = np.argmin(wav.shape)
        wav = wav.mean(dim=dim, keepdim=True).reshape(1, -1)  # type: ignore

    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_sample_rate)

    return wav.cpu().numpy().flatten()


def binary_to_list_int8(binary_array: pa.Array | pa.ChunkedArray) -> pa.Array:
    """
    Efficiently convert a pyarrow BinaryArray to a ListArray of int8.
    Each binary value becomes a list of int8 values (that's copy-less method)
    Nulls are preserved.
    """
    if not pa.types.is_binary(binary_array.type):
        raise ValueError("Input array must be of binary type.")
    if isinstance(binary_array, pa.ChunkedArray):
        binary_array = binary_array.combine_chunks()

    # Get buffers: [null_bitmap, offsets, data]
    buffers = binary_array.buffers()
    offsets = buffers[1]
    data = buffers[2]
    offset = binary_array.offset

    # Offsets as numpy array
    offsets_np = np.frombuffer(offsets, dtype="int32")[  # type: ignore
        offset : offset + len(binary_array) + 1
    ]

    data_np = np.frombuffer(data, dtype="int8")[offsets_np[0] :]  # type: ignore
    offsets_np -= offsets_np[0]
    values_array = pa.array(data_np, type=pa.int8())

    list_array = pa.ListArray.from_arrays(
        offsets_np, values_array, mask=binary_array.is_null()
    )
    return list_array
