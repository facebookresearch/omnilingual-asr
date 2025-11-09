# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import Set, Tuple

from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, FileMapper
from fairseq2.data.text import StrSplitter, read_text
from fairseq2.datasets import DatasetError
from fairseq2.gang import Gangs
from typing_extensions import override

from omnilingual_asr.datasets.interfaces.storage_interface import (
    StorageConfig,
    StorageInterface,
)


@dataclass
class ManifestStorageConfig(StorageConfig):
    """Manifest-specific storage configuration"""

    read_text: bool = False
    """Whether to look for transcription files.
    """

    cached_fd_count: int = 1000
    """Enables an LRU cache on the last ``cached_fd_count`` files read.
    ``FileMapper`` will memory map all the cached files, so this is especially useful for reading several slices of the same file.
    """


class ManifestStorage(StorageInterface[ManifestStorageConfig]):
    """Manifest-based storage backend. Expects to read paths from a manifest and to load audio files.
    Can optionally read splits with `.wrd` transcriptions if configuring `ManifestStorageConfig::read_text = True`.
    """

    def __init__(
        self,
        manifest_dir: Path,
        splits: Set[str],
        config: ManifestStorageConfig,
    ):
        super().__init__(config)
        self._manifest_dir = manifest_dir
        self._splits = splits

    @override
    def create_raw_data_pipeline(self, split: str, gangs: Gangs) -> DataPipelineBuilder:
        """Returns `{'audio': MemoryBlock, 'length': int}` with optional `'text': str` if reading text."""

        if self.config.read_text:
            tsv_pipeline = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()
            wrd_pipeline = ManifestStorage.read_wrd_file(
                manifest_dir=self._manifest_dir, split=split
            ).and_return()

            builder = DataPipeline.zip([tsv_pipeline, wrd_pipeline], flatten=True)
        else:
            builder = ManifestStorage.read_tsv_file(
                manifest_dir=self._manifest_dir, split=split
            )

        # Cast audio size to integer.
        builder.map(int, selector="length")

        audio_dir = ManifestStorage.retrieve_audio_directory(
            manifest_dir=self._manifest_dir, split=split
        )

        # FileMapper -> {'audio': FileMapperOutput{'path': str, 'data': MemoryBlock}, 'text': str, 'length': int}
        file_mapper = FileMapper(
            root_dir=audio_dir, cached_fd_count=self._config.cached_fd_count
        )
        builder.map(file_mapper, selector="audio")

        # Flatten audio key -> { 'audio': MemoryBlock, 'length': int, 'text': str }
        return builder.map(self.flatten_audio_key)

    @staticmethod
    def discover_splits(path: Path) -> Tuple[Set[str], Path]:
        """:returns: (splits: Set[str], manifest_dir: Path)"""

        path = path.expanduser().resolve()
        if not path.is_dir():
            return {path.stem}, path.parent
        else:
            try:
                splits = {f.stem for f in path.glob("*.tsv")}
            except OSError as ex:
                raise DatasetError(
                    f"The splits under the '{path}' directory of the dataset cannot be determined. See the nested exception for details."  # fmt: skip
                ) from ex
            return splits, path

    @staticmethod
    def retrieve_audio_directory(manifest_dir: Path, split: str) -> Path:
        """Retrieve audio directory from manifest file header."""
        manifest_file = manifest_dir.joinpath(f"{split}.tsv")

        try:
            with manifest_file.open(encoding="utf-8") as fp:
                header = fp.readline().rstrip()
        except OSError as ex:
            raise DatasetError(
                split,
                f"The {manifest_file} manifest file cannot be read. See the nested exception for details.",
            ) from ex

        try:
            audio_dir = Path(header)
            if audio_dir.exists():
                return audio_dir
            else:
                raise ValueError
        except ValueError:
            raise DatasetError(
                split,
                f"The first line of the '{manifest_file}' manifest file must point to a data directory.",
            ) from None

    @staticmethod
    def read_tsv_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        """Expecting the following file structure at `{split}.tsv`:
        ```
        /path-to-dataset/062419
        train-clean-100/1553/140047/1553-140047-0000.flac       180080
        train-clean-100/1553/140047/1553-140047-0001.flac       219840
        (...)
        ```
        """
        tsv_file = manifest_dir.joinpath(f"{split}.tsv")

        builder = read_text(tsv_file, rtrim=True, memory_map=True)

        builder.skip(1)  # Path to the data directory.

        field_splitter = StrSplitter(names=["audio", "length"])

        builder.map(field_splitter)

        return builder

    @staticmethod
    def read_wrd_file(manifest_dir: Path, split: str) -> DataPipelineBuilder:
        """Read WRD file containing text transcriptions."""
        wrd_file = manifest_dir.joinpath(f"{split}.wrd")

        return read_text(wrd_file, key="text", rtrim=True, memory_map=True)

    @staticmethod
    def flatten_audio_key(example):
        """Aligns the manifest dict with the parquet output to use a common output for the storage API.

        Transforms { 'audio': { 'data': MemoryBlock, 'path': str }}
                -> { 'audio': MemoryBlock }"""
        return {**example, "audio": example["audio"]["data"]}

    @property
    def splits(self) -> Set[str]:
        """Return available dataset splits"""
        return self._splits
