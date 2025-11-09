# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fairseq2.data._memory import MemoryBlock
from fairseq2.data.data_pipeline import DataPipeline, DataPipelineBuilder, read_sequence
from fairseq2.data.parquet import (
    FragmentLoadingConfig,
    FragmentStreamingConfig,
    NamedColumns,
    ParquetFragmentLoader,
    ParquetFragmentStreamer,
)
from fairseq2.data.parquet.fragment_streaming import ParquetDatasetLimitOptions
from fairseq2.data.parquet.fragment_streaming.primitives import process_filter
from fairseq2.datasets import SyncMode
from fairseq2.gang import Gangs
from fairseq2.logging import log
from pyarrow.dataset import get_partition_keys
from tqdm import tqdm
from typing_extensions import override

from omnilingual_asr.datasets.interfaces.storage_interface import (
    StorageConfig,
    StorageInterface,
)


@dataclass
class LangASRSchema(NamedColumns):
    """Schema for mixed parquet ASR datasets with language/corpus partitioning."""

    audio: str = "audio_bytes"
    length: str = "audio_size"
    text: str = "text"
    split: str = "split"
    lang: str = "language"
    corpus: str = "corpus"


@dataclass(unsafe_hash=True)
class Partition:
    """Represents a language-corpus partition."""

    lang: str
    corpus: str


@dataclass
class MixtureParquetStorageConfig(StorageConfig):
    """Configuration for mixed parquet storage with multilingual weighting."""

    fragment_streaming: FragmentStreamingConfig = field(
        default_factory=lambda: FragmentStreamingConfig(
            parquet_path=str(),
            filesystem=None,
            name=None,
            weight=1,
            partition_filters=None,
            limit=ParquetDatasetLimitOptions(
                fraction_of_files=None, nb_files=None, nb_fragments=None, nb_rows=None
            ),
            split_to_row_groups=True,
            seed=2,  # "local" parquet worker seed
            fragment_shuffle_window=0,  # override from default 40
            files_circular_shift=False,
            nb_epochs=None,
        )
    )
    """Explicit defaults for the external `FragmentStreamingConfig`."""

    fragment_loading: FragmentLoadingConfig = field(
        default_factory=lambda: FragmentLoadingConfig(
            columns=NamedColumns(),  # Can't use LangASRSchema here, using global namespace instead
            rename_columns=True,
            add_fragment_traces=False,  # override from default
            drop_null=False,  # override from default
            min_batch_size=1,
            filters=None,
            non_deterministic_read=False,
            use_threads=False,
            nb_prefetch=1,  # override from default 0
            num_parallel_fragments=1,
            cache=True,  # override from default
            cache_dir=None,
        )
    )
    """Explicit defaults for external `FragmentLoadingConfig`."""

    # Partition weighting
    dataset_summary_path: str | None = None
    """Path to TSV file containing corpus/language hour distribution for weighting."""

    beta_corpus: float | None = None
    """Beta parameter for corpus weighting. weight = (hours/total_hours)^beta."""

    beta_language: float | None = None
    """Beta parameter for language weighting within corpus."""

    # Additional fragment loading options
    pa_cpu_count: int = 20
    """Number of CPU threads for pyarrow operations."""

    # Misc
    max_workers: int = 30
    """Maximum number of workers for parallel partition loading."""

    split_column: str = "split"
    """Name of the column containing split information."""

    num_prefetch: int = 1
    """Number of batches to prefetch."""

    parquet_path_name: str = "_parquet_path"
    """Name of the field to store parquet path in partition dictionary."""

    sampling_seed: int = 261
    """Mixture sampling seed."""


class MixtureParquetStorage(StorageInterface[MixtureParquetStorageConfig]):
    """Mixture parquet storage implementation with partition weighting support.
    Enables to read multiple splits at the same time and sample respective to their weight
    provided in the `MixtureParquetStorageConfig::dataset_summary_path`.
    """

    def __init__(
        self,
        path: Path,
        config: MixtureParquetStorageConfig,
    ) -> None:
        super().__init__(config)

        # Init dataset and discover splits
        dataset, splits = MixtureParquetStorage.load_and_discover_splits(
            path=path,
            filesystem=config.fragment_streaming.filesystem,
            split_column=config.split_column,
        )

        self._dataset: pq.ParquetDataset = dataset
        self._splits: Set[str] = splits

        # Cache all partitions during initialization
        self._full_partition_df: pa.Table = (
            MixtureParquetStorage.get_all_mixture_partitions(
                dataset=self._dataset, parquet_path_name=self.config.parquet_path_name
            )
        )

    @staticmethod
    def load_and_discover_splits(
        path: Path, filesystem: Any | None, split_column: str
    ) -> Tuple[pq.ParquetDataset, Set[str]]:
        """Discovers the available splits in the `pd.ParquetDataset(path, filesystem)` and returns them with the loaded dataset."""
        # ParquetDataset expects a str instead of pathlib.Path
        dataset = pq.ParquetDataset(str(path), filesystem=filesystem)

        partition_columns: List[str] = []
        if dataset.partitioning is not None:
            partition_columns = dataset.partitioning.schema.names

        splits: Set[str] = set()
        if dataset.partitioning is not None and split_column in partition_columns:
            idx = partition_columns.index(split_column)
            _splits = dataset.partitioning.dictionaries[idx]  # type: ignore
            if _splits is None:
                splits = set()
            else:
                splits = set(_splits.to_pylist())

        return dataset, splits

    @staticmethod
    def get_all_mixture_partitions(
        dataset: pq.ParquetDataset, parquet_path_name: str
    ) -> pa.Table:
        """Returns a table mapping each partition to its parquet path."""
        dicts = []
        for fragment in dataset._dataset.get_fragments(  # type: ignore
            filter=dataset._filter_expression  # type: ignore
        ):
            dd = get_partition_keys(fragment.partition_expression)  # type: ignore
            dd = dd or {}
            dd[parquet_path_name] = fragment.path
            dicts.append(dd)
        return pa.Table.from_pylist(dicts)

    @staticmethod
    def is_train_streaming(split: str, sync_mode: SyncMode) -> bool:
        """Determine if this is a training split and we're streaming data based on name and mode."""
        return "train" in split and sync_mode == SyncMode.UNTIL_FIRST

    @override
    def create_raw_data_pipeline(self, split: str, gangs: Gangs) -> DataPipelineBuilder:

        config = self.config
        schema: LangASRSchema = LangASRSchema()

        is_train_streaming = MixtureParquetStorage.is_train_streaming(
            split=split, sync_mode=config.sync_mode
        )

        # Get relevant partition files to load
        full_partition_filters = MixtureParquetStorage.fix_partition_filters(
            split=split,
            schema=schema,
            partition_filters=config.fragment_streaming.partition_filters,  # type: ignore[arg-type]
            is_train_streaming=is_train_streaming,
        )

        # Map partitions to parquet paths
        split_paths = MixtureParquetStorage.get_filtered_paths(
            full_partition_df=self._full_partition_df,
            filter_exp=full_partition_filters,
            schema=schema,
            parquet_path_name=config.parquet_path_name,
        )

        assert (
            len(split_paths) > 0
        ), f"No parquet files found for the current split {split}."

        # Get partition weights if training
        if is_train_streaming:
            partition_weights = MixtureParquetStorage.get_partition_weights_from_betas(
                dataset_summary_path=config.dataset_summary_path,
                beta_corpus=config.beta_corpus,
                beta_language=config.beta_language,
            )
        else:
            partition_weights = None

        valid_mixed_pipeline = (
            partition_weights is not None
            and is_train_streaming
            and len(split_paths) > 1
        )

        # Runtime configuration
        fragment_streaming_config: FragmentStreamingConfig = (
            self.config.fragment_streaming
        )
        fragment_streaming_config.nb_epochs = None if is_train_streaming else 1
        fragment_streaming_config.fragment_shuffle_window = (
            -1 if is_train_streaming else 0
        )
        fragment_streaming_config.seed += (
            gangs.dp.size
        )  # shifts seed for parallel workers (buggy: constant for all dp, but required for BC)

        fragment_loading_config: FragmentLoadingConfig = self.config.fragment_loading
        fragment_loading_config.columns = schema
        fragment_loading_config.cache = (
            fragment_loading_config.cache if is_train_streaming else False
        )

        log.info(
            f"Creating a parquet reader for '{split}'-split with options: {fragment_streaming_config}, {fragment_loading_config}"
        )

        if valid_mixed_pipeline:
            # Multi-partition weighted sampling case
            return MixtureParquetStorage.create_mixed_pipeline(
                fragment_streaming_config=fragment_streaming_config,
                fragment_loading_config=fragment_loading_config,
                split_paths=split_paths,
                partition_weights=partition_weights,  # type: ignore
                max_workers=config.max_workers,
                sampling_seed=config.sampling_seed,
                pa_cpu_count=config.pa_cpu_count,
                gangs=gangs,
            )
        else:
            # Single partition or non-training case - simple pipeline
            files = [x for y in split_paths.values() for x in y]  # Flatten all files
            fragment_streaming_config.parquet_path = files

            return MixtureParquetStorage.reading_one_partition_pipeline(
                fragment_streaming_config=fragment_streaming_config,
                fragment_loading_config=fragment_loading_config,
                pa_cpu_count=config.pa_cpu_count,
                gangs=gangs,
            )

    @property
    def splits(self) -> Set[str]:
        """Return available dataset splits"""
        return self._splits

    @staticmethod
    def get_filtered_paths(
        full_partition_df: pa.Table,
        filter_exp: pa.compute.Expression | None,
        schema: LangASRSchema,
        parquet_path_name: str,
    ) -> Dict[Partition, List[str]]:
        """Returns a dict mapping partition names to lists of parquet paths after filtering."""
        df = full_partition_df
        if filter_exp is not None:
            df = df.filter(filter_exp)  # type: ignore

        pl_df = pl.from_arrow(
            df.select([schema.lang, schema.corpus, parquet_path_name])
        )

        assert isinstance(pl_df, pl.DataFrame)

        indexed_partitions = pl_df.partition_by(
            schema.lang,
            schema.corpus,
            as_dict=True,
            include_key=False,
            maintain_order=True,
        )

        log.info(f"Found {len(indexed_partitions)} partitions after filtering")
        return {
            Partition(lang=str(key[0]), corpus=str(key[1])): val[
                parquet_path_name
            ].to_list()
            for key, val in indexed_partitions.items()
        }

    @staticmethod
    def get_partition_weights_from_betas(
        dataset_summary_path: str | None,
        beta_corpus: float | None,
        beta_language: float | None,
    ) -> Dict[Partition, float] | None:
        """Returns a dict mapping partitions to sample weights."""

        if dataset_summary_path is None:
            return None

        assert beta_corpus is not None and beta_language is not None
        pd.options.mode.copy_on_write = True
        data_summary = pd.read_csv(
            dataset_summary_path, sep="\t"
        )  # FIXME: use polars for consistency with rest of codebase

        def _compute_sample_weights(
            beta: float, hours: "pd.Series[float]"
        ) -> "pd.Series[float]":
            total_hours = hours.sum()
            weights = pow((hours / total_hours), beta)
            norm_weights = weights / weights.sum()
            return norm_weights

        # Get Corpus weights
        corpus_summary = data_summary.groupby("corpus")["hours"].sum().reset_index()
        corpus_summary["corpus_weights"] = _compute_sample_weights(
            beta_corpus, corpus_summary.hours
        )
        data_summary = data_summary.merge(
            corpus_summary[["corpus", "corpus_weights"]], on="corpus"
        )

        # Get language weights within each corpus
        lang_weights = []
        for corpus in data_summary.corpus.unique():
            df_corpus = data_summary.loc[data_summary.corpus == corpus]
            df_corpus["language_weights"] = _compute_sample_weights(
                beta_language, df_corpus.hours
            )
            lang_weights.append(df_corpus)
        lang_weights = pd.concat(lang_weights)
        data_summary = data_summary.merge(
            lang_weights[["corpus", "language_weights", "language"]],  # type: ignore
            on=["corpus", "language"],
        )

        # Sample weight = corpus_weight * language_weight
        data_summary["sample_weights"] = (
            data_summary["corpus_weights"] * data_summary["language_weights"]
        )
        dicts = data_summary.to_dict(orient="records")
        weights = {
            Partition(lang=dd["language"], corpus=dd["corpus"]): float(
                dd["sample_weights"]
            )
            for dd in dicts
        }

        return weights

    @staticmethod
    def fix_partition_filters(
        split: str,
        schema: LangASRSchema,
        partition_filters: Optional[str | List[str]],
        is_train_streaming: bool,
    ) -> pa.compute.Expression:
        """Fix partition filters for the given split."""

        # Extra logic for specifying subsplits
        # Expects format: <split>_<corpus> (where corpus is optional and may contain _)
        corpus = None
        split_info = split.split("_")
        if len(split_info) >= 2:
            split = split_info[0]
            corpus = "_".join(split_info[1:])

        split_filters = pc.field(schema.split) == split
        filters = [split_filters]

        if partition_filters is not None and is_train_streaming:
            # We only apply partition filters to training splits, because we want to
            # be able to validate on held-out corpora
            filters.append(partition_filters)  # type: ignore

        if corpus is not None:
            corpus_filter = pc.field(schema.corpus) == corpus
            filters.append(corpus_filter)  # type: ignore

        full_partition_filters = process_filter(filters)  # type: ignore
        return full_partition_filters  # type: ignore

    @staticmethod
    def dispatch_table_to_examples(
        table: pa.Table, audio_column: str, memory_pool: pa.MemoryPool
    ) -> DataPipeline:
        """Convert Arrow tables to dictionaries with audio bytes as MemoryBlocks."""

        # Convert table to pandas and then to dicts
        records = table.to_pandas(memory_pool=memory_pool, self_destruct=True).to_dict(
            orient="records"
        )

        # Convert audio bytes to MemoryBlocks
        for record in records:
            if audio_column in record and record[audio_column] is not None:
                record[audio_column] = MemoryBlock(record[audio_column])

        return read_sequence(records).and_return()

    @staticmethod
    def reading_one_partition_pipeline(
        fragment_streaming_config: FragmentStreamingConfig,
        fragment_loading_config: FragmentLoadingConfig,
        gangs: Gangs,
        pa_cpu_count: int,
    ) -> DataPipelineBuilder:
        """Create pipeline for reading a single partition."""
        log.info(f"Reading one partition with {fragment_streaming_config.seed=}")
        pa.set_cpu_count(pa_cpu_count)
        pa.set_io_thread_count(pa_cpu_count)

        # Init parquet dataset reader
        fragment_builder = ParquetFragmentStreamer(
            config=fragment_streaming_config
        ).build_pipeline(rank=gangs.dp.rank, world_size=gangs.dp.size)

        # Load data in memory
        builder = ParquetFragmentLoader(config=fragment_loading_config).apply(
            fragment_builder
        )

        # Prepare memory pool for caching
        memory_pool = pa.default_memory_pool()
        try:
            memory_pool = pa.jemalloc_memory_pool()
            pa.jemalloc_set_decay_ms(0)
        except pa.ArrowNotImplementedError:
            pass

        # Convert tables to MemoryBlock interface
        return builder.yield_from(
            lambda table: MixtureParquetStorage.dispatch_table_to_examples(
                table=table, audio_column="audio", memory_pool=memory_pool
            )
        )

    @staticmethod
    def create_mixed_pipeline(
        fragment_streaming_config: FragmentStreamingConfig,
        fragment_loading_config: FragmentLoadingConfig,
        split_paths: Dict[Partition, List[str]],
        partition_weights: Dict[Partition, float],
        gangs: Gangs,
        pa_cpu_count: int,
        max_workers: int,
        sampling_seed: int,
    ) -> DataPipelineBuilder:
        """Create mixed pipeline with weighted partition sampling."""

        # Filter to only partitions that have weights
        weights, dir_files = zip(
            *[
                (partition_weights[d], files_)
                for d, files_ in split_paths.items()
                if d in partition_weights
            ]
        )

        def _training_pipeline(
            local_fragment_streaming_config: FragmentStreamingConfig, seed_offset: int
        ) -> DataPipeline:
            local_fragment_streaming_config.seed += seed_offset
            builder = MixtureParquetStorage.reading_one_partition_pipeline(
                fragment_streaming_config=local_fragment_streaming_config,
                fragment_loading_config=fragment_loading_config,
                pa_cpu_count=pa_cpu_count,
                gangs=gangs,
            )
            pipeline = builder.and_return()
            _ = next(iter(pipeline))  # return one element to warm up the pipeline
            return pipeline

        max_workers = min(max_workers, len(dir_files))
        reading_pipelines: List[DataPipeline] = []
        log.info(f"Creating {max_workers} threads to read {len(dir_files)} partitions.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _training_pipeline,
                    replace(fragment_streaming_config, parquet_path=files),
                    seed_offset,
                )
                for seed_offset, files in enumerate(dir_files, start=1)
            ]
            for future in tqdm(futures, desc="Loading partitions"):
                pipeline = future.result()
                assert isinstance(pipeline, DataPipeline)
                reading_pipelines.append(pipeline)

        builder = DataPipeline.sample(
            reading_pipelines,
            weights=weights,
            seed=sampling_seed,
        )
        builder.prefetch(len(reading_pipelines) * 10)
        return builder
