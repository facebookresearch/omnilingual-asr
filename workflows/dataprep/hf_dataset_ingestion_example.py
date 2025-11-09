# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from pathlib import Path

import datasets
import fire
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pa_ds
import ray
from audio_tools import AudioTableProcessor, map_to_target_schema
from datasets import load_dataset
from text_tools import text_normalize


class MLSTextProcessor:
    """
    Batch-level processor for MLS text data processing.

    Handles digit replacement, text normalization, and language mapping
    for batches of MLS data.
    """

    def __init__(self, lang: str):
        """
        Initialize the processor for a specific language.

        Args:
            lang: Language name (e.g., "german", "french", "spanish", etc.)
        """
        self.lang = lang
        self.lang_mapping = {
            "german": "deu_Latn",
            "dutch": "nld_Latn",
            "french": "fra_Latn",
            "spanish": "spa_Latn",
            "italian": "ita_Latn",
            "portuguese": "por_Latn",
            "polish": "pol_Latn",
        }
        self.language_2letter_map = {
            "german": "de",
            "dutch": "nl",
            "french": "fr",
            "spanish": "es",
            "italian": "it",
            "portuguese": "pt",
            "polish": "pl",
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        # Extract transcript column as Python list
        transcripts = batch["transcript"].to_pylist()
        processed_transcriptions = []

        for text in transcripts:
            # Normalize text
            processed_text = text_normalize(
                text, iso_code=self.language_2letter_map[self.lang]
            )
            processed_transcriptions.append(processed_text)

        # Add processed transcription column
        batch = batch.append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )

        if "language" in batch.column_names:
            batch = batch.drop(["language"])

        # Add language column
        language_values = [self.lang_mapping.get(self.lang, self.lang)] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )

        return batch


class FleursTextProcessor:
    """
    Batch-level processor for FLEURS text data processing.

    Handles digit replacement, text normalization, and language mapping
    for batches of FLEURS data.
    """

    def __init__(self, lang: str):
        """
        Initialize the processor for a specific language.

        Args:
            lang: Language code (e.g., "tr_tr", "ja_jp", "en_us", "fr_fr", "uk_ua")
        """
        self.lang = lang
        self.lang_mapping = {
            "tr_tr": "tur_Latn",
            "ja_jp": "jpn_Jpan",
            "en_us": "eng_Latn",
            "fr_fr": "fra_Latn",
            "uk_ua": "ukr_Cyrl",
        }

    def __call__(self, batch: pa.Table) -> pa.Table:
        # Extract transcription column as Python list
        transcriptions = batch["raw_transcription"].to_pylist()
        processed_transcriptions = []

        for text in transcriptions:
            # Normalize text
            processed_text = text_normalize(text, iso_code=self.lang[:2])
            processed_transcriptions.append(processed_text)

        # Drop original transcription column and add the processed one
        batch = batch.drop(["transcription"]).append_column(
            "transcription", pa.array(processed_transcriptions, type=pa.string())
        )

        if "language" in batch.column_names:
            batch = batch.drop(["language"])

        language_values = [self.lang_mapping.get(self.lang, self.lang)] * len(batch)
        batch = batch.append_column(
            "language", pa.array(language_values, type=pa.string())
        )

        return batch


class DataPrepCLI:
    """Command-line interface for ASR data preparation tasks."""

    MLS_LANT_SUBSET: list[str] = [
        "german",
        "dutch",
        "french",
        "spanish",
        "italian",
        "portuguese",
        "polish",
    ]

    FLEURS_LAG_SUBSET = ["tr_tr", "ja_jp", "en_us", "fr_fr", "uk_ua"]

    # Short subset for quick testing (only 2 languages from FLEURS)
    FLEURS_SHORT_SUBSET = ["en_us", "fr_fr"]

    @staticmethod
    def check_versions():
        """Check and display versions of critical packages used in data preparation.

        This helps ensure compatibility and reproducibility of the data preparation pipeline.
        """
        print("📦 Package Versions:")
        print(f"  datasets: {datasets.__version__}")
        print(f"  pyarrow:  {pa.__version__}")
        print(f"  ray:      {ray.__version__}")
        print(f"  polars:   {pl.__version__}")

        # Check for known compatibility issues
        if hasattr(datasets, "__version__"):
            datasets_ver = tuple(map(int, datasets.__version__.split(".")))
            if datasets_ver >= (3, 6, 0):
                print(
                    "⚠️  Warning: datasets version >= 3.6.0 may have compatibility issues"
                )

        if hasattr(ray, "__version__"):
            ray_ver = tuple(
                map(int, ray.__version__.split(".")[:2])
            )  # Major.minor only
            if ray_ver < (2, 49):
                print("⚠️  Warning: ray version < 2.49 may have performance issues")

    def _ingest_mls_internal(self, output_dir: str) -> None:
        """Internal method for MLS ingestion."""
        for lang in self.MLS_LANT_SUBSET:
            for split in ["test", "dev", "train"]:
                mls_hf = load_dataset(
                    "facebook/multilingual_librispeech",
                    lang,
                    split=split,
                    streaming=True,
                )
                mls_hf = mls_hf.shuffle(seed=123, buffer_size=10000)
                ray_ds_stream_ = ray.data.from_huggingface(mls_hf)

                # Use batch-level text processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    MLSTextProcessor,
                    fn_constructor_kwargs={"lang": lang},
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=10,
                )

                # Audio processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    AudioTableProcessor,
                    fn_constructor_kwargs={
                        "audio_column": "audio.bytes",
                        "audio_format": "flac",  # or "ogg", "wav", etc.
                    },
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=10,
                )
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    partial(map_to_target_schema, split=split, corpus="mls"),
                    batch_size=1000,
                    batch_format="pyarrow",
                )
                ray_ds_stream_.write_parquet(
                    output_dir,
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,  # https://github.com/ray-project/ray/issues/52481
                )

    def _ingest_fleurs_internal(
        self, output_dir: str, lang_subset: list[str] | None = None
    ):
        """Internal method for FLEURS ingestion."""
        # see https://huggingface.co/datasets/google/fleurs
        # doing it on a subset of languages for simplicity

        # Use provided subset or default to full subset
        langs_to_process = (
            lang_subset if lang_subset is not None else self.FLEURS_LAG_SUBSET
        )

        split_renaming = {"validation": "dev"}

        for lang in langs_to_process:
            for split in ["test", "validation", "train"]:
                fleurs_hf = load_dataset(
                    "google/fleurs",
                    lang,
                    split=split,
                    streaming=True,
                    trust_remote_code=True,
                )
                fleurs_hf = fleurs_hf.shuffle(seed=123, buffer_size=10000)
                ray_ds_stream_ = ray.data.from_huggingface(fleurs_hf)

                # Use batch-level text processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    FleursTextProcessor,
                    fn_constructor_kwargs={"lang": lang},
                    batch_size=1000,
                    batch_format="pyarrow",
                    concurrency=10,
                )

                # Audio processing
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    AudioTableProcessor,
                    fn_constructor_kwargs={
                        "audio_column": "audio.bytes",
                        "audio_format": "flac",  # or "ogg", "wav", etc.
                    },
                    batch_size=100,
                    batch_format="pyarrow",
                    concurrency=10,
                )
                ray_ds_stream_ = ray_ds_stream_.map_batches(
                    partial(
                        map_to_target_schema,
                        split=split_renaming.get(split, split),
                        corpus="fleurs",
                    ),
                    batch_size=100,
                    batch_format="pyarrow",
                )
                ray_ds_stream_.write_parquet(
                    output_dir,
                    partition_cols=["corpus", "split", "language"],
                    min_rows_per_file=10_000,
                    row_group_size=100,  # https://github.com/ray-project/ray/issues/52481
                )

    @staticmethod
    def _compute_distribution_stats_internal(
        parquet_dataset_root: str, output_path: str
    ):
        """Internal method for computing distribution statistics."""
        table = pa_ds.dataset(
            parquet_dataset_root, partitioning="hive", exclude_invalid_files=True
        ).to_table(columns=["language", "corpus", "audio_size"])
        pl_table = pl.from_arrow(table.combine_chunks())
        assert isinstance(pl_table, pl.DataFrame)
        stats = pl_table.group_by(["corpus", "language"]).agg(
            (pl.col("audio_size").sum() / 3600 / 16_000).alias("hours")
        )
        stats.write_csv(output_path, separator="\t")
        return output_path

    def ingest_mls(self, output_dir: str):
        """Ingest Multilingual LibriSpeech (MLS) datasets.

        Args:
            output_dir: Output directory path for processed Parquet files
        """
        print(f"Starting MLS ingestion to: {output_dir}")
        self._ingest_mls_internal(output_dir)
        print("MLS ingestion completed")

    def ingest_fleurs(self, output_dir: str):
        """Ingest FLEURS datasets.

        Args:
            output_dir: Output directory path for processed Parquet files
        """
        print(f"Starting FLEURS ingestion to: {output_dir}")
        self._ingest_fleurs_internal(output_dir)
        print("FLEURS ingestion completed")

    def compute_stats(self, parquet_dataset_root: str, output_path: str):
        """Compute distribution statistics from processed datasets.

        Args:
            parquet_dataset_root: Path to the root of partitioned Parquet dataset
            output_path: Output path for TSV statistics file
        """
        print(f"Computing stats for: {parquet_dataset_root}")
        result_path = self._compute_distribution_stats_internal(
            parquet_dataset_root, output_path
        )
        print(f"Statistics saved to: {result_path}")
        return result_path

    def test_dataset(self, dataset_path: str, **kwargs):
        """
        Test dataset functionality - redirects to dedicated dataloader_example module.

        Args:
            dataset_path: Path to the dataset directory
            **kwargs: Additional arguments passed to dataloader_example
        """
        print("📚 For dataset testing, use the dedicated dataloader_example module:")
        print(
            f"   python -m omnilingual_asr.dataprep.dataloader_example test_dataset --dataset_path='{dataset_path}'"
        )
        print("\n🔧 Available method:")
        print("   • test_dataset: Basic dataset testing with iterations")

        from dataloader_example import DataLoaderExample

        loader = DataLoaderExample()
        return loader.test_dataset(dataset_path, **kwargs)

    def run_short(
        self, output_dir: str, name: str = "all_asr_short", version: str = "0"
    ):
        """Run short data preparation pipeline (only 2 languages from FLEURS for quick testing).

        Args:
            output_dir: Base output directory path
            name: Dataset name (default: "all_asr_short")
            version: Dataset version (default: "0")
        """
        print("🚀 Starting SHORT data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")
        print(
            f"🌍 Processing only {len(self.FLEURS_SHORT_SUBSET)} languages from FLEURS: {self.FLEURS_SHORT_SUBSET}"
        )

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        # Only ingest FLEURS with short subset (no MLS for speed)
        print("🔄 Ingesting FLEURS with short language subset...")
        self._ingest_fleurs_internal(
            parquet_dataset_root, lang_subset=self.FLEURS_SHORT_SUBSET
        )

        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ SHORT pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path, num_iterations=5)
        return parquet_dataset_root, stats_path

    def run_full(self, output_dir: str, name: str = "all_asr", version: str = "0"):
        """Run complete data preparation pipeline (MLS + full FLEURS + stats).

        Args:
            output_dir: Base output directory path
            name: Dataset name (default: "all_asr")
            version: Dataset version (default: "0")
        """
        print("🚀 Starting FULL data preparation pipeline")
        print(f"📁 Output directory: {output_dir}")
        print(f"📊 Dataset name: {name}, Version: {version}")
        print(
            f"🌍 Processing {len(self.FLEURS_LAG_SUBSET)} languages from FLEURS: {self.FLEURS_LAG_SUBSET}"
        )
        print(
            f"📚 Processing {len(self.MLS_LANT_SUBSET)} languages from MLS: {self.MLS_LANT_SUBSET}"
        )

        parquet_dataset_root = str(Path(output_dir) / f"{name}/version={version}/")

        # Ingest both datasets
        print("🔄 Ingesting MLS datasets...")
        self.ingest_mls(parquet_dataset_root)
        print("🔄 Ingesting FLEURS datasets...")
        self.ingest_fleurs(parquet_dataset_root)

        # Compute statistics
        stats_path = Path(output_dir) / f"{name}/language_distribution_{version}.tsv"
        self.compute_stats(parquet_dataset_root, str(stats_path))

        print("✅ FULL pipeline finished successfully!")
        print(f"📈 Dataset ready at: {parquet_dataset_root}")
        print(f"📊 Statistics saved at: {stats_path}")

        # Test the dataset
        self.test_dataset(parquet_dataset_root, stats_path=stats_path)
        return parquet_dataset_root, stats_path


if __name__ == "__main__":
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()

    try:
        fire.Fire(DataPrepCLI)
    finally:
        # Clean shutdown of Ray
        if ray.is_initialized():
            ray.shutdown()
