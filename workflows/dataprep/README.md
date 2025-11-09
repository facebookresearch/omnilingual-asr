# ASR Dataset Preparation

Utilities and examples for preparing multilingual speech datasets for efficient training and data loading. The preparation scripts convert popular HuggingFace audio datasets into a standardized format optimized for massively multilingual speech model training.

## 1. Installation

Install the required dependencies:

```bash
# using pip
pip install omnilingual-asr[data]
# or using uv
uv add "omnilingual-asr[data]"
```

This installs standard data libraries (`pyarrow`, `polars`, `pandas`) for data loading.
For data preparation with HuggingFace datasets, you'll also need:
- [`ray`](https://github.com/ray-project/ray): Distributed processing framework
- [`datasets`](https://github.com/huggingface/datasets): HuggingFace datasets library


## 2. Parquet Dataset

ASR datasets are stored as [parquet files](https://en.wikipedia.org/wiki/Apache_Parquet), partitioned by `corpus`, `split`, and `language`:

```bash
dataset_root_dir/version=0/
├── corpus=mls/
│   ├── split=train/
│   │   ├── language=deu_Latn/
│   │   │   └── part-*.parquet
│   │   └── language=fra_Latn/
│   │       └── part-*.parquet
│   └── split=dev/
│       └── ...
└── corpus=fleurs/
    └── ...
```

Use partition filters during data loading to:
1. Load only training data (`split=train`)
2. Perform weighted data mixture sampling across corpus-language combinations
3. Load specific languages for evaluation (`split=dev` and `language is in ["deu_Latn", "fra_Latn"]`)


### 2.1 Schema

Resulting dataset has the following minimal pyarrow schema:

```python
text: string

audio_bytes: list<element: int8>
  child 0, element: int8

audio_size: int64

corpus: dictionary<values=string, indices=int32, ordered=0>

split: dictionary<values=string, indices=int32, ordered=0>

language: dictionary<values=string, indices=int32, ordered=0>
```

Schema details:
- **`text`**: Contains the normalized text transcription of the audio sample.

- **`audio_bytes`**: Contains the compressed (flac or ogg) binary audio data as a list of bytes. All audio data is converted to a 16kHz mono-channel format. Instead of more common `pa.binary()` we use an equivalent `pa.list_(pa.int8())` type which does not require additional copying when converting from pyarrow to pandas. Use `from omnilingual_asr.datapre.audio_tools import binary_to_list_int8` for the fast conversion.

- **`audio_size`**: Contains the size of the decoded audio waveform. Used in dataloading to filter out audio samples that are too short and to create batches with audio samples of a similar length. Also used to compute the audio duration in seconds (as `audio_size / 16_000`) which serves as a reference for temperature sampling aggregated statistics.

- **`corpus`**: Contains the corpus name where the data originates from (e.g., "fleurs" or "mls").

- **`split`**: Contains the dataset split (e.g., "train", "dev" or "test").

- **`language`**: Contains the [standardized language code](/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) (e.g., "deu_Latn" for German). Used for language-specific data filtering and statistics generation across the corpora.

**NOTE:** The parquet files are written with `row_group_size=100` to mitigate high memory footprints when streaming data from multiple parquet sources and to allow efficient shuffling.

### 2.2 Data Processing Pipeline

This example demonstrates the data preparation pipeline using subsets from the [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) and [FLEURS](https://huggingface.co/datasets/google/fleurs) corpora.

<details>
<summary>FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)</summary>

- **Paper**: [FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech](https://arxiv.org/abs/2205.12446)
- **HuggingFace**: [google/fleurs](https://huggingface.co/datasets/google/fleurs)

</details>

<details>
<summary>MLS (Multilingual LibriSpeech)</summary>

- **Paper**: [MLS: A Large-Scale Multilingual Dataset for Speech Research](https://arxiv.org/abs/2012.03411)
- **HuggingFace**: [facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech)

</details>

The pipeline uses HuggingFace's [datasets integration](https://docs.ray.io/en/latest/data/api/doc/ray.data.from_huggingface.html) with Ray for distributed processing (local processing also supported). Input datasets are shuffled with a configurable window size (1k-10k typically sufficient, as processed files are consumed in row groups of 100).

**Text Processing**:
- Language-specific text normalization (punctuation, lowercase, digit-only word removal)
- Language remapping

**Audio Processing**:
- Audio binary conversion to byte lists
- Read validation and resampling to 16kHz
- Audio size computation


## 3. Automatic Dataset Generation

See [`hf_dataset_ingestion_example.py`](./hf_dataset_ingestion_example.py) for a complete data ingestion example using Ray. After ingestion, the pipeline:
- Computes corpus-language audio duration statistics
- Instantiates a mixture ASR DataReader locally
- Runs test iterations to verify functionality

### 3.1 Command Line Interface

The script provides a CLI with two primary modes:

```bash
# Quick testing with only 2 languages (en_us, fr_fr) from FLEURS, ~5-10 minutes
python hf_dataset_ingestion_example.py run_short /path/to/output/dir
```

```bash
# Complete example with a subset of MLS (7) + FLEURS (5) languages, ~90 minutes
python hf_dataset_ingestion_example.py run_full /path/to/output/dir
```

It also supports optional `--name <str>` and `--version <int>` arguments for versioning:

```bash
python hf_dataset_ingestion_example.py run_short /path/to/output/dir --name my_asr_data --version 1
```

You can also process individual datasets or add additional functionality by using the following modes:

```bash
# Process only Multilingual LibriSpeech datasets
python hf_dataset_ingestion_example.py ingest_mls /path/to/output/dir

# Process only FLEURS datasets
python hf_dataset_ingestion_example.py ingest_fleurs /path/to/output/dir

# Generate statistics from existing processed data
python hf_dataset_ingestion_example.py compute_stats /path/to/parquet/dataset /path/to/output/stats.tsv
```

### 3.2 Additional Utilities

We provide a few additional utilities to help with audio dataset preparation.

#### 3.2.1 Text Processing

[`text_tools.py`](./text_tools.py):
- `text_normalize(text, iso_code, lower_case=True, remove_numbers=True, remove_brackets=False)`: Applies language-specific text normalization with configurable options

#### 3.2.2 Audio Processing

[`audio_tools.py`](./audio_tools.py):
- `AudioTableProcessor`: Main class for processing audio data in PyArrow tables with resampling and format conversion
- `map_to_target_schema(batch, split, corpus)`: Transforms batches to the target schema format
- `binary_to_list_int8(binary_array)`: Efficiently converts PyArrow `BinaryArray` to `ListArray` of int8
- `bytes_to_tensor(audio_arr, target_sample_rate=16_000)`: Converts numpy array of audio bytes to waveform array


## 4. Mixture Parquet ASR Dataloader

After creating the parquet dataset, you can use the [`MixtureParquetAsrDataset`](/src/omnilingual_asr/datasets/impl/mixture_parquet_asr_dataset.py) to iterate over the dataset for training or evaluation.

### 4.1 Dataloader Features

- **Weighted Sampling**: Sampling across languages and corpora
- **Streaming**: Efficient streaming from parquet files with configurable buffering
- **Audio Processing**: Automatic audio decoding, normalization and feature extraction (waveform or fbank)
- **Text Processing**: Tokenization, filtering
- **Batching Strategies**: Static or dynamic length-based batching
- **SpecAugment**: Spectrum based augmentation for audio data

### 4.2 Dataloading Verification

We provide a basic CLI to verify dataset creation and dataloading at [`dataloader_example.py`](./dataloader_example.py):

```bash
python -m workflows.dataprep.dataloader_example \
--dataset_path="root_ds/all_asr/version=0" \
--split="train" \
--num_iterations=10
```

This will:
1. Load the `omniASR_tokenizer`
2. Create the dataset (which will scan all files and organize them by corpus and language)
3. Create a data reader using task_config and storage_config
4. Iterate through a few batches and show statistics

### 4.3 Dataloading Integration

Once the dataset is prepared and verified:

```python
from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import MixtureParquetAsrDataset

# Create dataset and reader
dataset = MixtureParquetAsrDataset.from_path(path=dataset_path, name="my_asr")
reader = dataset.create_reader(
    split="train",
    tokenizer=tokenizer,
    gangs=gangs,
    dtype=torch.float32,
    storage_config=storage_config,
    task_config=task_config,
)

# Iterate through batches
for batches in reader:
    for batch in batches:
        # batch.source_seqs: audio features [batch, time, features]
        # batch.target_seqs: text tokens [batch, seq_len]
        # batch.source_seq_lens: audio sequence lengths
        # batch.target_seq_lens: text sequence lengths
        pass
```

If you wish to use the created dataset as part of the [finetuning recipe](/workflows/recipes/wav2vec2/asr/README.md), you need to create a new asset card under `src/omnilingual_asr/cards/datasets/my_dataset.yaml`:

```yaml
name: my_dataset
dataset_family: mixture_parquet_asr_dataset
dataset_config:
  data: /path/to/the/dataset
tokenizer_ref: omniASR_tokenizer
```

After the asset card is defined, you can simply reference its `name` in the recipe configuration file to query it during training or evaluation.

## 5. Citations

If you use this data preparation pipeline or the supported datasets in your research and wish to cite it, please use the following BibTex entries.

```bibtex
@article{conneau2022fleurs,
  title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
  author={Conneau, Alexis and Ma, Min and Khanuja, Simran and Zhang, Yu and Axelrod, Vera and Dalmia, Siddharth and Riesa, Jason and Rivera, Clara and Bapna, Ankur},
  journal={arXiv preprint arXiv:2205.12446},
  year={2022}
}
```

```bibtex
@article{pratap2020mls,
  title={MLS: A Large-Scale Multilingual Dataset for Speech Research},
  author={Pratap, Vineel and Xu, Qiantong and Sriram, Anuroop and Synnaeve, Gabriel and Collobert, Ronan},
  journal={arXiv preprint arXiv:2012.03411},
  year={2020}
}
```
