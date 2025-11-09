# Wav2Vec2 ASR Recipes

We provide two fairseq2 recipes, which are pre-configured training and evaluation workflows that combine models, datasets, and hyperparameters into reproducible experiments that you can run with a single command.

`wav2vec2.asr.recipe` - Training recipe supporting:
- Train the CTC model starting with a W2V encoder ([./configs/ctc-from-encoder.yaml](./configs/ctc-from-encoder.yaml))
- Train the CTC model given a CTC checkpoint ([./configs/ctc-finetune.yaml](./configs/ctc-finetune.yaml))
- Train the LLM model starting with a W2V encoder ([./configs/llm-from-encoder.yaml](./configs/llm-from-encoder.yaml))
- Train the LLM model given a LLM checkpoint ([./configs/llm-finetune.yaml](./configs/llm-finetune.yaml))

`wav2vec2.asr.eval.recipe` - Evaluation recipe for testing model performance on fleurs-mls-mini from our [data preparation tutorial](/workflows/dataprep/README.md).

Each recipe stores its configurations as YAML files in its config directory [`wav2vec2.asr.config`](./configs) and [`wav2vec2.asr.eval.config`](./eval/configs). The evaluation recipe reuses the datasets from the training recipe.

### Dataset Backends

Our dataset implementation supports flexible combinations of storage and task backends:
- Mixture parquet backend - [`MixtureParquetStorage`](/src/omnilingual_asr/datasets/storage/mixture_parquet_storage.py)
- Manifest-based backend - [`ManifestStorage`](/src/omnilingual_asr/datasets/storage/manifest_storage.py)
- SSL task  - [`SslTask`](/src/omnilingual_asr/datasets/tasks/ssl_task.py)
- ASR task  - [`AsrTask`](/src/omnilingual_asr/datasets/tasks/asr_task.py)

The SSL task returns a `SequenceBatch` (audio-only) rather than `Seq2SeqBatch` (audio + text), making integration non-trivial but kept here as a reference.
We also include the manifest-based storage implementation as an alternative to parquet; the codebase includes comprehensive comments to guide implementation.

### Usage

Set an output directory for the resulting artifacts (model checkpoints during training or hypothesis generation during evaluation) and run the recipe:

```bash
> cd omnilingual_asr
> export OUTPUT_DIR="/path/to/artifact/directory"
> python -m workflows.recipes.asr $OUTPUT_DIR --config-file workflows/recipes/asr/configs/ctc-finetune.yaml
```

## Core Recipe Structure

```bash
.
└── wav2vec2/asr
    ├── eval
    │   ├── configs/            # Eval recipe YAML recipe
    │   ├── default_config.py
    │   └── recipe.py           # Eval logic
    ├── configs/                # Train recipe YAML configs
    ├── criterion.py
    ├── dataset_selector.py     # Dataset backend switching
    ├── default_config.py
    ├── recipe.py               # Train logic
    └── wer_calculator.py       # WER metric
```

### Training Strategies

We offer the following recommendations for users who are compute-constrained and wish to fine-tune our smaller CTC checkpoints on specific low-resource languages. As reported in Section 5.7.5 of the paper, fine-tuning smaller-scale CTC models in these settings produced models that were competitive with our 7B LLM ASR model on the specific languages. Of course, the optimal fine-tuning hyper-parameters will vary from language to language, but the following presets performed generally well and serve as a good starting point.

```
dataset:
  (...)
  asr_task_config:
    max_audio_len: 960_000      # 60s at 16kHz
    max_num_elements: 7_680_000 # maximum of eight 60s samples, or more samples at lower lengths

optimizer:
  config:
    lr: 1e-05

trainer:
  grad_accumulation:
    num_batches: 4 # Increase gradient accumulation if running OOM during training, we use 32 GPUs for 300M, 64 GPUs for 1B and 96 GPUs for 3B

regime:
  num_steps: 5_000
```

We provide an example configuration under [`ctc-finetune-recommendation.yaml`](./configs/ctc-finetune-recommendation.yaml) to further train our CTC checkpoint, or use [`ctc-from-encoder-recommendation.yaml`](./configs/ctc-from-encoder-recommendation.yaml) to train your own CTC model from our W2V encoder checkpoint.
