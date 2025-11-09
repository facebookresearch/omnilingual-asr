# Wav2Vec2 Inference Pipeline

Quick start guide for transcribing audio with our multilingual ASR models.

---

## 1. Quick Start

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipeline = ASRInferencePipeline(model_card="omniASR_CTC_1B")
transcriptions = pipeline.transcribe(["/path/to/audio1.flac"], batch_size=1)
print(transcriptions[0])
```

---

## 2. Basic Usage

### 2.1 Audio Input Formats

Our pipeline accepts multiple input formats (`AudioInput`):

1. **File paths**: `["/path/to/audio.wav", "/path/to/audio.flac"]`
2. **Encoded audio binary data in memory**: `[open("audio.wav", "rb").read()]` or `[numpy_audio_array]` (int8)
3. **Decoded audio dicts**: `[{"waveform": tensor, "sample_rate": 16000}]`

The audio data is optionally decoded (`.wav/.flac`), resampled to 16kHz, converted to a mono-channel, and normalized before being ingested by the model.

> [!TIP]
> The models were mostly trained on audio durations of 30 seconds or less. Therefore, we do not recommend transcribing sequences longer than 30 seconds in a single sample.
> **⚠️ Important:** Currently only audio files shorter than 40 seconds are accepted for inference!

> [!TIP]
> We recommend keeping this preprocessing pipeline similar when integrating the model in downstream applications.

### 2.2 Batch Processing

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_1B")

audio_files = ["/path/to/audio1.flac", "/path/to/audio2.wav"]
transcriptions = pipeline.transcribe(audio_files, batch_size=2)

for file, trans in zip(audio_files, transcriptions):
    print(f"{file}: {trans}")
```

---

## 3. Model Types

### 3.1 Parallel Generation with CTC Models

The `omniASR_CTC_{300M,1B,3B,7B}` models are most useful for their parallel generation capabilities, resulting in faster throughput. Note that they don't support language conditioning or context examples.

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_CTC_3B", device=None)

audio_files = ["/path/to/audio1.flac", "/path/to/audio2.wav"]
transcriptions = pipeline.transcribe(audio_files, batch_size=2)

for file_path, text in zip(audio_files, transcriptions):
    print(f"CTC transcription - {file_path}: {text}")
```

### 3.2 Autoregressive Generation with Language-conditioned LLM Models

The `omniASR_LLM_{300M,1B,3B,7B}` models take an optional language code as argument to guide transcription towards the target language and script.
The language codes can be found in [lang_ids.py](/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py) and the language to language-id mapping is described in [our paper (Appendix A)](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/).

> [!TIP]
> We recommend providing the language code when possible as it helps with transcription quality.

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B")

audio_files = ["/path/to/russian_audio.wav", "/path/to/english_audio.flac", "/path/to/german_audio.wav"]

transcriptions = pipeline.transcribe(audio_files, lang=["rus_Cyrl", "eng_Latn", "deu_Latn"], batch_size=3)
```

### 3.3 Zero-Shot Generation

The `omniASR_LLM_7B_ZS` model is trained to accept in-context audio/transcription pairs to perform zero-shot inference on unseen languages via in-context learning. You can provide anywhere **from one to ten examples**, with more examples generally leading to better performance. Internally, the model uses exactly ten context slots: if fewer than ten examples are provided, samples are duplicated sequentially to fill all slots (and cropped to ten if more are provided).

> [!TIP]
> Similar to the other models, the zero-shot model has been trained with context samples of up to 30 seconds length and will likely perform suboptimally given longer samples (maximum audio length is 60 seconds).

```python
from omnilingual_asr.models.inference.pipeline import (
    ASRInferencePipeline,
    ContextExample
)

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")

context_examples = [
    ContextExample("/path/to/context_audio1.wav", "Hello world"),
    ContextExample("/path/to/context_audio2.wav", "How are you today"),
    ContextExample("/path/to/context_audio3.flac", "Nice to meet you")
]

transcriptions = pipeline.transcribe_with_context(
    ["/path/to/test_audio.wav"],
    context_examples=[context_examples],
    batch_size=1
)

print(f"Transcription: {transcriptions[0]}")
```

---

## 4. Developer Reference

### 4.1 Parquet Binary Audio Data Input
Our training parquet datasets follow a special format conversion (see [data preparation guide](/workflows/dataprep/README.md#21-schema) for schema details). We can use them in the inference pipeline as follows:

```python
import pyarrow.parquet as pq
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

ds = pq.ParquetDataset("/path/to/dataset/")
batch_data = ds._dataset.head(10).to_pandas()  # taking only few first samples
audio_bytes = batch_data["audio_bytes"].tolist()

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B")
transcriptions = pipeline.transcribe(audio_bytes, batch_size=4)

for i, text in enumerate(transcriptions):
    print(f"Sample {i+1}: {text}")
```

### 4.2 HuggingFace Datasets

```python
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load dataset
omni_dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
batch = next(omni_dataset.iter(5))

# Renaming columns to match the pipeline's expected input
audio_data = [{"waveform": x["array"], "sample_rate": x["sampling_rate"]}
              for x in batch["audio"]]

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_1B")
transcriptions = pipeline.transcribe(audio_data, batch_size=2)

for i, text in enumerate(transcriptions):
    print(f"Sample {i+1}: {text}")
```

### 4.3 Model Input Format Specification

This section contains the description of the `Wav2Vec2LlamaModel` input batch format. It is primarily for users who aim to integrate the model in their own applications while using the fairseq2 `Seq2SeqBatch` interface.

> [!WARNING]
> The behavior of `.forward()` can vary depending on the structure of the batch. In particular, the presence of extra fields in `batch.example` may cause the model to interpret the batch differently.

1. **Basic ASR**
```
  batch = Seq2SeqBatch(
      source_seqs=audio_tensor,           # [BS, T_audio, D_audio] - target audio
      source_seq_lens=audio_lengths,      # [BS] - actual audio lengths
      target_seqs=text_tensor,            # [BS, T_text] - target text tokens
      target_seq_lens=text_lengths,       # [BS] - actual text lengths
      example={}                          # Empty dict - no special fields needed
  )
```

2. **Language-Aware ASR**
```
  batch = Seq2SeqBatch(
      source_seqs=audio_tensor,           # [BS, T_audio, D_audio] - target audio
      source_seq_lens=audio_lengths,      # [BS] - actual audio lengths
      target_seqs=text_tensor,            # [BS, T_text] - target text tokens
      target_seq_lens=text_lengths,       # [BS] - actual text lengths
      example={
          "lang": ['mxs_Latn', ...]  # [BS] - language codes per sample
      }
  )
```
Our batch has some `lang_tokens` as part of the `batch.example` parameter, but they are only used for S2TT (kept for reference), not for LID conditioning, and can be ignored. The `lang` parameter should only consist of strings from [lang_ids.py](/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py).


3. **Zero-Shot Context**
```
  batch = Seq2SeqBatch(
      source_seqs=audio_tensor,           # [BS, T_audio, D_audio] - target audio
      source_seq_lens=audio_lengths,      # [BS] - actual audio lengths
      target_seqs=text_tensor,            # [BS, T_text] - target text tokens
      target_seq_lens=text_lengths,       # [BS] - actual text lengths
      example={
          "context_audio": [
              {"seqs": context_audio_1, "seq_lens": [audio_len_1]},  # Context audio 1 # List[Dict] - N context examples
              ...
              {"seqs": context_audio_BS, "seq_lens": [audio_len_BS]},  # BS Context audio
          ],
          "context_text": [               # List[Dict] - N context text examples
              {"seqs": context_text_1, "seq_lens": [text_len_1]},
              ...
              {"seqs": context_text_BS, "seq_lens": [text_len_BS]},   # BS Context text
          ]
      }
  )
```

### 4.4 Performance Optimization

This inference implementation serves as a reference baseline using PyTorch with BF16 precision and basic KV-caching. We've [benchmarked](/tests/integrations/test_performance_benchmark.py) Real-Time Factor (RTF) on an A100 for single-request setups — the kind you'd encounter when running locally — to give you a sense of what to expect (see [profiling results](/README.md#model-architectures)).

> [!TIP]
> These results are significantly memory-bottlenecked, meaning we spend more time shuffling the model in and out of tensor core memory than actually computing (~2TB/s memory bandwidth).
> As a rough guide, you can expect throughput to scale with memory bandwidth - about 1.75x faster on H100s (~3.5TB/s) and 0.5x on V100s (~1TB/s).


### 4.5 Punctuation and Capitalization

Our models are trained to output transcripts in spoken form without any punctuation or capitalization. If you would like transcripts in written form, we recommend passing our model's outputs through a third-party library to add punctuation, such as [this one](https://github.com/oliverguhr/deepmultilingualpunctuation). Note, however, that most punctuation libraries only cover a small subset of the 1600+ languages supported by our model.
