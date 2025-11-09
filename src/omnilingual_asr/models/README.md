# Models

This directory contains model implementations and configurations for all open-source models. The implementations leverage fairseq2's configuration system for easy architecture reuse.

## Architecture Overview

The models follow a hierarchical design built on the Wav2Vec2 encoder foundation:

```

W2V:     [Audio 16kHz] → Wav2Vec2 Feature Extractor → Wav2Vec2 Encoder →   [Audio Embeddings]
                         (CNN downsampling ~320x)       (Transformer)      (1024/1280/2048-dim)

CTC:     [Audio 16kHz] → Wav2Vec2 Feature Extractor → Wav2Vec2 Encoder →    Linear Projection     → [Vocab Logits]
                         (CNN downsampling ~320x)       (Transformer)      (1024/1280/2048-dim)      (9812 vocab)

LLM:     [Audio 16kHz] → Wav2Vec2 Feature Extractor → Wav2Vec2 Encoder →    Linear Projection     → LLama Decoder       → [Vocab Logits]
                         (CNN downsampling ~320x)       (Transformer)      (1024/1280/2048-dim)  (Transformer, 4096-dim)   (9812 vocab)
```

The W2V ([./wav2vec2_ssl](./wav2vec2_ssl/)) and CTC ([./wav2vec2_asr](./wav2vec2_asr/)) models use fairseq2's existing implementations with updated configurations for new training data and can be found at [here](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models/wav2vec2). The LLM family ([./wav2vec2_llama](./wav2vec2_llama)) introduces an encoder-decoder architecture implemented in this repository.


## Model Inputs and Outputs

**W2V Models:**
  - **Input**: Raw audio waveform (16kHz)
  - **Output**: Contextualized audio embeddings (1024-dim for 300M, 1280-dim for 1B, 2048-dim for 3B/7B)

The W2V encoder produces contextual audio embeddings with dimensions varying by model size. It is most useful as a starting point for building your own model architecture given the rich embedding information, similar to our CTC and LLM variants.

**CTC Models:**
  - **Input**: Raw audio waveform (16kHz)
  - **Output**: Vocabulary probability distributions over time steps (parallel prediction)
  - **Vocabulary sizes**: 9812 tokens

The CTC model projects these embeddings directly to vocabulary logits with a simple linear projection for parallel prediction with CTC alignment. It is most useful for on-device transcription tasks due to its non-autoregressive nature.

**LLM Models:**
- **Input**: Raw audio waveform (16kHz) + optional language ID or context examples
- **Output**: Text tokens (autoregressive generation via beam search)
- **Vocabulary sizes**: 9812 tokens
- **Internal dimensions**: Audio embeddings projected to Llama space (4096-dim), then to vocab

The LLM family introduces an encoder-decoder architecture that projects the audio embeddings to match the LLama decoder's input space for autoregressive text generation. It has the best transcription capabilities and flexibility given its diverse input combinations.

The additional inputs, such as the language ID and context examples, are added as part of the input batch with interspersed special tokens. The concrete grammar can be traced in the `(...)_create_syntax()` functions in the [model definition](./wav2vec2_llama/model.py).


## LLM Model Variants

### LLM+LID (Language Conditioning)

The LID-enabled variant supports either audio + language_id or audio-only inputes. During training, the model was exposed to a 50/50 split of samples with and without language identification tokens, enabling robust performance in both scenarios.

### LLM+ZS (Zero-Shot with Context)

The zero-shot variant requires exactly 10 context examples (audio-text pairs) for proper inference. This architectural constraint ensures consistent few-shot learning performance and is enforced through input validation. If fewer than 10 context examples are provided, we recommend to repeat the available examples, similar to our [inference API](./inference/pipeline.py).

## Input Validation

The `Wav2Vec2LlamaModel` implementation is used for both the LLM+LID and LLM+ZS models and performs input validation at every forward pass with `ensure_valid_forward_inputs` to ensure correct behavior:

- **LLM+LID**: Audio input with optional language identification
- **LLM+ZS**: Audio input with exactly 10 context examples

These additional inputs are encoded in the `.extra` field of the `Seq2SeqBatch` to give researchers flexibility while keeping a minimal stable interface.

