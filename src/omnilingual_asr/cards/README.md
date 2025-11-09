# Assets

fairseq2 manages models, tokenizers, and datasets as [assets](https://facebookresearch.github.io/fairseq2/stable/basics/assets.html#asset-store-configuration) defined in YAML files with these key fields:

For example, a model definition has the following parameters:

```yaml
name: omniASR_CTC_300M
model_family: wav2vec2_asr
model_arch: 300m
checkpoint: https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt
tokenizer_ref: omniASR_tokenizer
```

### Usage Examples

```python
from fairseq2.models.hub import load_model

model = load_model("omniASR_CTC_300M")
```

Or in a training recipe configuration (e.g., [`/workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml`](/workflows/recipes/wav2vec2/asr/configs/ctc-finetune.yaml)):

```yaml

model:
  name: "omniASR_CTC_300M"

trainer:
  (...)

optimizer:
  (...)
```

### Field Details

* `name`: Unique identifier for loading assets.

* `model_family`: Maps to model implementation (e.g., `Wav2Vec2LlamaModel`).

* `model_arch`: Specific configuration for the model family (e.g., [`1b`](/src/omnilingual_asr/models/wav2vec2_llama/config.py) for `wav2vec2_llama`)

* `checkpoint`: Model storage URI, can be a local path (`"$HOME/.cache/"`), a direct download link (`"https://dl.fbaipublicfiles.com/mms/omniASR_LLM_300M.pt"`) or a reference to a huggingface repository (`"hg://qwen/qwen2.5-7b"`) if the model is in a `.safetensors` format.

* `tokenizer_ref`: Links to tokenizer asset for training.

Add custom assets by duplicating existing cards and updating the name and checkpoint. Separate multiple entries with `---`.
