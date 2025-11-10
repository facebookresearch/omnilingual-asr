<div align="center">
  <a href="https://github.com/facebookresearch/omnilingual-asr">
    <img src="./omniASR_header.jpg" alt="Header image with a collage of on-the-ground photos from the transcription gathering efforts in Pakistan and Liberia." width="100%" />
  </a>
  <p><i>Photographs captured during corpus creation efforts in Pakistan and Liberia.</i></p>
</div>

<div align="center">

# Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages

</div>

Omnilingual ASR is an open-source speech recognition system supporting over 1,600 languages — including hundreds never previously covered by any ASR technology. Designed for broad accessibility, it enables new languages to be added with just a few paired examples without requiring specialized expertise or large datasets. By combining scalable zero-shot learning with a flexible model family, Omnilingual ASR aims to make speech technology more inclusive and adaptable for communities and researchers worldwide.

<p align="center">
  <a href="https://huggingface.co/spaces/facebook/omniasr-transcriptions"><strong>🤗 Huggingface Demo</strong></a> &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://huggingface.co/datasets/facebook/omnilingual-asr-corpus"><strong> Huggingface Dataset</strong></a> &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/"><strong> Paper</strong></a> &nbsp;&nbsp;•&nbsp;&nbsp;
  <a href="http://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition"><strong> Blogpost</strong></a>
</p>

<div align="center">
  <img src="./result_table.png" alt="Performance results table" width="100%" />
  <p><i>Our 7B-LLM-ASR system achieves state-of-the-art performance across 1,600+ languages, with character error rates (CER) below 10 for 78% of those languages.</i></p>
</div>

---

## Documentation

<table align="center" style="width:100%; border: none; background-color: transparent;">
  <tr style="background-color: transparent;">
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3 align="left">Quick Start</h3>
      <ul>
        <li align="left"><strong><a href="#installation">Installation & Basic Usage</a></strong> - Setup and first transcription</li>
        <li align="left"><strong><a href="src/omnilingual_asr/models/inference/README.md">Inference Pipeline</a></strong> - Comprehensive transcription guide with batch processing, language conditioning, and context examples</li>
        <li align="left"><strong><a href="#supported-languages">Supported Languages</a></strong> - View the complete list of 1600+ supported languages</li>
      </ul>
    </td>
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3 align="left">Models & Architecture</h3>
      <ul>
        <li align="left"><strong><a href="#model-architectures">Model Specifications</a></strong> - Available models, parameters, and memory requirements</li>
        <li align="left"><strong><a href="src/omnilingual_asr/models/README.md">Architecture Overview</a></strong> - Technical details on W2V, CTC, and LLM model families</li>
        <li align="left"><strong><a href="src/omnilingual_asr/cards/README.md">Asset Management</a></strong> - Configuration system for models, tokenizers, and datasets</li>
      </ul>
    </td>
    <td align="center" valign="top" width="33%" style="border: none; padding: 10px;">
      <h3 align="left">Training & Data Pipeline</h3>
      <ul>
        <li align="left"><strong><a href="workflows/dataprep/README.md">Data Preparation</a></strong> - End-to-end guide for multilingual dataset preparation, HuggingFace integration, and parquet processing</li>
        <li align="left"><strong><a href="workflows/recipes/wav2vec2/asr/README.md">Training Recipes</a></strong> - Pre-configured workflows for CTC and LLM model training</li>
      </ul>
    </td>
  </tr>
</table>

---

## Installation

The models were developed using [fairseq2](https://github.com/facebookresearch/fairseq2), a research-focused sequence modeling toolkit. While we provide a **reference** inference pipeline that works across platforms, audio support requires [libsndfile](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#system-dependencies) (Mac: `brew install libsndfile`; Windows may need an additional [setup](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file#installing-on-windows)).

```bash
# using pip
pip install omnilingual-asr

# using uv
uv add omnilingual-asr
```

## Inference

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")

audio_files = ["/path/to/eng_audio1.flac", "/path/to/deu_audio2.wav"]
lang = ["eng_Latn", "deu_Latn"]
transcriptions = pipeline.transcribe(audio_files, lang=lang, batch_size=2)
```
More details on running specific models can be found in the [src/omnilingual_asr/models/inference](/src/omnilingual_asr/models/inference/README.md) directory.

<br>

<table style="width:100%; border: 2px solid #D8A22E; border-radius: 8px; background-color: #302604; padding: 15px;">
<tr>
<td style="width: 40px; vertical-align: top; padding-top: 5px;">⚠️</td>
<td style="vertical-align: top;">
<strong>Important:</strong> Currently only audio files shorter than 40 seconds are accepted for inference. We plan to add support for transcribing unlimited-length audio files shortly.
</td>
</tr>
</table>

### 🌐 Supported Languages

To view the full list of 1600+ supported languages, you can access the language list [programmatically](/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py):

```python
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# Print all supported languages
print(f"Total supported languages: {len(supported_langs)}")
print(supported_langs)

# Check if a specific language is supported
if "eng_Latn" in supported_langs:
    print("English (Latin script) is supported!")
```
Languages follow the format `{language_code}_{script}`, for example `eng_Latn` - English (Latin script), `cmn_Hans` - Mandarin Chinese (Simplified), ...

### 🤗 Using the HuggingFace Dataset

We provide a large-scale multilingual speech dataset on HuggingFace under [CC-BY-4.0 License](./LICENSE-CC-BY-4.0.md): [`facebook/omnilingual-asr-corpus`](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus).
This dataset can be directly used with our inference pipeline for evaluation or testing:

```bash
pip install "omnilingual-asr[data]"
```
```python
from datasets import load_dataset
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# Load dataset for a specific language (e.g., Ligurian)
omni_dataset = load_dataset("facebook/omnilingual-asr-corpus", "lij_Latn", split="train", streaming=True)
batch = next(omni_dataset.iter(5))

# Convert to pipeline input format
audio_data = [{"waveform": x["array"], "sample_rate": x["sampling_rate"]}
              for x in batch["audio"]]

# Run inference
pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B")
transcriptions = pipeline.transcribe(audio_data, batch_size=2)

# Display results
for i, (transcription, original_text) in enumerate(zip(transcriptions, batch["raw_text"]), 1):
    print(f"\n Sample {i}:")
    print(f"   Ground Truth: {original_text}")
    print(f"   Predicted:    {transcription}")
```

## Model Architectures

| Model Name          | Features      | Parameters | Download Size (FP32) | Inference VRAM¹ | Real-Time Factor¹ (relative speed)² |
|:--------------------|:--------------|:----------:|:--------------------:|:---------------:|:-----------------------------------:|
| [`omniASR_W2V_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-300M.pt)      | SSL  | 317,390,592   | 1.2 GiB | | |
| [`omniASR_W2V_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-1B.pt)          | SSL  | 965,514,752   | 3.6 GiB | | |
| [`omniASR_W2V_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-3B.pt)          | SSL  | 3,064,124,672 | 12.0 GiB | | |
| [`omniASR_W2V_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-W2V-7B.pt)          | SSL  | 6,488,487,168 | 25.0 GiB | | |
| [`omniASR_CTC_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-300M.pt)      | ASR  | 325,494,996   | 1.3 GiB   | ~2 GiB  | 0.001 (96x) |
| [`omniASR_CTC_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-1B.pt)          | ASR  | 975,065,300   | 3.7 GiB   | ~3 GiB  | 0.002 (48x) |
| [`omniASR_CTC_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-3B.pt)          | ASR  | 3,080,423,636 | 12.0 GiB  | ~8 GiB  | 0.003 (32x) |
| [`omniASR_CTC_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-CTC-7B.pt)          | ASR  | 6,504,786,132 | 25.0 GiB  | ~15 GiB | 0.006 (16x) |
| [`omniASR_LLM_300M`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt)      | ASR with optional language conditioning  | 1,627,603,584 | 6.1 GiB   | ~5 GiB  | 0.090 (~1x) |
| [`omniASR_LLM_1B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt)          | ASR with optional language conditioning  | 2,275,710,592 | 8.5 GiB   | ~6 GiB  | 0.091 (~1x) |
| [`omniASR_LLM_3B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt)          | ASR with optional language conditioning  | 4,376,679,040 | 17.0 GiB  | ~10 GiB | 0.093 (~1x) |
| [`omniASR_LLM_7B`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt)          | ASR with optional language conditioning  | 7,801,041,536 | 30.0 GiB  | ~17 GiB | 0.092 (~1x) |
| [`omniASR_LLM_7B_ZS`](https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B-ZS.pt)    | Zero-Shot ASR | 7,810,900,608 | 30.0 GiB | ~20 GiB | 0.194 (~0.5x) |
| [`omniASR_tokenizer`](https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model) | Tokenizer for most of architectures (except omniASR_LLM_7B) | - | 100 KiB | - | |
| [`omniASR_tokenizer_v7`](https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_v7.model) | Tokenizer for omniASR_LLM_7B model | - | 100 KiB | - | |

¹ (batch=1, audio_len=30s, BF16, A100)
<br>
² Relative speed to `omniASR_LLM_7B`

###  Model Download & Storage
- **Automatic Download**: Models are automatically downloaded on first use during training or inference
- **Storage Location**: Models are saved to [`~/.cache/fairseq2/assets/`](https://facebookresearch.github.io/fairseq2/stable/basics/assets.html#the-asset-store-system)

### Architecture Documentation
We provide a high-level model architecture overview in the model directory ([`src/omnilingual_asr/models`](/src/omnilingual_asr/models)), with individual configurations for each model family in the respective directories:
- **SSL Models**: [`src/omnilingual_asr/models/wav2vec2_ssl`](/src/omnilingual_asr/models/wav2vec2_ssl/)
- **CTC Models**: [`src/omnilingual_asr/models/wav2vec2_asr`](/src/omnilingual_asr/models/wav2vec2_asr/)
- **LLM Models**: [`src/omnilingual_asr/models/wav2vec2_llama`](/src/omnilingual_asr/models/wav2vec2_llama/)

## Training
To further finetune the released checkpoints on your own data, use our [data preparation guide](/workflows/dataprep/README.md) followed by the [finetuning recipe guide](/workflows/recipes/wav2vec2/asr/README.md).

## License
Omnilingual ASR code and models are released under the [Apache 2.0](./LICENSE).

## Citation
If you use the omnilingual ASR model suite in your research and wish to cite us, please use the following BibTeX entry (arxiv version will be added soon)!

### For Reference Managers & LaTeX
This version uses `et al.` for a clean, universally compatible entry. **This is the recommended version for direct copying.**
```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} et al.},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```

### Full Author List
The full list of contributors to this work is provided below for complete attribution.
<details>
<summary><strong>Click to expand the full author list</strong></summary>

<table style="width:100%; border: none; background-color: transparent;">
  <tr style="background-color: transparent;">
    <td valign="top" width="33%" style="border: none;">
      <ul>
        <li>Keren, Gil</li>
        <li>Kozhevnikov, Artyom</li>
        <li>Meng, Yen</li>
        <li>Ropers, Christophe</li>
        <li>Setzler, Matthew</li>
        <li>Wang, Skyler</li>
        <li>Adebara, Ife</li>
        <li>Auli, Michael</li>
        <li>Chan, Kevin</li>
        <li>Cheng, Chierh</li>
      </ul>
    </td>
    <td valign="top" width="33%" style="border: none;">
      <ul>
        <li>Chuang, Joe</li>
        <li>Droof, Caley</li>
        <li>Duppenthaler, Mark</li>
        <li>Duquenne, Paul-Ambroise</li>
        <li>Erben, Alexander</li>
        <li>Gao, Cynthia</li>
        <li>Mejia Gonzalez, Gabriel</li>
        <li>Lyu, Kehan</li>
        <li>Miglani, Sagar</li>
        <li>Pratap, Vineel</li>
      </ul>
    </td>
    <td valign="top" width="33%" style="border: none;">
      <ul>
        <li>Sadagopan, Kaushik Ram</li>
        <li>Saleem, Safiyyah</li>
        <li>Turkatenko, Arina</li>
        <li>Ventayol-Boada, Albert</li>
        <li>Yong, Zheng-Xin</li>
        <li>Chung, Yu-An</li>
        <li>Maillard, Jean</li>
        <li>Moritz, Rashel</li>
        <li>Mourachko, Alexandre</li>
        <li>Williamson, Mary</li>
        <li>Yates, Shireen</li>
      </ul>
    </td>
  </tr>
</table>

</details>

