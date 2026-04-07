# Installing Omnilingual ASR on aarch64 (DGX Spark / GB10)

> Tested: 2026-04-06, Ubuntu 24.04, Python 3.11, CUDA 12.9, NVIDIA GB10 (Blackwell)

## Problem

The standard `pip install omnilingual-asr` does not work on **aarch64** because `fairseq2n` (the C++ backend of fairseq2) does not provide pre-built wheels for this platform. You need to build it from source.

---

## 1. Create a Virtual Environment

```bash
cd ~/projects/omnilingual-asr
uv venv --python 3.11
source .venv/bin/activate
```

## 2. Install PyTorch with CUDA

```bash
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu129
```

Verify:
```bash
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0)}')"
# Expected output: torch=2.11.0+cu129, cuda=True, gpu=NVIDIA GB10
```

## 3. System Dependencies

```bash
sudo apt install -y libsndfile-dev
```

## 4. Build fairseq2 from Source

### 4.1 Clone the Repository

```bash
git clone --recurse-submodules https://github.com/facebookresearch/fairseq2.git /tmp/fairseq2
```

### 4.2 Install Build Tools

```bash
uv pip install cmake ninja numpy
```

### 4.3 CMake Configure

```bash
# Important: unset CONDA_PREFIX if conda is active, otherwise cmake will fail
unset CONDA_PREFIX

cd /tmp/fairseq2/native
rm -rf build

cmake -GNinja \
  -DFAIRSEQ2N_USE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="100-real;100-virtual" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DFAIRSEQ2N_SUPPORT_IMAGE=OFF \
  -B build
```

> **Notes:**
> - `CMAKE_CUDA_ARCHITECTURES="100-real;100-virtual"` — for Blackwell (GB10). For Ampere (A100) use `"80-real;80-virtual"`, for Volta `"70-real;70-virtual"`
> - `CMAKE_POLICY_VERSION_MINIMUM=3.5` — workaround for cmake 4.x compatibility issues with submodules
> - `FAIRSEQ2N_SUPPORT_IMAGE=OFF` — disables JPEG/PNG support (not needed for ASR, avoids an additional cmake error)

### 4.4 Build

```bash
cmake --build build
```

Expect `[37/37]` — all targets built successfully.

Verify the shared library was created:
```bash
ls build/src/fairseq2n/libfairseq2n.so*
# Should show: libfairseq2n.so, libfairseq2n.so.0, libfairseq2n.so.0.9.0
```

## 5. Install Python Packages into the venv

> **Important:** Use `uv pip`, NOT bare `pip` — otherwise packages may end up in conda's site-packages!

### 5.1 fairseq2n + fairseq2

```bash
uv pip install --no-deps /tmp/fairseq2/native/python
uv pip install --no-deps /tmp/fairseq2
```

### 5.2 All fairseq2 Dependencies

```bash
uv pip install \
  importlib_resources importlib_metadata "ruamel.yaml" rich psutil \
  sacrebleu "tiktoken[blobfile]" torcheval transformers wandb tensorboard \
  clusterscope editdistance mypy-extensions retrying xxhash s3fs \
  huggingface_hub safetensors soundfile docstring_parser jiwer \
  typing_extensions tabulate packaging pyyaml tqdm
```

### 5.3 omnilingual-asr

First, relax the fairseq2 version constraint in `pyproject.toml` (the source build produces 0.9.x, but the constraint requires `<=0.6.0`):

```diff
- "fairseq2[arrow]>=0.5.2,<=0.6.0",
+ "fairseq2[arrow]>=0.5.2",
```

Then install:
```bash
cd ~/projects/omnilingual-asr
uv pip install -e . --no-deps
uv pip install pyarrow numba pandas numpy kenlm polars torchaudio editdistance
```

## 6. Verification

```bash
python3 -c "
import torch
print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, ContextExample
print('All imports OK!')
"
# Expected output:
# torch=2.11.0+cu129, cuda=True
# All imports OK!
```

## 7. Downloading Models (via tmux)

Checkpoint downloads (~30 GiB) take time. Use tmux so you don't lose your session if disconnected.

### tmux Cheat Sheet

```bash
# Create a new session
tmux new -s asr

# Detach from session (process keeps running)
# Press: Ctrl+B, then D

# Re-attach to session
tmux attach -t asr

# List sessions
tmux ls

# Kill a session
tmux kill-session -t asr
```

### 7.1 Download LLM Unlimited (primary model for transcription)

> ⚠️ **Important:** Inside a tmux session, run `conda deactivate` or `unset CONDA_PREFIX` first, otherwise fairseq2n may not find libsndfile!

```bash
tmux new -s asr-download
conda deactivate
source ~/projects/omnilingual-asr/.venv/bin/activate
python3 -c "
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipeline = ASRInferencePipeline(model_card='omniASR_LLM_Unlimited_7B_v2')
print('LLM Unlimited 7B v2 loaded!')
"
# Ctrl+B, D — to detach, download continues in background
```

Usage:
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_Unlimited_7B_v2")

# Basic transcription — no audio length limit
transcriptions = pipeline.transcribe(
    ["/path/to/audio.wav"],
    lang=["rus_Cyrl"],  # optional, improves quality
    batch_size=1,
)
print(transcriptions[0])
```

### 7.2 Download ZS Model (for new languages, requires context examples)

```bash
tmux new -s asr-zs
source ~/projects/omnilingual-asr/.venv/bin/activate
python3 -c "
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
pipeline = ASRInferencePipeline(model_card='omniASR_LLM_7B_ZS')
print('ZS 7B loaded!')
"
```

Usage:
```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline, ContextExample

pipeline = ASRInferencePipeline(model_card="omniASR_LLM_7B_ZS")

# Provide 1-10 examples (audio + text), model recognizes new audio
context_examples = [
    ContextExample("/path/to/example1.wav", "example text one"),
    ContextExample("/path/to/example2.wav", "example text two"),
]

transcriptions = pipeline.transcribe_with_context(
    ["/path/to/target_audio.wav"],
    context_examples=[context_examples],
    batch_size=1,
)
print(transcriptions[0])
```

---

## Available Models

| Model | Features | VRAM | Checkpoint Size |
|---|---|---|---|
| `omniASR_LLM_Unlimited_7B_v2` | Transcription, unlimited audio length, lang conditioning | ~17 GiB | 30 GiB |
| `omniASR_LLM_7B_v2` | Transcription, up to 40 sec, lang conditioning | ~17 GiB | 30 GiB |
| `omniASR_LLM_7B_ZS` | Zero-shot, requires context examples | ~20 GiB | 30 GiB |
| `omniASR_CTC_7B_v2` | Fast (16x), parallel decoding, no lang conditioning | ~15 GiB | 25 GiB |

## Hardware Specifications

| Parameter | Value |
|---|---|
| GPU | NVIDIA GB10 (Blackwell) |
| Architecture | aarch64 (ARM64) |
| CUDA Toolkit | 12.9 |
| Model cache | `~/.cache/fairseq2/assets/` |
