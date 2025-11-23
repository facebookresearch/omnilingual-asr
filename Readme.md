# Omnilingual ASR

Omnilingual ASR is an open-source multilingual automatic speech recognition (ASR) system, supporting over 1600 languages. It is built on top of Fairseq2 and provides pipelines for training, inference, and evaluation.

## Installation (CUDA-compatible setup)

To avoid CUDA incompatibility issues (issue #26), follow these steps:

1. **Install system dependency**  
   Omnilingual ASR requires `libsndfile1` for audio processing:

```bash
sudo apt install libsndfile1
```

2. **Install compatible Fairseq2**  
   Make sure to match PyTorch + CUDA versions:

```bash
pip install fairseq2==0.6 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu126
```

> This ensures PyTorch 2.8.0 + CUDA 12.6 compatibility and avoids runtime errors.

3. **Install Omnilingual ASR**

```bash
pip install omnilingual-asr
```

4. **Optional: Verify installation**

```bash
python -c "from omnilingual_asr import ASRInferencePipeline; print('Omnilingual ASR loaded successfully')"
```

5. **Optional: Runtime CUDA check**  
   Add this snippet in your code to warn if CUDA version is incompatible:

```python
import torch

if torch.version.cuda != "12.6":
    print("Warning: Omnilingual ASR was tested on CUDA 12.6 with PyTorch 2.8.0")
```

## Usage

After installation, you can import and use the `ASRInferencePipeline` for speech recognition tasks:

```python
from omnilingual_asr import ASRInferencePipeline

pipeline = ASRInferencePipeline()
text = pipeline.transcribe("your_audio_file.wav")
print(text)
```

## Contributing

We welcome contributions! If you want to contribute, please follow the GitHub workflow: fork the repo, create a branch, make your changes, and open a Pull Request.

Refer to issue #26 for CUDA compatibility fixes and best practices.

## License

This project is licensed under the MIT License.
