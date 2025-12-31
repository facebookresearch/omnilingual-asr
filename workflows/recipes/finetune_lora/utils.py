from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver
from omnilingual_asr.models.wav2vec2_llama import (
    Wav2Vec2LlamaConfig,
    convert_wav2vec2_llama_state_dict,
    create_wav2vec2_llama_model,
)


def load_llm_asr(
    checkpoint_path: Path | str | None = None,
    *,
    model_size: str = "300m",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    lang_embeddings_p: float | None = None,
) -> Tuple[torch.nn.Module, object]:
    """
    Load an omniASR LLM-ASR model (wav2vec2_llama) from a local checkpoint or URL.

    Args:
        checkpoint_path: Absolute path (or URL) to the .pt file to load. If None,
            defaults to the official checkpoint URL for `model_size`.
        model_size: Model size identifier. One of: "300m", "1b", "3b", "7b".
        device: Optional device override (e.g., "cuda", "cpu").
        dtype: Optional dtype override (defaults to bfloat16 on CUDA, fp32 otherwise).

    Returns:
        A tuple of (model, tokenizer).
    """

    model_size = model_size.lower().strip()
    supported_model_sizes = {"300m", "1b", "3b", "7b"}
    if model_size not in supported_model_sizes:
        raise ValueError(
            f"Unsupported model_size={model_size!r}. Expected one of {sorted(supported_model_sizes)}."
        )

    if checkpoint_path is None:
        checkpoint_path = {
            "300m": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-300M.pt",
            "1b": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-1B.pt",
            "3b": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt",
            "7b": "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-7B.pt",
        }[model_size]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Instantiate the wav2vec2-LLM ASR model and load weights from a local checkpoint.
    resolver = get_dependency_resolver()
    config = get_config(resolver, Wav2Vec2LlamaConfig, model_size)

    if lang_embeddings_p is not None:
        config.lang_embeddings_p = lang_embeddings_p

    # Create an uninitialized model instance on CPU.
    model = create_wav2vec2_llama_model(config)

    # Support loading from URL
    if str(checkpoint_path).startswith("http://") or str(checkpoint_path).startswith(
        "https://"
    ):
        print(f"Downloading checkpoint from {checkpoint_path}...")
        checkpoint = torch.hub.load_state_dict_from_url(
            str(checkpoint_path), map_location=device, check_hash=False
        )
    else:
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. Update CHECKPOINT_PATH first."
            )
        # Load checkpoint and extract raw state dict.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict") or checkpoint.get("model") or checkpoint
        )
    else:
        state_dict = checkpoint

    # Convert checkpoint weights, then load into the model.
    converted_state_dict = convert_wav2vec2_llama_state_dict(state_dict, config)
    model.load_state_dict(converted_state_dict)

    model.to(device=device, dtype=dtype)

    tokenizer = load_tokenizer("omniASR_tokenizer")

    model.eval()
    return model, tokenizer
