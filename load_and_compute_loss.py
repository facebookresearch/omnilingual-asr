from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torchaudio
from datasets import Audio, Dataset, load_dataset
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.runtime.config_registry import get_config
from fairseq2.runtime.dependency import get_dependency_resolver
from torch.utils.data import Dataset as TorchDataset
from omnilingual_asr.models.wav2vec2_llama import (
    Wav2Vec2LlamaConfig,
    convert_wav2vec2_llama_state_dict,
    create_wav2vec2_llama_model,
)
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.model import (
    Wav2Vec2LlamaBeamSearchConfig,
    Wav2Vec2LlamaModel,
)

# Update this to the absolute path of the checkpoint you want to load.
CHECKPOINT_PATH = Path("model_weights/omniASR-LLM-300M.pt")
PARQUET_PATH = Path("casablanca_uae.parquet")

# Default language code for LID model inference.
# Change this to match your dataset language (see
# `omnilingual_asr/models/wav2vec2_llama/lang_ids.py` for supported codes).
DEFAULT_LANG_CODE = "afb_Arab"


def load_llm_asr_300m(
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    lang_embeddings_p: float | None = None,
) -> Tuple[torch.nn.Module, object]:
    """
    Load the 300M LLM-ASR model (wav2vec2_llama) from a local checkpoint.

    Args:
        checkpoint_path: Absolute path to the .pt file to load.
        device: Optional device override (e.g., "cuda", "cpu").
        dtype: Optional dtype override (defaults to bfloat16 on CUDA, fp32 otherwise).

    Returns:
        A tuple of (model, tokenizer).
    """

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Update CHECKPOINT_PATH first."
        )

    # Choose sensible defaults

    device = torch.device(device)

    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Instantiate the wav2vec2-LLM ASR model and load weights from a local checkpoint.
    resolver = get_dependency_resolver()
    config = get_config(resolver, Wav2Vec2LlamaConfig, "300m")

    if lang_embeddings_p is not None:
        config.lang_embeddings_p = lang_embeddings_p

    # Create an uninitialized model instance on CPU.
    model = create_wav2vec2_llama_model(config)

    # Load checkpoint and extract raw state dict.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get(
            "model"
        ) or checkpoint
    else:
        state_dict = checkpoint

    # Convert checkpoint weights, then load into the model.
    converted_state_dict = convert_wav2vec2_llama_state_dict(state_dict, config)
    model.load_state_dict(converted_state_dict)

    # Move to requested device / dtype.
    model.to(device=device, dtype=dtype)

    tokenizer = load_tokenizer("omniASR_tokenizer")

    model.eval()
    return model, tokenizer


def compute_batch_loss(model: torch.nn.Module, batch: Seq2SeqBatch) -> torch.Tensor:
    """Compute loss for a prepared Seq2SeqBatch (no grad)."""
    model.eval()
    with torch.no_grad():
        loss = model(batch)
    return loss


def run_single_inference_from_parquet(
    model: torch.nn.Module,
    tokenizer: object,
    parquet_path: Path | str = PARQUET_PATH,
    *,
    row_idx: int = 0,
    lang_code: str | None = DEFAULT_LANG_CODE,
) -> tuple[str, str]:
    """
    Load a single audio example from a Parquet dataset and run LID ASR inference.

    Assumes the Parquet file has an `audio` column where each entry is a dict with
    an `array` field (raw waveform samples) and a `sampling_rate` field.
    """
    parquet_path = Path(parquet_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {parquet_path}")

    # Load dataset and select a single row.
    ds = Dataset.from_parquet(str(parquet_path))
    if len(ds) == 0:
        raise ValueError(f"Dataset at {parquet_path} is empty.")
    if not (0 <= row_idx < len(ds)):
        raise IndexError(
            f"row_idx {row_idx} out of range for dataset of length {len(ds)}."
        )

    row = ds[int(row_idx)]
    if "audio" not in row:
        raise KeyError("Expected an 'audio' column in the dataset.")
    if "transcription" not in row:
        raise KeyError("Expected a 'transcription' column in the dataset.")

    audio_obj = row["audio"]
    if not isinstance(audio_obj, dict) or "array" not in audio_obj or "sampling_rate" not in audio_obj:
        raise ValueError(
            "Expected each 'audio' entry to be a dict with 'array' and 'sampling_rate' "
            "fields."
        )

    # array is typically a NumPy array (time,) or (channels, time)
    waveform = torch.as_tensor(audio_obj["array"])
    sample_rate = int(audio_obj["sampling_rate"])

    # Infer device and dtype from the loaded model.
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    # Ensure shape is (time,) and convert to mono if needed.
    if waveform.dim() == 2:
        # (channels, time) -> mono (time,)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sample_rate, target_sample_rate
        ).squeeze(0)

    # Move to model device / dtype.
    waveform = waveform.to(device=device, dtype=dtype)

    # Build a minimal Seq2SeqBatch for a single example.
    source_seqs = waveform.unsqueeze(0)  # (1, T)
    source_seq_lens = [source_seqs.size(1)]

    # Dummy target tokens (needed for model.forward, but not for inference loss).
    target_seqs = torch.zeros((1, 1), dtype=torch.long, device=device)
    target_seq_lens = [1]

    example = {}
    if lang_code is not None:
        example["lang"] = [lang_code]

    batch = Seq2SeqBatch(
        source_seqs=source_seqs,
        source_seq_lens=source_seq_lens,
        target_seqs=target_seqs,
        target_seq_lens=target_seq_lens,
        example=example,
    )

    # Run a single forward pass to obtain decoder inputs for generation.
    assert isinstance(model, Wav2Vec2LlamaModel)

    beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
        nbest=1,
        length_norm=False,
    )
    generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
        model=model,
        config=beam_search_config,
    )

    with torch.inference_mode():
        decoder_context, decoder_context_layout = model(
            batch, return_decoder_inputs=True
        )

        hypothesis_tokens, hypothesis_layout = generator.generate_hypotheses(
            decoder_context_inputs=decoder_context,
            decoder_context_input_layout=decoder_context_layout,
        )

    # Decode token IDs to text using the tokenizer.
    token_decoder = tokenizer.create_decoder(skip_special_tokens=True)
    seq_len = hypothesis_layout.seq_lens[0]
    tokens = hypothesis_tokens[0, :seq_len]
    transcription = token_decoder(tokens)
    ground_truth = row["transcription"]

    return transcription, ground_truth


class ParquetOmniASRDataset(TorchDataset):
    """
    Simple Dataset that wraps a Parquet ASR dataset and performs audio/text
    preprocessing suitable for Wav2Vec2LlamaModel training.

    Each item returns a dict containing:
        - source_seqs, source_seq_lens
        - target_seqs, target_seq_lens
        - example (with optional `lang` field)
        - transcription (original text, for convenience)
    """

    def __init__(
        self,
        parquet_path: Path | str,
        tokenizer: object,
        *,
        lang_code: str | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """
        If `parquet_path` points to an existing local file, we load from Parquet.
        Otherwise, we interpret `parquet_path` as a Hugging Face dataset name and
        load it via `load_dataset`, casting the `audio` column with `Audio(...)`
        as done in `finetune_omni_300.py:create_streaming_datasets`.
        """
        # maybe_path = Path(parquet_path).expanduser()
        # if maybe_path.exists():
        #     self.parquet_path = maybe_path.resolve()
        #     self.ds = Dataset.from_parquet(str(self.parquet_path))
        #     if len(self.ds) == 0:
        #         raise ValueError(f"Dataset at {self.parquet_path} is empty.")
        # else:
        dataset_name = "UBC-NLP/Casablanca"
        # Defaults mirror `create_streaming_datasets` in `finetune_omni_300.py`.
        config_name = "UAE"
        split = "validation"
        sampling_rate = 16000

        self.ds = load_dataset(
            dataset_name,
            config_name,
            split=split,
            streaming=False,
        )
        # Ensure we get dicts with `array` and `sampling_rate` in the `audio` col.
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=sampling_rate))
        if len(self.ds) == 0:
            raise ValueError(
                f"Hugging Face dataset '{dataset_name}'[{split}] is empty."
            )

        self.tokenizer = tokenizer
        self.lang_code = lang_code
        self.device = device
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.ds[int(idx)]

        if "audio" not in row:
            raise KeyError("Expected an 'audio' column in the dataset.")
        if "transcription" not in row:
            raise KeyError("Expected a 'transcription' column in the dataset.")

        audio_obj = row["audio"]
        if (
            not isinstance(audio_obj, dict)
            or "array" not in audio_obj
            or "sampling_rate" not in audio_obj
        ):
            raise ValueError(
                "Expected each 'audio' entry to be a dict with 'array' and 'sampling_rate' "
                "fields."
            )

        waveform = torch.as_tensor(audio_obj["array"])
        sample_rate = int(audio_obj["sampling_rate"])

        # Ensure shape is (time,) and convert to mono if needed.
        if waveform.dim() == 2:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        target_sample_rate = 16000
        if sample_rate != target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, target_sample_rate
            ).squeeze(0)

        waveform = waveform.to(device=self.device, dtype=self.dtype)

        # Build `Seq2SeqBatch`-compatible fields.
        source_seqs = waveform.unsqueeze(0)  # (1, T)
        source_seq_lens = [source_seqs.size(1)]

        # Tokenize ground-truth transcription.
        text_encoder = self.tokenizer.create_encoder()
        target = text_encoder(row["transcription"]).to(self.device)
        target_seqs = target.unsqueeze(0)  # (1, S)
        target_seq_lens = [target.size(0)]

        example: dict = {}
        if self.lang_code is not None:
            example["lang"] = [self.lang_code]

        return {
            "source_seqs": source_seqs,
            "source_seq_lens": source_seq_lens,
            "target_seqs": target_seqs,
            "target_seq_lens": target_seq_lens,
            "example": example,
            "transcription": row["transcription"],
        }


def compute_single_example_loss_from_parquet(
    model: torch.nn.Module,
    tokenizer: object,
    parquet_path: Path | str = PARQUET_PATH,
    *,
    row_idx: int = 0,
    lang_code: str | None = DEFAULT_LANG_CODE,
) -> torch.Tensor:
    """
    Build a training-style batch from a single Parquet row and return the loss.

    Uses the ground-truth `transcription` field as the decoder target instead of
    a dummy sequence and runs `model(batch)` to obtain the training loss.
    """
    # Infer device and dtype from the loaded model and delegate preprocessing
    # to the Dataset class.
    first_param = next(model.parameters())
    device = first_param.device
    dtype = first_param.dtype

    ds = ParquetOmniASRDataset(
        parquet_path=parquet_path,
        tokenizer=tokenizer,
        lang_code=lang_code,
        device=device,
        dtype=dtype,
    )

    if not (0 <= row_idx < len(ds)):
        raise IndexError(
            f"row_idx {row_idx} out of range for dataset of length {len(ds)}."
        )

    sample = ds[row_idx]
    batch = Seq2SeqBatch(
        source_seqs=sample["source_seqs"],
        source_seq_lens=sample["source_seq_lens"],
        target_seqs=sample["target_seqs"],
        target_seq_lens=sample["target_seq_lens"],
        example=sample["example"],
    )

    # Standard training-style forward: returns loss tensor.
    loss = model(batch)
    return loss


if __name__ == "__main__":
    model, tokenizer = load_llm_asr_300m()

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded omniASR_LLM_300M (LID) from {CHECKPOINT_PATH} "
        f"({total_params:,} parameters)."
    )

    # Example: compute loss for the same example using a training-style forward pass.
    loss = compute_single_example_loss_from_parquet(
        model,
        tokenizer,
        parquet_path=PARQUET_PATH,
        row_idx=0,
        lang_code=DEFAULT_LANG_CODE,
    )
    print(f"\nTraining-style loss for row 0: {loss.item():.4f}")

