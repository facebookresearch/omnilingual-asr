from __future__ import annotations

from typing import Dict, List

import os
import random
import torch
import torchaudio
import wandb
from datasets import Audio, load_dataset, concatenate_datasets
from evaluate import load as load_metric
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from omnilingual_asr.models.lora import LoraConfig, inject_lora, lora_parameters

from fairseq2.datasets.batch import Seq2SeqBatch
from datetime import datetime

from load_and_compute_loss import (
    load_llm_asr_300m,
)
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
from omnilingual_asr.models.lora import LoRALinear

DEFAULT_LANG_CODE = "afb_Arab"
MODEL_NAME = "omniASR-LLM-300M"

# Dataset / audio configs.
DATASET_NAME = "UBC-NLP/Casablanca"
DATASET_CONFIG_NAME = "UAE"
DATASET_CONFIG_NAMES = ["UAE", "Egypt", "Algeria"]  # your configs
TARGET_SAMPLING_RATE = 16_000

MAPPING_DIALECT_TO_LANG_CODE = {
    "UAE": "afb_Arab",
    "Egypt": "arz_Arab",
    "Algeria": "arq_Arab",
}

# LoRA configs.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT_P = 0.1
LORA_TARGET_KEYWORDS = ("llama_decoder.layers",)

# Training loop configs.
TRAIN_SPLIT = "validation"
EVAL_SPLIT = "test"
BATCH_SIZE = 2
NUM_EPOCHS = 3
# STEPS_PER_EPOCH = 100
EVAL_EVERY_STEPS = 35
LEARNING_RATE = 5e-5

# DataLoader configs.
NUM_WORKERS_GPU = 2
NUM_WORKERS_CPU = 0
PIN_MEMORY = True

# Validation / decoding configs.
EVAL_MAX_BATCHES = 50
EVAL_LOG_EXAMPLES = 3
BEAM_NBEST = 1
BEAM_LENGTH_NORM = False

# WandB / metrics configs.
WANDB_PROJECT = "finetune-omniASR-llm"
RUN_NAME_DATASET_SUFFIX = "UBC-NLP_Casablanca"
WER_METRIC_NAME = "wer"

# Checkpoint / artifact configs.
SANITIZED_WANDB_PROJECT = WANDB_PROJECT.replace("/", "_")
DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
BEST_VAL_LOSS_CHECKPOINT_PATH = f"best_model_valloss_{DATE}_{SANITIZED_WANDB_PROJECT}.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aggregated configuration for logging to Weights & Biases.
WANDB_CONFIG = {
    "model_name": MODEL_NAME,
    "default_lang_code": DEFAULT_LANG_CODE,
    "dataset_name": DATASET_NAME,
    "dataset_config_name": DATASET_CONFIG_NAME,
    "target_sampling_rate": TARGET_SAMPLING_RATE,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout_p": LORA_DROPOUT_P,
    "lora_target_keywords": LORA_TARGET_KEYWORDS,
    "train_split": TRAIN_SPLIT,
    "eval_split": EVAL_SPLIT,
    "batch_size": BATCH_SIZE,
    "num_epochs": NUM_EPOCHS,
    # "steps_per_epoch": STEPS_PER_EPOCH,
    "eval_every_steps": EVAL_EVERY_STEPS,
    "learning_rate": LEARNING_RATE,
    "num_workers_gpu": NUM_WORKERS_GPU,
    "num_workers_cpu": NUM_WORKERS_CPU,
    "pin_memory": PIN_MEMORY,
    "eval_max_batches": EVAL_MAX_BATCHES,
    "eval_log_examples": EVAL_LOG_EXAMPLES,
    "beam_nbest": BEAM_NBEST,
    "beam_length_norm": BEAM_LENGTH_NORM,
    "wandb_project": WANDB_PROJECT,
    "run_name_dataset_suffix": RUN_NAME_DATASET_SUFFIX,
    "wer_metric_name": WER_METRIC_NAME,
    "device": str(DEVICE),
}

# For metrics; the omnilingual model computes loss internally when called as model(batch).
wer_metric = load_metric(WER_METRIC_NAME)

class CasablancaOmniASRDataset(TorchDataset):
    """
    Non-streaming Casablanca dataset adapted for omniASR training.

    Returns dicts with:
      - 'waveform': torch.Tensor (T,)
      - 'transcription': str
    """

    def __init__(
        self,
        split: str,
        sampling_rate: int = TARGET_SAMPLING_RATE,
    ) -> None:
        

        ds_list = []
        for cfg_name in DATASET_CONFIG_NAMES:
            ds_cfg = load_dataset(
                DATASET_NAME,
                cfg_name,
                split=split,        # keep whatever split logic you already have
                streaming=False,
            )
            # add a column identifying which config it came from
            ds_cfg = ds_cfg.map(lambda x, cfg_name=cfg_name: {"config_name": cfg_name, "lang_code": MAPPING_DIALECT_TO_LANG_CODE[cfg_name]})
            ds_list.append(ds_cfg)

        ds = concatenate_datasets(ds_list)
        # ds = load_dataset(
        #     DATASET_NAME,
        #     DATASET_CONFIG_NAME,
        #     split=split,
        #     streaming=False,
        # )
        ds = ds.cast_column("audio", Audio(sampling_rate=sampling_rate))

        if len(ds) == 0:
            raise ValueError(
                f"Hugging Face dataset '{DATASET_NAME}'[{split}] is empty."
            )

        self.ds = ds
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, object]:
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
                "Expected each 'audio' entry to be a dict with 'array' and 'sampling_rate' fields."
            )

        waveform = torch.as_tensor(audio_obj["array"])
        sample_rate = int(audio_obj["sampling_rate"])

        # Ensure mono waveform (T,)
        if waveform.dim() == 2:
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

        # Resample if needed
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0),
                sample_rate,
                self.sampling_rate,
            ).squeeze(0)

        transcription: str = row["transcription"]

        lang_code = row["lang_code"]

        return {
            "waveform": waveform,
            "transcription": transcription,
            "lang_code": lang_code,
        }


def collate_to_seq2seq_batch(
    batch: List[Dict[str, object]],
    *,
    tokenizer: object,
    lang_code: str | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Seq2SeqBatch:
    """
    Collate a list of samples into a `Seq2SeqBatch` expected by `Wav2Vec2LlamaModel`.
    """
    # Audio: pad to max length in batch
    waveforms = [sample["waveform"] for sample in batch]
    assert all(isinstance(w, torch.Tensor) and w.dim() == 1 for w in waveforms)

    source_seq_lens = [w.size(0) for w in waveforms]
    max_len = max(source_seq_lens)

    padded_waveforms = torch.zeros(len(waveforms), max_len, dtype=waveforms[0].dtype)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, : w.size(0)] = w

    # Match model dtype to avoid "Input type (double) and bias type (float)" errors.
    source_seqs = padded_waveforms.to(device=device, dtype=dtype)

    # Text tokenization
    text_encoder = tokenizer.create_encoder()
    token_tensors: List[torch.Tensor] = []
    target_seq_lens: List[int] = []

    for sample in batch:
        text = sample["transcription"]
        tok = text_encoder(text)
        tok = tok.to(device)
        token_tensors.append(tok)
        target_seq_lens.append(int(tok.size(0)))

    max_t_len = max(target_seq_lens)

    # omni tokenizer exposes pad_idx via its vocab info; fall back to 0 if absent.
    pad_idx = tokenizer.vocab_info.pad_idx
    target_seqs = torch.full(
        (len(batch), max_t_len),
        fill_value=pad_idx,
        dtype=token_tensors[0].dtype,
        device=device,
    )

    for i, tok in enumerate(token_tensors):
        target_seqs[i, : tok.size(0)] = tok

    example: Dict[str, object] = {}
    
    example["lang"] = [sample["lang_code"] for sample in batch]

    return Seq2SeqBatch(
        source_seqs=source_seqs,
        source_seq_lens=source_seq_lens,
        target_seqs=target_seqs,
        target_seq_lens=target_seq_lens,
        example=example,
    )


def evaluate_loss_only(
    model: torch.nn.Module,
    data_loader: DataLoader,
    tokenizer: object,
) -> float:
    """
    Simple validation: average training loss on `data_loader`.
    The omni model itself computes the loss when called as `model(batch)`.
    """
    model.eval()
    # Create a decoder once so we can inspect / decode targets during validation.
    token_decoder = tokenizer.create_decoder(skip_special_tokens=True)
    # Set up a simple beam-search generator if this is the LLM ASR model.
    beam_search_generator = None
    if isinstance(model, Wav2Vec2LlamaModel):
        beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
            nbest=BEAM_NBEST,
            length_norm=BEAM_LENGTH_NORM,
        )
        beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
            model=model,
            config=beam_search_config,
        )

    total_loss = 0.0
    num_batches = EVAL_MAX_BATCHES

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Validating")):
            if batch_idx > EVAL_MAX_BATCHES:
                break

            target_seqs = batch.target_seqs
            loss, logits, decoder_inputs_layout, decoder_context_inputs, decoder_context_layout = model(batch, return_logits=True)
            if batch_idx < EVAL_LOG_EXAMPLES:
                # Pick a random example from the batch to decode.
                batch_size = target_seqs.size(0)
                rand_idx = torch.randint(
                    low=0,
                    high=batch_size,
                    size=(1,),
                    device=target_seqs.device,
                ).item()

                pred_tokens, pred_layout = beam_search_generator.generate_hypotheses(
                    decoder_context_inputs=decoder_context_inputs,
                    decoder_context_input_layout=decoder_context_layout,
                )

                # Use the recorded sequence lengths to slice out valid tokens.
                target_len = batch.target_seq_lens[rand_idx]
                decoded_text = token_decoder(
                    target_seqs[rand_idx, :target_len]
                )

                pred_len = int(pred_layout.seq_lens_pt[rand_idx].item())
                pred_text = token_decoder(
                    pred_tokens[rand_idx, :pred_len]
                )

                print(f"    [example {rand_idx}, ] target text (groundtruth transcription): {decoded_text}")
                print(f"    [example {rand_idx}, ] predicted text (beam search): {pred_text}")
            total_loss += float(loss.item())
            num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    print(f"Validation loss: {avg_loss:.4f}")
    wandb.log({"eval/val_loss": avg_loss})
    # Save checkpoint if this is the best validation loss so far.
    save_lora_checkpoint(model, avg_loss)

    return avg_loss

def save_lora_checkpoint(
    model: torch.nn.Module,
    avg_val_loss: float,
) -> None:
    """
    Save a LoRA checkpoint when a new best validation loss is observed.
    """
    global lowest_val_loss

    if "lowest_val_loss" not in globals():
        lowest_val_loss = float("inf")

    if avg_val_loss >= lowest_val_loss:
        return

    lowest_val_loss = avg_val_loss

    checkpoint_path_str = BEST_VAL_LOSS_CHECKPOINT_PATH
    dir_name = os.path.dirname(checkpoint_path_str)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path_str)
    print(
        f"New best val_loss {avg_val_loss:.4f}; model saved to {checkpoint_path_str}"
    )

if __name__ == "__main__":
    # Load omniASR model and tokenizer from local checkpoint as defined in load_and_compute_loss.py
    model, tokenizer = load_llm_asr_300m(lang_embeddings_p=0.5, device=DEVICE)

    # Freeze the entire model first to prevent catastrophic forgetting
    # of non-LoRA components (like the encoder, embeddings, etc.)
    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout_p=LORA_DROPOUT_P,
        target_keywords=LORA_TARGET_KEYWORDS,
    )
    inject_lora(model, config=lora_config, freeze_base=True)

    lora_param_count = sum(p.numel() for p in lora_parameters(model))
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    

    with open("lora_params.txt", "w") as f:
        f.write("=== LoRA Fine-Tuning Parameter Analysis ===\n\n")
        f.write(f"Total Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Total Trainable Parameters: {trainable_params:,}\n")
        f.write(f"LoRA-only Parameters: {lora_param_count:,}\n\n")
        
        f.write("--- Module-wise Breakdown ---\n")
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                a_params = sum(p.numel() for p in module.lora_A.parameters())
                b_params = sum(p.numel() for p in module.lora_B.parameters())
                print(f"{name}: LoRA params = {a_params + b_params:,}")
                f.write(f"{name}: LoRA params = {a_params + b_params:,}\n")
        
        f.write("\n--- Detailed Parameter Status (Trainable vs Frozen) ---\n")
        for name, param in model.named_parameters():
             status = "TRAINABLE" if param.requires_grad else "FROZEN"
             f.write(f"{name}: {status} | Shape: {tuple(param.shape)} | Numel: {param.numel():,}\n")
            
    print(f"LoRA-only params: {lora_param_count:,}")
    print(f"Total trainable params: {trainable_params:,}")

    if trainable_params != lora_param_count:
        print("WARNING: Trainable params != LoRA params. Some base model weights are not frozen!")

    model.to(device=DEVICE)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded omniASR_LLM_300M on {DEVICE} "
        f"({total_params:,} parameters)."
    )

    # Datasets: Casablanca HF splits, adapted for omniASR.
    train_split = TRAIN_SPLIT
    eval_split = EVAL_SPLIT

    train_dataset = CasablancaOmniASRDataset(
        split=train_split,
        sampling_rate=TARGET_SAMPLING_RATE,
    )
    eval_dataset = CasablancaOmniASRDataset(
        split=eval_split,
        sampling_rate=TARGET_SAMPLING_RATE,
    )

    batch_size = BATCH_SIZE

    # Infer the dtype used by the model parameters so inputs match.
    first_param = next(model.parameters())
    model_dtype = first_param.dtype

    def collate_fn(batch: List[Dict[str, object]]) -> Seq2SeqBatch:
        return collate_to_seq2seq_batch(
            batch,
            tokenizer=tokenizer,
            lang_code=DEFAULT_LANG_CODE,
            device=DEVICE,
            dtype=model_dtype,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS_GPU if torch.cuda.is_available() else NUM_WORKERS_CPU,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS_GPU if torch.cuda.is_available() else NUM_WORKERS_CPU,
        pin_memory=PIN_MEMORY,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    num_epochs = NUM_EPOCHS
    # steps_per_epoch = STEPS_PER_EPOCH  # number of training batches per epoch
    eval_every_steps = EVAL_EVERY_STEPS

    run_name = f"{MODEL_NAME}-{RUN_NAME_DATASET_SUFFIX}"
    wandb.init(project=WANDB_PROJECT, name=run_name, config=WANDB_CONFIG)

    global_step = 0

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            train_loader,
            # total=steps_per_epoch,
            desc=f"Epoch {epoch+1}/{num_epochs} (omniASR finetune)",
        )

        for step, batch in enumerate(progress_bar):
            # if step >= steps_per_epoch:
            #     break

            optimizer.zero_grad()

            # The omni model returns a loss tensor when called with a Seq2SeqBatch.
            loss = model(batch)
            loss.backward()
            optimizer.step()

            loss_item = float(loss.item())
            wandb.log({"train/loss": loss_item, "train/step": global_step})
            progress_bar.set_postfix({"batch_loss": loss_item})

            total_loss += loss_item
            num_batches += 1
            global_step += 1

            if global_step % eval_every_steps == 0:
                evaluate_loss_only(model, eval_loader, tokenizer)
                model.train()

        epoch_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")


