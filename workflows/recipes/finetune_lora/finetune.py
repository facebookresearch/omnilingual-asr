from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
import wandb
from datasets import Audio, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from fairseq2.datasets.batch import Seq2SeqBatch
from workflows.recipes.finetune_lora.lora import (
    LoraConfig,
    inject_lora,
)
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

from workflows.recipes.finetune_lora.utils import load_llm_asr


@dataclass
class TrainingConfig:
    # Model & Data
    model_name: str = "omniASR-LLM-300M"
    dataset_name: str = "UBC-NLP/Casablanca"
    dataset_configs: List[str] = field(default_factory=lambda: ["Egypt"])
    mapping_dialect_to_lang: Dict[str, str] = field(
        default_factory=lambda: {
            "UAE": "afb_Arab",
            "Egypt": "arz_Arab",
            "Algeria": "arq_Arab",
        }
    )
    default_lang_code: str = "afb_Arab"
    target_sampling_rate: int = 16_000

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_targets: Tuple[
        str, ...
    ] = (  # currently targetting all components of the model
        "llama_decoder",
        "encoder_frontend",
        "encoder",
        "encoder_proj",
        "final_proj",
        "lang_embeddings",
        "text_frontend",
    )

    # Training
    train_split: str = "validation"  # Original code used validation for train
    eval_split: str = "test"
    batch_size: int = 2
    num_epochs: int = 3
    learning_rate: float = 5e-5

    # Validation
    eval_every_steps: int = 35
    eval_max_batches: int = 50
    eval_log_examples: int = 3

    # Generation
    beam_nbest: int = 1
    beam_length_norm: bool = False

    # System
    num_workers_gpu: int = 2
    num_workers_cpu: int = 0
    pin_memory: bool = True
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )

    # Logging
    wandb_project: str = "finetune-omniASR-llm"
    run_suffix: str = "Egypt-only-clean"
    wer_metric_name: str = "wer"

    @property
    def checkpoint_path(self) -> str:
        sanitized_project = self.wandb_project.replace("/", "_")
        return f"best_model_valloss_{sanitized_project}.pt"

    def to_dict(self) -> Dict:
        return {
            k: str(v) if isinstance(v, torch.device) else v
            for k, v in dataclasses.asdict(self).items()
        }


class CasablancaOmniASRDataset(TorchDataset):
    """
    Non-streaming Casablanca dataset adapted for omniASR training.
    Combines multiple dataset configurations (dialects).
    """

    def __init__(self, config: TrainingConfig, split: str) -> None:
        self.config = config
        self.sampling_rate = config.target_sampling_rate

        ds_list = []
        for cfg_name in config.dataset_configs:
            try:
                ds_cfg = load_dataset(
                    config.dataset_name,
                    cfg_name,
                    split=split,
                    streaming=False,
                )
                # Tag examples with config and language code
                lang_code = config.mapping_dialect_to_lang.get(
                    cfg_name, config.default_lang_code
                )
                ds_cfg = ds_cfg.map(
                    lambda x, c=cfg_name, l=lang_code: {
                        "config_name": c,
                        "lang_code": l,
                    }
                )
                ds_list.append(ds_cfg)
            except Exception as e:
                print(
                    f"Warning: Could not load config {cfg_name} for split {split}: {e}"
                )

        if not ds_list:
            raise ValueError(f"No datasets loaded for {config.dataset_name}[{split}]")

        self.ds = concatenate_datasets(ds_list)
        self.ds = self.ds.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

        if len(self.ds) == 0:
            raise ValueError(f"Dataset is empty after loading.")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.ds[int(idx)]

        # Audio processing
        audio_obj = row["audio"]
        waveform = torch.as_tensor(audio_obj["array"])
        sample_rate = int(audio_obj["sampling_rate"])

        # Ensure mono
        if waveform.dim() == 2:
            waveform = (
                waveform.mean(dim=0) if waveform.size(0) > 1 else waveform.squeeze(0)
            )

        # Resample if needed
        if sample_rate != self.sampling_rate:
            waveform = torchaudio.functional.resample(
                waveform.unsqueeze(0), sample_rate, self.sampling_rate
            ).squeeze(0)

        return {
            "waveform": waveform,
            "transcription": row["transcription"],
            "lang_code": row.get("lang_code", self.config.default_lang_code),
        }


class OmniASRCollator:
    def __init__(self, tokenizer: object, device: torch.device, dtype: torch.dtype):
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.text_encoder = tokenizer.create_encoder()
        # Fallback pad index
        self.pad_idx = getattr(tokenizer.vocab_info, "pad_idx", 0)

    def __call__(self, batch: List[Dict[str, object]]) -> Seq2SeqBatch:
        # Audio
        waveforms = [s["waveform"] for s in batch]
        source_seq_lens = [w.size(0) for w in waveforms]
        max_src = max(source_seq_lens)

        padded_wav = torch.zeros(len(batch), max_src, dtype=waveforms[0].dtype)
        for i, w in enumerate(waveforms):
            padded_wav[i, : w.size(0)] = w

        source_seqs = padded_wav.to(self.device, self.dtype)

        # Text
        tokens = [self.text_encoder(s["transcription"]).to(self.device) for s in batch]
        target_seq_lens = [t.size(0) for t in tokens]
        max_tgt = max(target_seq_lens)

        target_seqs = torch.full(
            (len(batch), max_tgt),
            self.pad_idx,
            dtype=tokens[0].dtype,
            device=self.device,
        )
        for i, t in enumerate(tokens):
            target_seqs[i, : t.size(0)] = t

        return Seq2SeqBatch(
            source_seqs=source_seqs,
            source_seq_lens=source_seq_lens,  # List of ints
            target_seqs=target_seqs,
            target_seq_lens=target_seq_lens,  # List of ints
            example={"lang": [s["lang_code"] for s in batch]},
        )


def setup_model(config: TrainingConfig):
    print(f"Loading {config.model_name}...")
    model, tokenizer = load_llm_asr(lang_embeddings_p=0.5, device=config.device)

    # LoRA Injection
    lora_config = LoraConfig(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout_p=config.lora_dropout,
        target_keywords=config.lora_targets,
    )
    inject_lora(model, config=lora_config, freeze_base=True)

    # Param counting
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model loaded on {config.device}. Total params: {total_params:,}. Trainable (LoRA): {trainable_params:,}"
    )

    model.to(config.device)
    return model, tokenizer


class OmniTrainer:
    def __init__(
        self, config: TrainingConfig, model: torch.nn.Module, tokenizer: object
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.lowest_val_loss = float("inf")
        self.global_step = 0

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # Generator for validation
        self.beam_generator = None
        if isinstance(model, Wav2Vec2LlamaModel):
            self.beam_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
                model=model,
                config=Wav2Vec2LlamaBeamSearchConfig(
                    nbest=config.beam_nbest,
                    length_norm=config.beam_length_norm,
                ),
            )
        self.token_decoder = tokenizer.create_decoder(skip_special_tokens=True)

    def save_checkpoint(self, current_loss: float):
        if current_loss < self.lowest_val_loss:
            self.lowest_val_loss = current_loss
            dir_name = os.path.dirname(self.config.checkpoint_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            lora_state_dict = {
                k: v for k, v in self.model.state_dict().items() if "lora_" in k
            }

            # Save config so we can reconstruct the same architecture
            lora_config_dict = {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout_p": self.config.lora_dropout,
                "target_keywords": self.config.lora_targets,
            }

            torch.save(
                {"model_state_dict": lora_state_dict, "lora_config": lora_config_dict},
                self.config.checkpoint_path,
            )

            print(
                f"New best val_loss {current_loss:.4f}; LoRA adapters saved to {self.config.checkpoint_path}"
            )

    def validate(self, dataloader: DataLoader, log_examples: bool = False):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(dataloader, desc="Validating", leave=False)
            ):
                if batch_idx >= self.config.eval_max_batches:
                    break

                loss, _, _, decoder_context_inputs, decoder_context_layout = self.model(
                    batch, return_logits=True
                )
                total_loss += float(loss.item())
                num_batches += 1

                # Log examples
                if (
                    log_examples
                    and batch_idx < self.config.eval_log_examples
                    and self.beam_generator
                ):
                    self._log_example(
                        batch, decoder_context_inputs, decoder_context_layout, batch_idx
                    )

        avg_loss = total_loss / max(1, num_batches)
        print(f"Validation loss: {avg_loss:.4f}")
        wandb.log({"eval/val_loss": avg_loss, "eval/step": self.global_step})

        self.save_checkpoint(avg_loss)
        self.model.train()

    def _log_example(
        self, batch, decoder_context_inputs, decoder_context_layout, batch_idx
    ):
        rand_idx = torch.randint(0, batch.target_seqs.size(0), (1,)).item()

        pred_tokens, pred_layout = self.beam_generator.generate_hypotheses(
            decoder_context_inputs=decoder_context_inputs,
            decoder_context_input_layout=decoder_context_layout,
        )

        target_len = batch.target_seq_lens[rand_idx]
        target_text = self.token_decoder(batch.target_seqs[rand_idx, :target_len])

        pred_len = int(pred_layout.seq_lens_pt[rand_idx].item())
        pred_text = self.token_decoder(pred_tokens[rand_idx, :pred_len])

        print(f"  [Ex {batch_idx}] Ref: {target_text}")
        print(f"  [Ex {batch_idx}] Hyp: {pred_text}")

    def train_epoch(self, dataloader: DataLoader, eval_loader: DataLoader):
        self.model.train()
        progress_bar = tqdm(dataloader, desc=f"Training")

        total_loss = 0.0
        batches = 0

        for batch in progress_bar:
            self.optimizer.zero_grad()
            loss = self.model(batch)
            loss.backward()
            self.optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            batches += 1
            self.global_step += 1

            wandb.log({"train/loss": loss_val, "train/step": self.global_step})
            progress_bar.set_postfix({"loss": f"{loss_val:.4f}"})

            if self.global_step % self.config.eval_every_steps == 0:
                self.validate(eval_loader)

        return total_loss / max(1, batches)

    def train(self, train_loader: DataLoader, eval_loader: DataLoader):
        wandb.init(
            project=self.config.wandb_project,
            name=f"{self.config.model_name}-{self.config.run_suffix}",
            config=self.config.to_dict(),
        )

        for epoch in range(self.config.num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.config.num_epochs} ---")
            avg_loss = self.train_epoch(train_loader, eval_loader)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")


def main():
    config = TrainingConfig()

    # Setup Model
    model, tokenizer = setup_model(config)

    # Dtype inference for collator
    model_dtype = next(model.parameters()).dtype

    # Data Setup
    print("Preparing datasets...")
    train_ds = CasablancaOmniASRDataset(config, split=config.train_split)
    eval_ds = CasablancaOmniASRDataset(config, split=config.eval_split)

    collator = OmniASRCollator(tokenizer, config.device, model_dtype)

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers_gpu
        if torch.cuda.is_available()
        else config.num_workers_cpu,
        "pin_memory": config.pin_memory,
        "collate_fn": collator,
    }

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    eval_loader = DataLoader(eval_ds, shuffle=True, **loader_kwargs)

    # Trainer
    trainer = OmniTrainer(config, model, tokenizer)
    trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    main()
