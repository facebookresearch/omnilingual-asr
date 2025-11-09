# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict, List, Tuple, final

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq2.data.tokenizers import VocabularyInfo
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.device import Device
from fairseq2.logging import get_log_writer
from fairseq2.models.asr import AsrModel
from fairseq2.models.transformer import TransformerEncoder
from fairseq2.models.transformer_lm import TransformerLMDecoder
from fairseq2.models.wav2vec2 import Wav2Vec2Frontend, Wav2Vec2Masker
from fairseq2.nn import BatchLayout, StandardEmbedding
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from omnilingual_asr.models.wav2vec2_llama.config import (
    ModelType,
    Wav2Vec2LlamaBeamSearchConfig,
)

log = get_log_writer(__name__)


@final
class Wav2Vec2LlamaModel(AsrModel):
    """Represents a wav2vec 2.0 encoder feeding to a Llama decoder for ASR."""

    def __init__(
        self,
        model_type: ModelType,
        model_dim: int,
        encoder_frontend: Wav2Vec2Frontend,
        encoder: TransformerEncoder,
        encoder_proj: nn.Module,
        text_frontend: StandardEmbedding,
        llama_decoder: TransformerLMDecoder,
        final_proj: nn.Module,
        target_vocab_info: VocabularyInfo,
        *,
        masker: Wav2Vec2Masker | None = None,
        max_generation_length: int = 8192,
        encoder_stacking: int = 1,
        lang_embeddings_p: float = 0.0,
        language_column_name: str = "lang",
        lang_embeddings: StandardEmbedding | None = None,
        lang_mapping: dict[str, int] | None = None,
        context_text_only: bool = False,
        beam_search_config: Wav2Vec2LlamaBeamSearchConfig = Wav2Vec2LlamaBeamSearchConfig(),
        n_context_examples: int = 0,
    ) -> None:
        """
        :param model_type:
            The high-level model variant (standard LLM-ASR / LLM-ASR with LID / zero-shot model).
        :param model_dim:
            Model dimension of the transformer decoder.
        :param encoder_frontend:
            The w2v2 encoder frontend.
        :param encoder:
            The w2v2 encoder.
        :param encoder_proj:
            A projection layer projecting the encoder outputs to the decoder's model dim.
        :text_frontend:
            The embedding module for text tokens.
        :param llama_decoder:
            The decoder-only model.
        :param final_proj:
            The last linear layer(s) projecting from the decoder to logits.
        :param target_vocab_info:
            The vocabulary information (size, special token IDS, etc).
        :param masker:
            The w2v2 feature masker.
        :param max_generation_length:
            The maximum length of training or generated sequences in the decoder model.
        :param encoder_stacking:
            The number audio embeddings frames to stack before the decoder calls.
        :param lang_embeddings_p:
            For the LID model, the probability of dropping the language embeddings.
        :param language_column_name:
            For the LID model, the name of the column containing the language information.
        :param beam_search_config:
            The beam search configuration.
        :param n_context_examples:
            For the zero-shot model, the number of context examples to use for zero-shot inference.
        """

        super().__init__()

        self.model_type = model_type
        self.model_dim = model_dim
        self.encoder_frontend = encoder_frontend
        self.encoder = encoder
        self.encoder_proj = encoder_proj
        self.text_frontend = text_frontend
        self.llama_decoder = llama_decoder
        self.final_proj = final_proj
        self.target_vocab_info = target_vocab_info
        self.max_generation_length = max_generation_length  # move to beamsearch config
        self.encoder_stacking = encoder_stacking
        self.lang_embeddings_p = lang_embeddings_p
        self.lang_embeddings = lang_embeddings
        self.lang_mapping = lang_mapping
        self.rng = torch.Generator()
        self.rng.manual_seed(42)
        self.audio_encoder_calls = 0
        self.language_column_name = language_column_name
        self.context_text_only = context_text_only
        self.beam_search_config = beam_search_config
        self.n_context_examples = n_context_examples

        self.register_module("masker", masker)

        assert self.target_vocab_info.pad_idx is not None
        assert self.target_vocab_info.eos_idx is not None
        assert self.target_vocab_info.bos_idx is not None

    def forward(  # type: ignore
        self,
        batch: Seq2SeqBatch,
        return_logits: bool = False,
        return_decoder_inputs: bool = False,
    ) -> (
        Tensor
        | Tuple[Tensor, BatchLayout]
        | Tuple[Tensor, Tensor, BatchLayout, Tensor, BatchLayout]
    ):
        """Model entry point, accepts a `Seq2Seq` batch and embeds the inputs w.r.t.
        their modality:
        - audio is embedded with encoder_frontend, encoder, final_proj
        - text is embedded with text_frontend (simple embedding)
        - lang tokens are embedded with lang_frontend (simple embedding)

        This is goverend by the metadata info from `batch.example`.

        Returns:
        - (default): loss
        - `return_decoder_inputs`: decoder_context, decoder_context_layout
        - `return_logits=True`: loss, logits, logits_layout, decoder_context, decoder_context_layout
        """
        device = batch.source_seqs.device
        dtype = batch.source_seqs.dtype

        # Prepare the batch for forward computation
        batch = self.prepare_batch(batch)

        # Validate that input tensor match model type
        self.ensure_valid_forward_inputs(
            batch,
            self.model_type,
            self.language_column_name,
            self.n_context_examples,
            self.lang_embeddings,
            self.training,
        )

        # Choose syntax by model type
        match self.model_type:
            case ModelType.LLM_ASR:
                inputs = self.create_default_syntax(batch, device)
            case ModelType.LLM_ASR_LID:
                inputs = self.create_default_syntax(batch, device)
            case ModelType.ZERO_SHOT:
                inputs = self.create_zero_shot_syntax(batch, device)

        # Embed all modalities
        embedded = self.embed_inputs(inputs, dtype)

        # Concat all decoder inputs
        (
            decoder_inputs,
            decoder_inputs_layout,
            decoder_context_inputs,
            decoder_context_layout,
        ) = self.concat_inputs(embedded)

        # short-circuit when using beamsearch during inference
        if return_decoder_inputs:
            return decoder_context_inputs, decoder_context_layout

        # Run the decoder
        dec_out = self.llama_decoder(decoder_inputs, decoder_inputs_layout)
        logits = self.final_proj(dec_out)

        targets, targets_layout = batch.as_target_input()
        loss = self.compute_loss(
            logits=logits,
            logit_layout=decoder_inputs_layout,
            targets=targets,
            target_layout=targets_layout,
            decoder_context_layout=decoder_context_layout,
            pad_idx=self.target_vocab_info.pad_idx,  # type: ignore
            eos_idx=self.target_vocab_info.eos_idx,  # type: ignore
        )

        if return_logits:
            return (
                loss,
                logits,
                decoder_inputs_layout,
                decoder_context_inputs,
                decoder_context_layout,
            )

        return loss

    @staticmethod
    def ensure_valid_forward_inputs(
        batch,
        model_type,
        language_column_name,
        n_context_examples,
        lang_embeddings,
        is_training,
    ):
        if model_type == ModelType.LLM_ASR_LID:
            if is_training:
                # Force lang id availability during training, optional during inference
                if language_column_name not in batch.example:
                    raise ValueError(
                        f"Language column '{language_column_name}' must be preset in batch.example for an LID model."
                    )
                if len(batch.example[language_column_name]) != batch.source_seqs.size(
                    0
                ):
                    raise ValueError(
                        f"Language column '{language_column_name}' size must match the batch size."
                    )
            if lang_embeddings is None:
                raise ValueError(
                    "Wav2Vec2LlamaModel.lang_embeddings must be set for an LID model. Please set lang_embeddings."
                )
        elif model_type == ModelType.ZERO_SHOT:
            if (
                "context_audio" not in batch.example
                or "context_text" not in batch.example
            ):
                raise ValueError(
                    "context_audio and context_text must be preset in batch.example for a zero-shot model."
                )
            if (
                len(batch.example["context_audio"]) < n_context_examples
                or len(batch.example["context_text"]) < n_context_examples
            ):
                raise ValueError(
                    f"context_audio and context_text must of length {n_context_examples} for this zero-shot model."
                )

    def compute_loss(
        self,
        logits: Tensor,
        logit_layout: BatchLayout,
        targets: Tensor,
        target_layout: BatchLayout,
        decoder_context_layout: BatchLayout,
        pad_idx: int,
        eos_idx: int,
    ) -> Tensor:
        """Compute cross-entropy loss for speech-to-text generation.

        Returns:
            A tensor representing the loss per sample in the batch.
        """
        # Add EOS to the targets
        targets, target_layout = Wav2Vec2LlamaModel.add_eos(
            targets, target_layout, pad_idx, eos_idx
        )

        # Choose the indices BOS : BOS + max_target_length
        logits_no_enc = Wav2Vec2LlamaModel.remove_context_logits(
            logits=logits,
            logit_layout=logit_layout,
            targets=targets,
            target_layout=target_layout,
            decoder_context_layout=decoder_context_layout,
        )

        # Run CE loss
        loss = torch.nn.functional.cross_entropy(
            input=logits_no_enc.transpose(1, 2),
            target=targets,
            ignore_index=pad_idx,
            reduction="sum",
        )

        # Average per token, but multiple by the number of samples in the batch,
        # Resulting in the required summed loss across the batch, but still considering
        # every token equally in the batch (no advantage to shorter sequences)
        loss = loss / (target_layout.seq_lens_pt).sum() * targets.size(0)
        return loss

    @staticmethod
    def add_eos(
        targets: Tensor, target_layout: BatchLayout, pad_idx: int, eos_idx: int
    ) -> tuple[Tensor, BatchLayout]:
        """Expands `targets` by one additional pad token and emplaces the eos token
        at the end of every sequence.
        """
        # (N, S, D) -> (N, S+1, D)
        targets = torch.cat(
            [
                targets,
                torch.full_like(targets[:, :1], fill_value=pad_idx),
            ],
            dim=-1,
        )
        # (N, S+1, D) -> (N, S+1, D) (emplace eos with the pad token at the end of every seq)
        targets[torch.arange(targets.size(0)), target_layout.seq_lens] = eos_idx

        new_seq_lens: List[int] = (target_layout.seq_lens_pt + 1).tolist()
        new_target_layout = BatchLayout.of(targets, new_seq_lens)

        return targets, new_target_layout

    @staticmethod
    def remove_context_logits(
        logits: Tensor,
        logit_layout: BatchLayout,
        targets: Tensor,
        target_layout: BatchLayout,
        decoder_context_layout: BatchLayout,
    ) -> Tensor:
        """Extracts target logits by removing context portion from decoder output.

        The decoder processes concatenated context+target sequences. This function
        extracts only the logits corresponding to the target portion of the loss computation.
        """
        # zero-filled tensor matching the target size
        logits_no_context = torch.zeros_like(
            logits[:, : targets.size(1), :],
        )
        # copy logits[context_start:context_start + target_len] to output
        for i in range(logits.size(0)):
            context_len_i = decoder_context_layout.seq_lens_pt[i]
            tgt_len_i = target_layout.seq_lens_pt[i]
            total_len_i = logit_layout.seq_lens_pt[i]
            assert context_len_i + tgt_len_i == total_len_i
            logits_no_context[i, :tgt_len_i] = logits[
                i, context_len_i - 1 : context_len_i - 1 + tgt_len_i
            ]
        return logits_no_context

    def prepare_batch(self, batch: Seq2SeqBatch) -> Seq2SeqBatch:
        """Transforms context data from batch-of-sequences to sequence-of-batch layout.

        Before: context[batch_id][position]
        After:  context[position][batch_id]
        """

        example = batch.example if batch.example is not None else {}
        assert isinstance(example, dict)

        # Change from one context tensor per example in the batch to one tensor per context example location
        if "context_audio" in example:
            max_context_len = max(
                item["seqs"].size(0) for item in example["context_audio"]
            )
            audio_result = []
            audio_zeros = torch.zeros_like(example["context_audio"][0]["seqs"][0][:1])
            text_result = []
            text_zeros = torch.zeros_like(example["context_text"][0]["seqs"][0][:1])
            # for every turn in the conversation
            for i in range(max_context_len):
                # For audio
                # collect seq_lens for i'th turn, zero if missing
                lens = torch.tensor(
                    [
                        x["seq_lens"][i] if i < len(x["seq_lens"]) else 0
                        for x in example["context_audio"]
                    ],
                    dtype=torch.int64,
                    device=batch.source_seqs.device,
                )
                # collect seqs for i'th turn, zero-tensor if missing
                tensor_list = [
                    x["seqs"][i, : lens[b]] if i < len(x["seqs"]) else audio_zeros
                    for b, x in enumerate(example["context_audio"])
                ]
                # construct batch from sequences and pad accordingly to the longest seq
                audio_result.append(
                    {
                        "seqs": pad_sequence(
                            tensor_list, batch_first=True, padding_value=0
                        ),
                        "seq_lens": lens.tolist(),
                    }
                )

                # For text
                # collect text seq_lens for i'th turn, 0 if missing
                lens = torch.tensor(
                    [
                        x["seq_lens"][i] if i < len(x["seq_lens"]) else 0
                        for x in example["context_text"]
                    ],
                    dtype=torch.int64,
                    device=batch.source_seqs.device,
                )
                # collect seqs for i'th turn, zero-tensor if missing
                tensor_list = [
                    x["seqs"][i, : lens[b]] if i < len(x["seqs"]) else text_zeros
                    for b, x in enumerate(example["context_text"])
                ]
                # construct batch from sequences and pad accordingly to longest seq
                text_result.append(
                    {
                        "seqs": pad_sequence(
                            tensor_list, batch_first=True, padding_value=0
                        ),
                        "seq_lens": lens.tolist(),
                    }
                )
            assert isinstance(batch.example, dict)
            batch.example["context_audio"] = audio_result
            batch.example["context_text"] = text_result

        return batch

    def _lang_id_getter(self, lang: str) -> int:
        """Get the lang ID for a given language code."""
        assert self.lang_mapping is not None, "lang_mapping must be set"
        if lang.lower() in self.lang_mapping:
            return self.lang_mapping[lang.lower()]
        if lang in self.lang_mapping:
            return self.lang_mapping[lang]

        log.info(f"lang not in mapping: {lang}")
        return 0

    def create_default_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[Dict[str, object]]:
        # Create a dict of inputs for the base case. Ths syntax is:
        # target audio <special token> lang <bos> target text <eos>
        # in case lang code is not in the batch, or not in the mapping, we use ID 0.
        # in addition, we use ID 0 with prob 1 - self.lang_embeddings_p.
        # if self.lang_embeddings_p == 0.0, we omit the "<special token> lang" part.

        # Choose lang IDs
        example = batch.example or {}
        assert isinstance(example, dict)

        batch_size = batch.source_seqs.size(0)
        if self.lang_embeddings_p == 0.0 or self.language_column_name not in example:
            lang_id_ = [0] * batch_size
        elif self.lang_embeddings_p > 0.0 and self.language_column_name in example:
            lang_array = example[self.language_column_name]
            assert (
                len(lang_array) == batch_size
            ), f"lang array length {len(lang_array)} != {batch_size} (should be batch size)"
            lang_id_ = list(map(self._lang_id_getter, lang_array))
        else:
            raise ValueError("lang not in batch, but lang_embeddings_p > 0.0")

        lang_id = torch.tensor(lang_id_, device=device, dtype=torch.int64)
        lang_seq_lens: List[int] = [1] * batch_size

        # only drop lang id during training
        if self.training:
            drop_mask = (
                torch.rand([batch_size], generator=self.rng)
                < 1 - self.lang_embeddings_p
            )
            lang_id[drop_mask] = 0

        special_token = self.target_vocab_info.size

        inputs = [
            {
                "value": {
                    "seqs": batch.source_seqs,
                    "seq_lens": batch.source_seq_lens,
                },
                "type": "audio",
                "loss": False,
            },
        ]  # audio

        if self.lang_embeddings_p > 0.0:
            inputs += [
                {
                    "value": {
                        "seqs": self.create_single_char(batch, special_token, device)
                    },
                    "type": "text",
                    "loss": False,
                },  # special token
                {
                    "value": {"seqs": lang_id.unsqueeze(-1), "seq_lens": lang_seq_lens},
                    "type": "lang",
                    "loss": False,
                },  # lang
            ]

        inputs += [
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.bos_idx, device  # type: ignore
                    )
                },
                "type": "text",
                "loss": False,
            },  # bos
            {
                "value": {
                    "seqs": batch.target_seqs,
                    "seq_lens": batch.target_seq_lens,
                },
                "type": "text",
                "loss": True,
            },  # target text
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.eos_idx, device  # type: ignore
                    )
                },
                "type": "text",
                "loss": True,
            },  # eos
        ]
        return inputs

    def create_text_context_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[Dict[str, object]]:
        # Create a dict of inputs. The syntax is:
        # <context> (<context example> context text </context example>) X N </context> target audio <bos> target text <eos>
        assert self.target_vocab_info.bos_idx is not None  # Silence linter
        n_context = len(batch.example["context_audio"])  # type: ignore
        if n_context == 0:
            raise ValueError("No context examples found")

        inputs = []

        # Set indices for special tokens in the syntax
        context_start = self.target_vocab_info.size
        context_end = self.target_vocab_info.size + 1
        context_example_start = self.target_vocab_info.size + 2
        context_example_end = self.target_vocab_info.size + 3

        # Build syntax
        inputs += [
            {
                "value": {
                    "seqs": self.create_single_char(batch, context_start, device)
                },
                "type": "text",
                "loss": False,
            },
        ]

        for i in range(n_context):
            inputs += [
                {
                    "value": {
                        "seqs": self.create_single_char(
                            batch, context_example_start, device
                        )
                    },
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": batch.example["context_text"][i],  # type: ignore
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": {
                        "seqs": self.create_single_char(
                            batch, context_example_end, device
                        )
                    },
                    "type": "text",
                    "loss": False,
                },
            ]
        inputs += [
            {
                "value": {"seqs": self.create_single_char(batch, context_end, device)},
                "type": "text",
                "loss": False,
            },
            {
                "value": {
                    "seqs": batch.source_seqs,
                    "seq_lens": batch.source_seq_lens,
                },
                "type": "audio",
                "loss": False,
            },
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.bos_idx, device
                    )
                },
                "type": "text",
                "loss": False,
            },
            {
                "value": {
                    "seqs": batch.target_seqs,
                    "seq_lens": batch.target_seq_lens,
                },
                "type": "text",
                "loss": True,
            },
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.eos_idx, device  # type: ignore
                    )
                },
                "type": "text",
                "loss": True,
            },
        ]
        return inputs

    def create_zero_shot_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[Dict[str, object]]:
        # Create a dict of inputs. Ths syntax is:
        # <context> (<context example> context audio <context_bos> context text <context_eos> </context example>) X N </context> target audio <bos> target text <eos>
        n_context = len(batch.example["context_audio"])  # type: ignore
        inputs = []

        # Set indices for special tokens in the syntax
        context_start = self.target_vocab_info.size
        context_end = self.target_vocab_info.size + 1
        context_example_start = self.target_vocab_info.size + 2
        context_example_end = self.target_vocab_info.size + 3
        context_bos = self.target_vocab_info.size + 4
        context_eos = self.target_vocab_info.size + 5

        # Build syntax
        inputs += [
            {
                "value": {
                    "seqs": self.create_single_char(batch, context_start, device)
                },
                "type": "text",
                "loss": False,
            },
        ]

        for i in range(n_context):
            inputs += [
                {
                    "value": {
                        "seqs": self.create_single_char(
                            batch, context_example_start, device
                        )
                    },
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": batch.example["context_audio"][i],  # type: ignore
                    "type": "audio",
                    "loss": False,
                },
                {
                    "value": {
                        "seqs": self.create_single_char(batch, context_bos, device)
                    },
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": batch.example["context_text"][i],  # type: ignore
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": {
                        "seqs": self.create_single_char(batch, context_eos, device)
                    },
                    "type": "text",
                    "loss": False,
                },
                {
                    "value": {
                        "seqs": self.create_single_char(
                            batch, context_example_end, device
                        )
                    },
                    "type": "text",
                    "loss": False,
                },
            ]
        inputs += [
            {
                "value": {"seqs": self.create_single_char(batch, context_end, device)},
                "type": "text",
                "loss": False,
            },
            {
                "value": {
                    "seqs": batch.source_seqs,
                    "seq_lens": batch.source_seq_lens,
                },
                "type": "audio",
                "loss": False,
            },
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.bos_idx, device  # type: ignore
                    )
                },
                "type": "text",
                "loss": False,
            },
            {
                "value": {
                    "seqs": batch.target_seqs,
                    "seq_lens": batch.target_seq_lens,
                },
                "type": "text",
                "loss": True,
            },
            {
                "value": {
                    "seqs": self.create_single_char(
                        batch, self.target_vocab_info.eos_idx, device  # type: ignore
                    )
                },
                "type": "text",
                "loss": True,
            },
        ]
        return inputs  # type: ignore[return-value]

    @staticmethod
    def create_single_char(batch: Seq2SeqBatch, char: int, device: Device) -> Tensor:
        return torch.full_like(
            batch.target_seqs[:, :1], fill_value=char, device=device  # type: ignore
        )

    def embed_inputs(
        self, inputs: List[Dict[str, object]], dtype: torch.dtype
    ) -> List[Dict[str, object]]:
        # Embed the different modalities
        for inp in inputs:
            # Embed the modality
            if inp["type"] == "audio":
                inp["value"]["seqs"], inp["value"]["seq_lens"] = self.embed_audio(  # type: ignore
                    seqs=inp["value"]["seqs"], seq_lens=inp["value"]["seq_lens"]  # type: ignore
                )
            elif inp["type"] == "text":
                inp["value"]["seqs"] = self.embed_text(inp["value"]["seqs"], dtype)  # type: ignore
            elif inp["type"] == "lang":
                inp["value"]["seqs"] = self.lang_embeddings(inp["value"]["seqs"]).to(  # type: ignore
                    dtype
                )
            else:
                raise ValueError(f"Unknown input type: {inp['type']}")
        return inputs  # type: ignore[return-value]

    def embed_audio(
        self, seqs: Tensor, seq_lens: List[int]
    ) -> Tuple[Tensor, List[int]]:
        """Runs the encoder and its frontend on the audio tensors.
        Maintains the seqs/seq_lens interface.

        :returns: Tuple(seqs, seq_lens)
        """

        seqs_layout = BatchLayout.of(batch=seqs, seq_lens=seq_lens)

        # This is somewhat more memory efficient than setting param.requires_grad to False
        # Since the encoder activations will not be saved in the graph too.
        enc_out, enc_layout, _ = self.encoder_frontend.extract_features(
            seqs, seqs_layout
        )
        enc_out, _ = self.encoder_frontend.process_features(
            enc_out, enc_layout, self.masker if self.training else None  # type: ignore
        )
        enc_out = self.encoder(enc_out, enc_layout)

        # Stack the encoder outputs
        if enc_out.size(1) % self.encoder_stacking != 0:
            n_padding = self.encoder_stacking - (
                enc_out.size(1) % self.encoder_stacking
            )
            enc_out = F.pad(enc_out, (0, 0, 0, n_padding))
        assert enc_out.size(1) % self.encoder_stacking == 0
        enc_out = enc_out.view(
            enc_out.size(0),
            enc_out.size(1) // self.encoder_stacking,
            enc_out.size(-1) * self.encoder_stacking,
        )
        new_lengths = torch.where(
            (enc_layout.seq_lens_pt % self.encoder_stacking) == 0,
            enc_layout.seq_lens_pt // self.encoder_stacking,
            enc_layout.seq_lens_pt // self.encoder_stacking + 1,
        )
        enc_seq_lens = new_lengths.tolist()

        # Project encoder outputs to decoder input dimension
        enc_out = self.encoder_proj(enc_out)
        self.audio_encoder_calls += 1
        return enc_out, enc_seq_lens

    def embed_text(self, seqs: Tensor, dtype: torch.dtype) -> Tensor:
        return self.text_frontend(seqs).to(dtype)

    def concat_inputs(
        self, inputs: List[Dict[str, object]]
    ) -> Tuple[Tensor, BatchLayout, Tensor, BatchLayout]:

        # Get input information
        t = inputs[0]["value"]["seqs"]  # type: ignore

        device = t.device
        dtype = t.dtype
        batch_size = t.size(0)
        input_dim = t.size(2)
        ones_list = [1] * batch_size

        # Compute total lengths
        lengths: List[List[int]] = [
            inp["value"]["seq_lens"] if "seq_lens" in inp["value"] else ones_list  # type: ignore
            for inp in inputs
        ]
        # Sum the lengths per batch element
        total_lengths: List[int] = [
            sum(length[b] for length in lengths) for b in range(batch_size)
        ]
        max_total_length = max(total_lengths)

        # Init the matrix with zeros
        decoder_inputs = torch.zeros(
            [batch_size, max_total_length, input_dim], device=device, dtype=dtype
        )

        # Put everything in the right place
        lengths_tensor = [torch.tensor(length, device=device) for length in lengths]
        for b in range(batch_size):
            b_inputs_ = [
                inp["value"]["seqs"][b : b + 1, : length[b]]  # type: ignore
                for (inp, length) in zip(inputs, lengths_tensor)
            ]
            b_inputs = torch.cat(b_inputs_, dim=1)
            del b_inputs_
            assert b_inputs.size(1) == total_lengths[b]
            decoder_inputs[b, : b_inputs.size(1)] = b_inputs

        # Compute total context length (everything that we don't train the loss for)
        context_lengths: List[List[int]] = [
            inp["value"]["seq_lens"] if "seq_lens" in inp["value"] else ones_list  # type: ignore
            for inp in inputs
            if inp["loss"] is False
        ]
        total_context_lengths: List[int] = [
            sum(length[b] for length in context_lengths) for b in range(batch_size)
        ]
        max_context_length = max(total_context_lengths)
        decoder_context_inputs = decoder_inputs[:, :max_context_length]

        decoder_input_layout = BatchLayout.of(
            batch=decoder_inputs, seq_lens=total_lengths
        )
        decoder_context_layout = BatchLayout.of(
            batch=decoder_context_inputs, seq_lens=total_context_lengths
        )

        return (
            decoder_inputs,
            decoder_input_layout,
            decoder_context_inputs,
            decoder_context_layout,
        )
