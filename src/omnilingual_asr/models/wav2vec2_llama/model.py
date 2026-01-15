# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import List, Sequence, Tuple, final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq2.data.tokenizers import TokenEncoder, VocabularyInfo
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
    Wav2Vec2LlamaSpecialTokens,
    Wav2Vec2LlamaStreamingConfig,
)
from omnilingual_asr.models.wav2vec2_llama.syntax import (
    Modality,
    ModalityInput,
    create_lang_inputs,
    create_single_char_input,
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
        streaming_config: Wav2Vec2LlamaStreamingConfig = Wav2Vec2LlamaStreamingConfig(),
        text_encoder: TokenEncoder | None = None,
        n_context_examples: int = 0,
        seed: int = 42,
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
        self.rng.manual_seed(seed)
        self.audio_encoder_calls = 0
        self.language_column_name = language_column_name
        self.context_text_only = context_text_only
        self.beam_search_config = beam_search_config
        self.n_context_examples = n_context_examples
        self.streaming_config = streaming_config
        self.text_encoder = text_encoder
        self.special_tokens = Wav2Vec2LlamaSpecialTokens(self.target_vocab_info.size)

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
        | Tuple[List[Tensor], List[List[int]], List[ModalityInput]]
        | Tuple[
            Tensor,
            Tensor,
            BatchLayout,
            List[Tensor],
            List[List[int]],
            List[ModalityInput],
        ]
    ):
        """Model entry point, accepts a `Seq2Seq` batch and embeds the inputs w.r.t.
        their modality:
        - audio is embedded with encoder_frontend, encoder, final_proj
        - text is embedded with text_frontend (simple embedding)
        - lang tokens are embedded with lang_frontend (simple embedding)

        This is goverend by the metadata info from `batch.example`.

        Returns:
        - (default): loss
        - `return_decoder_inputs=True`: decoder_context, decoder_context_seq_lens, audio_embeddings
        - `return_logits=True`: loss, logits, decoder_inputs_layout, decoder_context_inputs, decoder_context_seq_lens, audio_embeddings
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
        match (self.model_type, self.streaming_config.is_streaming):
            case (ModelType.LLM_ASR, False):
                inputs = self.create_default_syntax(batch, device)
            case (ModelType.ZERO_SHOT, False):
                inputs = self.create_zero_shot_syntax(batch, device)
            case (ModelType.LLM_ASR_LID, False):
                inputs = self.create_default_syntax(batch, device)
            case (ModelType.LLM_ASR_LID, True):
                inputs = self.create_streaming_syntax(batch, device)

        # Embed all modalities
        if self.training:
            embedded = self.embed_inputs_training(inputs, dtype)  # type: ignore
        else:
            embedded = self.embed_inputs(inputs, dtype)  # type: ignore

        # Extract embeddings for incremental decoding
        audio_embeddings = [inp for inp in embedded if inp.modality == Modality.AUDIO]
        # Concat all decoder inputs
        (
            decoder_inputs,
            decoder_inputs_layout,
            decoder_context_inputs,
            decoder_context_seq_lens,
            loss_mask,
        ) = self.concat_inputs(embedded)

        # short-circuit when using beamsearch during inference
        if return_decoder_inputs:
            return decoder_context_inputs, decoder_context_seq_lens, audio_embeddings

        # Run the decoder
        dec_out = self.llama_decoder(decoder_inputs, decoder_inputs_layout)
        logits_ = self.final_proj(dec_out)

        # BC SDPA workaround
        logits, loss_mask = Wav2Vec2LlamaModel.crop_to_true_lengths(
            logits=logits_,
            loss_mask=loss_mask,
            true_total_lengths=decoder_inputs_layout.seq_lens,
        )

        targets, targets_layout = batch.as_target_input()
        loss = self.compute_loss(
            logits=logits,
            logit_layout=decoder_inputs_layout,
            targets=targets,
            target_layout=targets_layout,
            decoder_context_seq_lens=decoder_context_seq_lens,
            loss_mask=loss_mask,
            pad_idx=self.target_vocab_info.pad_idx,  # type: ignore
            eos_idx=self.target_vocab_info.eos_idx,  # type: ignore
            batch=batch,
        )

        if return_logits:
            return (
                loss,
                logits,
                decoder_inputs_layout,
                decoder_context_inputs,
                decoder_context_seq_lens,
                audio_embeddings,
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
        """Validate batch data matches model type requirements.

        LLM_ASR_LID: Requires lang column during training and lang_embeddings.
        ZERO_SHOT: Requires context_audio and context_text with sufficient examples.
        """
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
        decoder_context_seq_lens: List[List[int]],
        loss_mask: Tensor,
        pad_idx: int,
        eos_idx: int,
        batch: Seq2SeqBatch,
    ) -> Tensor:
        """Compute cross-entropy loss for speech-to-text generation.

        Returns:
            A tensor representing the loss per sample in the batch.
        """

        # Different loss calculation if streaming
        if self.streaming_config.is_streaming:
            assert isinstance(batch.example, dict)
            n_segments = len(batch.example["token_segments"])

            # Concatenate targets
            B = logits.size(0)
            target_tensors = []
            eos_tensor = torch.tensor([eos_idx], device=logits.device)
            for b in range(B):
                for seg_i in range(n_segments):
                    target_tensors.append(
                        batch.example["token_segments"][seg_i].seqs[
                            b,
                            : batch.example["token_segments"][seg_i].seq_lens[b],
                        ]
                    )
                    target_tensors.append(eos_tensor)
            targets = torch.cat(target_tensors, dim=0)

            # Mask logits and compute loss
            relevant_logits = logits[loss_mask]
            loss = torch.nn.functional.cross_entropy(
                input=relevant_logits,
                target=targets,
                ignore_index=pad_idx,
                reduction="sum",
            )
        else:
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
                decoder_context_seq_lens=decoder_context_seq_lens,
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
        decoder_context_seq_lens: List[List[int]],
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
            context_len_i = decoder_context_seq_lens[0][i]
            tgt_len_i = target_layout.seq_lens_pt[i]
            total_len_i = logit_layout.seq_lens_pt[i]
            assert context_len_i + tgt_len_i == total_len_i
            logits_no_context[i, :tgt_len_i] = logits[
                i, context_len_i - 1 : context_len_i - 1 + tgt_len_i
            ]
        return logits_no_context

    def prepare_batch(self, batch: Seq2SeqBatch) -> Seq2SeqBatch:
        """Prepare batch for forward pass, transposing context and segmenting for streaming.

        For zero-shot/ICL: Transposes context_audio and context_text from
        [batch_id][position] to [position][batch_id] layout, enabling iteration
        through context positions during syntax construction. Pads with zeros
        for batch elements with fewer context examples.

        For streaming: Segments audio into fixed-duration chunks with aligned text.

        Before: context_audio[batch_id] → all context examples for one batch element
        After: context_audio[position] → batched data for one context position
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
                    ModalityInput(
                        modality=Modality.AUDIO,
                        seqs=pad_sequence(
                            tensor_list, batch_first=True, padding_value=0
                        ),
                        seq_lens=lens.tolist(),
                        loss=False,
                    ),
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
                    ModalityInput(
                        modality=Modality.TEXT,
                        seqs=pad_sequence(
                            tensor_list, batch_first=True, padding_value=0
                        ),
                        seq_lens=lens.tolist(),
                        loss=False,
                    ),
                )
            assert isinstance(batch.example, dict)
            batch.example["context_audio"] = audio_result
            batch.example["context_text"] = text_result

        if self.streaming_config.is_streaming:
            batch = self.prepare_streaming_batch(batch)

        return batch

    def prepare_streaming_batch(self, batch: Seq2SeqBatch) -> Seq2SeqBatch:
        """Segment continuous audio into fixed-size chunks with aligned text.

        Splits audio into fixed-duration segments, dropping final short segments.
        Uses word-level alignment to segment reference text matching audio boundaries.
        Adds audio_segments and token_segments to batch.example for streaming syntax.
        """

        device = batch.source_seqs.device
        # Drop last segments if shorter than min_seg_length
        seg_size = int(
            self.streaming_config.sample_rate * self.streaming_config.segment_secs
        )
        min_seg_length = (
            self.streaming_config.sample_rate
            * self.streaming_config.min_audio_ms
            // 1000
        )
        source_seq_lens_pt = torch.tensor(batch.source_seq_lens, device=device)
        residues = source_seq_lens_pt % seg_size
        trim_mask = residues <= min_seg_length

        new_lengths = source_seq_lens_pt.clone()
        new_lengths[trim_mask] = new_lengths[trim_mask] - residues[trim_mask]
        new_source_seqs = batch.source_seqs[:, : new_lengths.max()]
        new_source_seq_lens = new_lengths.tolist()
        # we have to init a new batch because modifying inplace is not allowed
        batch = Seq2SeqBatch(
            source_seqs=new_source_seqs,
            source_seq_lens=new_source_seq_lens,
            target_seqs=batch.target_seqs,
            target_seq_lens=list(batch.target_seq_lens),
            example=batch.example,
        )
        # recreate the same source len tensor, dirty but we're looking for functionality for now
        source_seq_lens_pt = torch.tensor(batch.source_seq_lens, device=device)

        # Get number of segments
        local_n_segments_all = torch.ceil(source_seq_lens_pt / seg_size).int()
        local_n_segments = local_n_segments_all.max()
        n_segments = local_n_segments

        # Split audio to segments. Each example in the batch may have a different number of
        # segments. In addition, we add some dummy segments to match the nax number of
        # batches across workers, to avoid fsdp hangs.
        audio_segments = []
        zero_lengths = torch.zeros_like(source_seq_lens_pt)
        for i in range(n_segments):
            if i < local_n_segments:
                audio_seg = batch.source_seqs[:, i * seg_size : (i + 1) * seg_size]
                last_seg = (i + 1) * seg_size > source_seq_lens_pt
                after_last_seg = (i + 1) * seg_size > source_seq_lens_pt + seg_size
                seg_lengths = torch.where(
                    last_seg,
                    torch.where(
                        after_last_seg,
                        zero_lengths,
                        source_seq_lens_pt - i * seg_size,
                    ),
                    seg_size,
                )
                assert seg_lengths.min() >= 0
            else:
                audio_seg = batch.source_seqs[:, :min_seg_length]  # Dummy
                seg_lengths = zero_lengths
            audio_segments.append(
                ModalityInput(
                    modality=Modality.AUDIO,
                    seqs=audio_seg,
                    seq_lens=seg_lengths.tolist(),
                    loss=False,
                ),
            )

        # Aggregate segment refs
        segment_refs: List[List[str]] = []
        B = batch.source_seqs.size(0)
        sample_lengths = (
            np.array(batch.source_seq_lens) / self.streaming_config.sample_rate
        )
        assert isinstance(batch.example, dict)
        for b in range(B):
            durations = batch.example["word_duration"][b]
            word_ends = durations.cumsum()
            word_ends[word_ends > sample_lengths[b]] = sample_lengths[
                b
            ]  # Trim if over the sample length
            word_refs = batch.example["text_words_merged"][b]
            segment_id = (
                (word_ends - 1e-4) // self.streaming_config.segment_secs
            ).astype(
                np.int32
            )  # -1e-4 to avoid assining to a new segment if word ends on the boundary

            segment_refs.append([])
            if len(segment_id) > 0:  # Can be 0 if the segment has no text
                for i in range(segment_id.max() + 1):
                    seg_ref = "".join(word_refs[segment_id == i])
                    segment_refs[b].append(seg_ref)
            assert len(segment_refs[b]) <= local_n_segments, batch

        assert self.text_encoder is not None  # linter

        # Tokenize segment refs and batch again
        token_segments = []
        pad_idx = self.target_vocab_info.pad_idx
        assert pad_idx is not None

        for i in range(n_segments):
            # Get ref for segment i for all the batch
            seg_refs = [
                segment_refs[b][i] if i < len(segment_refs[b]) else "" for b in range(B)
            ]
            seg_tokens_: List[Tensor] = [
                self.text_encoder("=" + seg_ref + "=")[1:-1] for seg_ref in seg_refs
            ]  # Adding "=" since the tokenizer runs strip()
            seg_ref_lengths = [x.size(0) for x in seg_tokens_]
            seg_tokens = pad_sequence(
                seg_tokens_,
                batch_first=True,
                padding_value=pad_idx,
            ).to(device)
            token_segments.append(
                ModalityInput(
                    modality=Modality.TEXT,
                    seqs=seg_tokens,
                    seq_lens=list(seg_ref_lengths),
                    loss=False,
                ),
            )

        # Set segments in the batch
        assert len(audio_segments) == len(
            token_segments
        ), f"{len(audio_segments)=} {len(token_segments)=}"
        assert len(audio_segments) == n_segments
        batch.example["audio_segments"] = audio_segments
        batch.example["token_segments"] = token_segments
        batch.example["n_segments"] = local_n_segments_all

        return batch

    def create_default_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[ModalityInput]:
        """Create default decoder input syntax for LLM-ASR with optional language ID.

        Syntax: target audio [<special token> lang] <bos> target text <eos>
        Language ID included if lang_embeddings_p > 0, with dropout during training.
        """

        inputs = [
            ModalityInput(
                modality=Modality.AUDIO,
                seqs=batch.source_seqs,
                seq_lens=list(batch.source_seq_lens),
                loss=False,
            )
        ]

        if self.lang_embeddings_p > 0.0:
            assert (
                self.lang_mapping is not None
            ), f"{self.lang_embeddings_p=} without lang_mapping"

            # Generate dropout mask during training
            dropout_mask = None
            if self.training:
                batch_size = batch.source_seqs.size(0)
                dropout_mask = torch.rand(batch_size, device=device) < (
                    1 - self.lang_embeddings_p
                )

            lid_marker_input, lang_id_input = create_lang_inputs(
                batch=batch,
                lid_marker=self.special_tokens.lid_marker,
                lang_mapping=self.lang_mapping,
                lang_column_name=self.language_column_name,
                dropout_mask=dropout_mask,
                device=device,
            )
            inputs += [lid_marker_input, lang_id_input]
        bos_idx, eos_idx = self._get_bos_eos()
        bos_input = create_single_char_input(batch, bos_idx, device=device)
        eos_input = create_single_char_input(batch, eos_idx, device=device, loss=True)

        inputs += [
            bos_input,
            ModalityInput(
                modality=Modality.TEXT,
                seqs=batch.target_seqs,
                seq_lens=list(batch.target_seq_lens),
                loss=True,
            ),
            eos_input,
        ]

        return inputs

    def _get_bos_eos(self) -> Tuple[int, int]:
        bos_idx = self.target_vocab_info.bos_idx
        eos_idx = self.target_vocab_info.eos_idx
        assert bos_idx is not None
        assert eos_idx is not None
        return bos_idx, eos_idx

    def create_text_context_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[ModalityInput]:
        """Create decoder input syntax with text-only context examples.

        Syntax: <context> (<context example> context text </context example>) x N </context> target audio <bos> target text <eos>
        Uses n_context_examples text demonstrations before main input.
        """
        bos_idx, eos_idx = self._get_bos_eos()

        n_context = len(batch.example["context_audio"])  # type: ignore
        assert n_context != 0, "No context examples found."

        context_start_input = create_single_char_input(
            batch, self.special_tokens.context_start, device=device
        )

        inputs = [context_start_input]

        for i in range(n_context):

            context_example_start_input = create_single_char_input(
                batch, self.special_tokens.context_example_start, device=device
            )
            context_example_end_input = create_single_char_input(
                batch, self.special_tokens.context_example_end, device=device
            )

            inputs += [
                context_example_start_input,
                ModalityInput(
                    modality=Modality.TEXT,
                    seqs=batch.example["context_text"][i]["seqs"],  # type: ignore
                    seq_lens=batch.example["context_text"][i]["seq_lens"],  # type: ignore
                    loss=False,
                ),
                context_example_end_input,
            ]

        context_end_input = create_single_char_input(
            batch, self.special_tokens.context_end, device=device
        )
        bos_input = create_single_char_input(batch, bos_idx, device=device)
        eos_input = create_single_char_input(batch, eos_idx, device=device, loss=True)

        inputs += [
            context_end_input,
            ModalityInput(
                modality=Modality.AUDIO,
                seqs=batch.source_seqs,
                seq_lens=list(batch.source_seq_lens),
                loss=False,
            ),
            bos_input,
            ModalityInput(
                modality=Modality.TEXT,
                seqs=batch.target_seqs,
                seq_lens=list(batch.target_seq_lens),
                loss=True,
            ),
            eos_input,
        ]
        return inputs

    def create_zero_shot_syntax(
        self, batch: Seq2SeqBatch, device: Device
    ) -> List[ModalityInput]:
        """Create decoder input syntax for zero-shot learning with audio+text context.

        Syntax: <context> (<context example> audio <bos> text <eos> </context example>) x N </context> target audio <bos> target text <eos>
        Demonstrates audio-to-text mapping with n_context_examples before main task.
        """
        assert isinstance(batch.example, dict)
        n_context = len(batch.example["context_audio"])  # type: ignore

        context_start_input = create_single_char_input(
            batch, self.special_tokens.context_start, device=device
        )
        context_example_start_input = create_single_char_input(
            batch, self.special_tokens.context_example_start, device=device
        )
        context_example_end_input = create_single_char_input(
            batch, self.special_tokens.context_example_end, device=device
        )
        context_bos_input = create_single_char_input(
            batch, self.special_tokens.context_bos, device=device
        )
        context_eos_input = create_single_char_input(
            batch, self.special_tokens.context_eos, device=device
        )

        inputs = []
        inputs += [context_start_input]

        for i in range(n_context):

            inputs += [
                context_example_start_input,
                ModalityInput(
                    modality=Modality.AUDIO,
                    seqs=batch.example["context_audio"][i].seqs,
                    seq_lens=batch.example["context_audio"][i].seq_lens,
                    loss=False,
                ),
                context_bos_input,
                ModalityInput(
                    modality=Modality.TEXT,
                    seqs=batch.example["context_text"][i].seqs,
                    seq_lens=batch.example["context_text"][i].seq_lens,
                    loss=False,
                ),
                context_eos_input,
                context_example_end_input,
            ]

        context_end_input = create_single_char_input(
            batch, self.special_tokens.context_end, device=device
        )
        bos_idx, eos_idx = self._get_bos_eos()
        bos_input = create_single_char_input(batch, bos_idx, device=device)
        eos_input = create_single_char_input(batch, eos_idx, device=device)

        inputs += [
            context_end_input,
            ModalityInput(
                modality=Modality.AUDIO,
                seqs=batch.source_seqs,
                seq_lens=list(batch.source_seq_lens),
                loss=False,
            ),
            bos_input,
            ModalityInput(
                modality=Modality.TEXT,
                seqs=batch.target_seqs,
                seq_lens=list(batch.target_seq_lens),
                loss=True,
            ),
            eos_input,
        ]
        return inputs  # type: ignore[return-value]

    def create_streaming_syntax(
        self,
        batch: Seq2SeqBatch,
        device: Device,
        inference: bool = False,
    ) -> List[ModalityInput]:
        """Create decoder input syntax for streaming ASR with segmented audio.

        Syntax: [lang <lang>] (audio_i <segment marker> <bos> text_i <eos>) x N
        Processes audio in segments, marking last segment differently for proper termination.
        During inference, reuses cached embeddings from previous segments.
        """
        assert isinstance(batch.example, dict)
        inputs = []
        if self.lang_embeddings_p > 0.0:
            assert (
                self.lang_mapping is not None
            ), f"{self.lang_embeddings_p=} without lang_mapping"

            # Generate dropout mask during training
            dropout_mask = None
            if self.training:
                batch_size = batch.source_seqs.size(0)
                dropout_mask = torch.rand(batch_size, device=device) < (
                    1 - self.lang_embeddings_p
                )

            lid_marker_input, lang_id_input = create_lang_inputs(
                batch=batch,
                lid_marker=self.special_tokens.lid_marker,
                lang_mapping=self.lang_mapping,
                lang_column_name=self.language_column_name,
                dropout_mask=dropout_mask,
                device=device,
            )

            inputs += [lang_id_input, lid_marker_input]

        # Add segments
        n_segments = len(batch.example["audio_segments"])
        for seg_i in range(n_segments):
            inputs += [
                ModalityInput(
                    modality=Modality.AUDIO,
                    seqs=batch.example["audio_segments"][seg_i].seqs,
                    seq_lens=batch.example["audio_segments"][seg_i].seq_lens,
                    loss=False,
                ),
            ]

            # Compute token type (last/regular) for each batch element
            is_last = seg_i == batch.example["n_segments"] - 1  # [B] bool
            segment_tokens = torch.where(
                is_last,
                self.special_tokens.last_segment,
                self.special_tokens.regular_segment,
            )  # [B] int tensor

            segment_marker_input = ModalityInput(
                modality=Modality.TEXT,
                seqs=segment_tokens.unsqueeze(-1),  # [B] -> [B, 1]
                seq_lens=[1] * batch.source_seqs.size(0),
                loss=False,
            )
            bos_idx, eos_idx = self._get_bos_eos()
            bos_input = create_single_char_input(batch, bos_idx, device=device)
            eos_input = create_single_char_input(
                batch, eos_idx, device=device, loss=True
            )

            inputs += [
                segment_marker_input,
                bos_input,
                ModalityInput(
                    modality=Modality.TEXT,
                    seqs=batch.example["token_segments"][seg_i].seqs,
                    seq_lens=batch.example["token_segments"][seg_i].seq_lens,
                    loss=True,
                ),
                eos_input,
            ]

        # Adaptations for inference
        if inference:
            for inp in inputs:
                inp.loss = False
            inputs = inputs[:-2]  # Remove segment text and EOS

        return inputs

    def embed_inputs(
        self, inputs: List[ModalityInput], dtype: torch.dtype
    ) -> List[ModalityInput]:
        """Embed all modalities, mutating inputs in-place.

        Audio -> encoder frontend + encoder + projection
        Text -> text embedding
        Lang -> lang embedding

        Zero-length sequences temporarily padded to avoid encoder crashes.
        """

        for inp in inputs:
            # Skip if marked as embedded already (used for streaming inference)
            if inp.embedded:
                continue

            # Pretend zero lengths are longer, to not get an exception. Set back at the end
            zero_indices = [i for i, length in enumerate(inp.seq_lens) if length == 0]

            # Temporarily set zero lengths to max to avoid encoder crashes
            if zero_indices:
                max_len = inp.seqs.size(-1)  # Last dimension (sequence length)
                for i in zero_indices:
                    inp.seq_lens[i] = max_len

            # Embed the modality
            if inp.modality == Modality.AUDIO:
                inp.seqs, inp.seq_lens = self.embed_audio(inp.seqs, inp.seq_lens)
            elif inp.modality == Modality.TEXT:
                inp.seqs = self.embed_text(inp.seqs, dtype)
            elif inp.modality == Modality.LANG:
                inp.seqs = self.lang_embeddings(inp.seqs).to(dtype)  # type: ignore
            else:
                raise ValueError(f"Unknown input modality: {inp.modality}")
            inp.embedded = True

            assert not torch.any(
                inp.seqs.isnan()
            ), f"Found NaNs after embedding in {inp}"

            # Set back the length to zero where needed
            if zero_indices:
                for i in zero_indices:
                    inp.seq_lens[i] = 0

        return inputs

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
        self, inputs: List[ModalityInput]
    ) -> Tuple[Tensor, BatchLayout, List[Tensor], List[List[int]], Tensor]:
        """Concatenate multi-modal inputs into single decoder sequence.

        Flattens all modality segments (audio, text, lang) into a single
        contiguous sequence per batch element, computing per-modality lengths
        and loss masks for training.
        """
        # Get input information
        t = inputs[0].seqs
        device = t.device
        dtype = t.dtype
        batch_size = t.size(0)
        input_dim = t.size(2)

        # Compute total lengths
        lengths: List[List[int]] = [inp.seq_lens for inp in inputs]

        # Sum the lengths per batch element
        total_lengths: List[int] = [
            sum(length[b] for length in lengths) for b in range(batch_size)
        ]
        max_total_length = (
            max(total_lengths) + 1
        )  # Added padding to force the correct SDPA backend for BC parity

        # Init the matrix with zeros
        decoder_inputs = torch.zeros(
            [batch_size, max_total_length, input_dim], device=device, dtype=dtype
        )

        # Put everything in the right place
        lengths_tensor = [torch.tensor(length, device=device) for length in lengths]
        for b in range(batch_size):
            b_inputs_ = [
                inp.seqs[b : b + 1, : length[b]]  # type: ignore
                for (inp, length) in zip(inputs, lengths_tensor)
            ]
            b_inputs = torch.cat(b_inputs_, dim=1)
            del b_inputs_
            decoder_inputs[b, : b_inputs.size(1)] = b_inputs

        # Get the context tensor before each ref text.
        # For example if the syntax is:
        # lang <lang> audio_1 <regular segment> <BOS> text 1 <EOS> audio_2 <last segment> <BOS> text 2 <EOS>
        # then we return:
        # [concat([lang, <lang>, audio_1, <regular segment> <BOS>]),
        #  concat([<EOS>, audio_2, <last segment>, <BOS>])]
        inputs_to_group = []
        decoder_context_inputs = []
        decoder_context_seq_lens = []
        for i, inp in enumerate(inputs):
            if inp.loss is False:
                inputs_to_group.append(inp)

                # If context for segment is done, group it
                if i == len(inputs) - 1 or inputs[i + 1].loss is True:
                    # Get lengths
                    context_lengths = [inp.seq_lens for inp in inputs_to_group]
                    context_lengths_sum: List[int] = [
                        sum(length[b] for length in context_lengths)
                        for b in range(batch_size)
                    ]

                    # Group inputs
                    context_inputs = torch.zeros(
                        [batch_size, int(max(context_lengths_sum)), input_dim],
                        device=device,
                        dtype=dtype,
                    )
                    for b in range(batch_size):
                        b_context_inputs_ = [
                            inp.seqs[b : b + 1, : length[b]]
                            for inp, length in zip(inputs_to_group, context_lengths)
                        ]
                        b_context_inputs = torch.cat(b_context_inputs_, dim=1)
                        context_inputs[b, : b_context_inputs.size(1)] = b_context_inputs

                    # Next
                    inputs_to_group = []
                    decoder_context_inputs.append(context_inputs)
                    decoder_context_seq_lens.append(context_lengths_sum)

        # Prepare loss mask. A boolean mask indicating which output we train for.
        loss_mask = torch.zeros_like(decoder_inputs[:, :, 0], dtype=torch.bool)
        for b in range(batch_size):
            loc = 0
            for inp in inputs:
                inp_b_len = inp.seq_lens[b]
                if inp.loss:
                    loss_mask[b, loc - 1 : loc - 1 + inp_b_len] = True
                loc += inp_b_len

        decoder_input_layout = BatchLayout.of(
            batch=decoder_inputs, seq_lens=total_lengths
        )

        return (
            decoder_inputs,
            decoder_input_layout,
            decoder_context_inputs,
            decoder_context_seq_lens,
            loss_mask,
        )

    def embed_inputs_training(
        self, inputs: List[ModalityInput], dtype: torch.dtype
    ) -> List[ModalityInput]:
        """Embed all modalities with batched audio encoding for training efficiency.

        Batches audio encoder calls across all segments to reduce overhead,
        particularly beneficial for streaming with small segment sizes.
        Text and lang embeddings processed normally.
        """

        # Concate all audio inputs
        audio_inputs = [inp for inp in inputs if inp.modality == Modality.AUDIO]
        sum_b = sum([inp.seqs.size(0) for inp in audio_inputs])
        max_audio_len = max([inp.seqs.size(1) for inp in audio_inputs])
        all_audio_seqs = torch.zeros(
            [sum_b, max_audio_len],
            device=audio_inputs[0].seqs.device,
            dtype=audio_inputs[0].seqs.dtype,
        )
        loc = 0
        for audio_inp in audio_inputs:
            b = audio_inp.seqs.size(0)
            all_audio_seqs[loc : loc + b, : audio_inp.seqs.size(1)] = audio_inp.seqs
            loc += b
        all_audio_seq_lens = [item for inp in audio_inputs for item in inp.seq_lens]

        # Call regular embed_inputs
        audio_inputs = [
            ModalityInput(
                modality=Modality.AUDIO,
                seqs=all_audio_seqs,
                seq_lens=all_audio_seq_lens,
                loss=False,
            ),
        ]
        audio_inputs = self.embed_inputs(audio_inputs, dtype)
        all_audio_seqs = audio_inputs[0].seqs
        all_audio_seq_lens = audio_inputs[0].seq_lens

        # Spread result back to the audio inputs and mark them as already embedded
        audio_loc = 0
        for inp in inputs:
            if inp.modality == Modality.AUDIO:
                b = inp.seqs.size(0)
                lengths = all_audio_seq_lens[audio_loc : audio_loc + b]
                max_len = max(lengths)
                inp.seq_lens = lengths
                inp.seqs = all_audio_seqs[audio_loc : audio_loc + b, :max_len]
                audio_loc += b
                inp.embedded = True

        # Embed the rest of the inputs and return
        inputs = self.embed_inputs(inputs, dtype)
        return inputs

    @staticmethod
    def crop_to_true_lengths(
        logits: Tensor,
        loss_mask: Tensor,
        true_total_lengths: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Crop logits to true sequence lengths, removing SDPA workaround padding.

        Args:
            logits: Logit tensor [B, S+1, V] with workaround padding
            true_total_lengths: Original sequence lengths without padding

        Returns:
            Cropped logits [B, S, V]
        """
        max_true_length = max(true_total_lengths)
        return logits[:, :max_true_length, :], loss_mask[:, :max_true_length]
