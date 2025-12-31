# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import zlib
from typing import List, Tuple, final

import numpy as np
import torch
import torch.nn.functional as F
from fairseq2.datasets import Seq2SeqBatch
from fairseq2.logging import get_log_writer
from fairseq2.nn import BatchLayout, IncrementalStateBag
from torch import Tensor

from omnilingual_asr.models.wav2vec2_llama.config import (
    Wav2Vec2LlamaBeamSearchConfig,
    Wav2Vec2LlamaStreamingConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import (
    Modality,
    ModalityInput,
    Wav2Vec2LlamaModel,
)

log = get_log_writer(__name__)


@final
class Wav2Vec2LlamaBeamSearchSeq2SeqGenerator:
    """Beam search generator for ``Wav2Vec2LLamaModel`` speech-to-text models.

    Performs beam search decoding by maintaining multiple hypothesis candidates,
    prefilling with audio context embeddings, then iteratively generating tokens
    while tracking scores and handling early stopping via compression ratio analysis.
    """

    def __init__(
        self,
        model: Wav2Vec2LlamaModel,
        config: Wav2Vec2LlamaBeamSearchConfig,
        streaming_config: Wav2Vec2LlamaStreamingConfig,
    ) -> None:
        self.model = model
        self.pad_idx = model.target_vocab_info.pad_idx
        self.eos_idx = model.target_vocab_info.eos_idx
        self.bos_idx = model.target_vocab_info.bos_idx
        assert self.bos_idx is not None, "BOS token must be specified"
        assert self.eos_idx is not None, "EOS token must be specified"

        self.config = config
        self.streaming_config = streaming_config

    @staticmethod
    def idx_1d_to_2d(idx: Tensor, dim2: int) -> tuple[Tensor, Tensor]:
        """Convert 1D indices to 2D indices for beam search tracking."""
        return idx // dim2, idx % dim2

    @staticmethod
    def compression_ratio(text: str) -> float:
        """Calculate text compression ratio as heuristic for hallucination detection."""
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))

    @torch.no_grad()
    def generate_hypotheses(
        self,
        decoder_context_inputs: List[Tensor] | None,
        decoder_context_seq_lens: List[List[int]] | None,
        audio_embeddings: List[ModalityInput] | None,
        batch: Seq2SeqBatch | None,
    ) -> Tuple[Tensor, List[int]]:
        """Generate text hypotheses using beam search decoding.

        For streaming mode, processes segments sequentially with caching.
        For non-streaming mode, decodes entire sequence at once.
        Returns top hypothesis tokens and their layout.
        """

        if self.streaming_config.is_streaming:
            assert audio_embeddings is not None
            assert batch is not None
            # In the streaming setting, we decode segments one by one,
            # and concatenate segment hypotheses at the end.
            # When decoding each segment, we create the syntax using the previous
            # already decoded segments. We cache the audio embeddings to not recompute
            # them (which is the largest GPU load).
            assert isinstance(batch.example, dict)
            n_segments = len(audio_embeddings)
            n_context_segments = self.model.streaming_config.n_context_segments
            previous_audio_embeddings: List[ModalityInput] = []
            previous_text_tokens: List[ModalityInput] = []
            languages: List[str] = batch.example.get(self.model.language_column_name, None)  # type: ignore
            B = audio_embeddings[0].seqs.size(0)
            device = audio_embeddings[0].seqs.device

            # Decode segment by segment
            for i in range(n_segments):
                current_audio_embeddings = audio_embeddings[i].seqs
                current_audio_embedding_seq_lens = audio_embeddings[i].seq_lens

                # Apply nonzero mask, to transcribe only samples that didn't end yet
                nonzero_mask = torch.tensor(
                    [x > 0 for x in current_audio_embedding_seq_lens], device=device
                )
                if i == 0 and not nonzero_mask.all():
                    raise ValueError("First segment must be full for the entire batch")
                current_audio_embeddings_nonzero = current_audio_embeddings[
                    nonzero_mask
                ]
                current_audio_embedding_seq_lens_nonzero = [
                    x for x in current_audio_embedding_seq_lens if x > 0
                ]
                n_total_segments_nonzero = batch.example["n_segments"][nonzero_mask]  # type: ignore
                langs_nonzero = (
                    [x[0] for x in zip(languages, nonzero_mask) if x[1]]
                    if languages is not None
                    else None
                )

                previous_audio_embeddings_nonzero = [
                    ModalityInput(
                        modality=Modality.AUDIO,
                        seqs=x.seqs[nonzero_mask],
                        seq_lens=[
                            x_[0] for x_ in zip(x.seq_lens, nonzero_mask) if x_[1] > 0
                        ],
                        loss=False,
                    )
                    for x in previous_audio_embeddings
                ]
                previous_text_tokens_nonzero = [
                    ModalityInput(
                        modality=Modality.TEXT,
                        seqs=x.seqs[nonzero_mask],
                        seq_lens=[
                            x_[0] for x_ in zip(x.seq_lens, nonzero_mask) if x_[1] > 0
                        ],
                        loss=False,
                    )
                    for x in previous_text_tokens
                ]

                # Calculate n_segments, so that the last_segment token is applied on time in the syntax function.
                # in case this is the last segment: min(n_segments, 1+n_context segments)
                # otherwise, 2 + n_context segments (no last segment token)
                n_total_segments_nonzero_adjusted = torch.where(  # type: ignore
                    i == n_total_segments_nonzero - 1,  # type: ignore
                    torch.clamp(n_total_segments_nonzero, max=1 + n_context_segments),
                    2 + n_context_segments,
                )

                # Get tokens for this segment
                segment_tokens, segment_token_seq_lens = (
                    self.generate_hypotheses_one_segment_streaming(
                        new_audio_embeddings=current_audio_embeddings_nonzero,
                        new_audio_embedding_seq_lens=current_audio_embedding_seq_lens_nonzero,
                        n_total_segments=n_total_segments_nonzero_adjusted,
                        langs=langs_nonzero,
                        previous_audio_embeddings=previous_audio_embeddings_nonzero,
                        previous_text_tokens=previous_text_tokens_nonzero,
                    )
                )

                # Revert nonzero mask
                if not nonzero_mask.all():
                    segment_tokens_full = torch.zeros(
                        [B, segment_tokens.size(1)],
                        device=device,
                        dtype=segment_tokens.dtype,
                    )
                    segment_tokens_full[nonzero_mask] = segment_tokens
                    segment_tokens = segment_tokens_full
                    segment_token_seq_lens_full = [0] * B
                    indices = nonzero_mask.nonzero()[:, 0]
                    for i, val in zip(indices, segment_token_seq_lens):
                        segment_token_seq_lens_full[i] = val  # type: ignore
                    segment_token_seq_lens = segment_token_seq_lens_full

                # Add results to use as context for next segment
                previous_audio_embeddings.append(
                    ModalityInput(
                        modality=Modality.AUDIO,
                        seqs=current_audio_embeddings,
                        seq_lens=current_audio_embedding_seq_lens,
                        loss=False,
                    ),
                )
                previous_text_tokens.append(
                    ModalityInput(
                        modality=Modality.TEXT,
                        seqs=segment_tokens,
                        seq_lens=(
                            list(segment_token_seq_lens)
                            if segment_token_seq_lens is not None
                            else [0] * len(segment_tokens)
                        ),
                        loss=False,
                    ),
                )

            # Merge all segment tokens
            token_lengths: List[torch.Tensor] = [
                torch.tensor(inp.seq_lens) for inp in previous_text_tokens
            ]

            total_token_lengths: torch.Tensor = sum(token_lengths)  # type: ignore
            max_length_ = int(total_token_lengths.int().max().item())
            B = current_audio_embeddings.size(0)  # type: ignore
            device = current_audio_embeddings.device  # type: ignore
            dtype = previous_text_tokens[0].seqs.dtype

            assert self.pad_idx is not None, "PAD token must be specified"
            final_tokens = torch.full(
                fill_value=self.pad_idx,
                size=[B, max_length_],
                device=device,
                dtype=dtype,
            )
            for b in range(B):
                b_tokens_ = [
                    tokens.seqs[b, : length[b]]
                    for (tokens, length) in zip(previous_text_tokens, token_lengths)
                ]
                b_tokens = torch.cat(b_tokens_, dim=0)
                final_tokens[b, : b_tokens.size(0)] = b_tokens

            final_token_lens = total_token_lengths.tolist()

        else:
            assert decoder_context_inputs is not None
            assert decoder_context_seq_lens is not None
            # Nonstreaming - treat everything as a single segment.
            final_tokens, final_token_lens = self.generate_hypotheses_one_segment(
                decoder_context_inputs=decoder_context_inputs[0],
                decoder_context_seq_lens=decoder_context_seq_lens[0],
            )

        return final_tokens, final_token_lens

    @torch.no_grad()
    def generate_hypotheses_one_segment_streaming(
        self,
        new_audio_embeddings: torch.Tensor,
        new_audio_embedding_seq_lens: List[int],
        n_total_segments: Tensor,
        langs: List[str],
        previous_audio_embeddings: List[ModalityInput],
        previous_text_tokens: List[ModalityInput],
    ) -> tuple[Tensor, List[int]]:
        """Decode a single segment using previous segments as context.

        Builds streaming syntax with limited context window, caches embeddings,
        and decodes current segment conditioned on previous outputs.
        """
        # Limit to n_context_segments
        n_max_segments = self.model.streaming_config.n_context_segments
        previous_audio_embeddings = previous_audio_embeddings[-n_max_segments:]
        previous_text_tokens = previous_text_tokens[-n_max_segments:]

        # Add to previous segments
        previous_audio_embeddings.append(
            ModalityInput(
                modality=Modality.AUDIO,
                seqs=new_audio_embeddings,
                seq_lens=new_audio_embedding_seq_lens,
                loss=False,
            ),
        )
        # add a placeholder
        previous_text_tokens.append(
            ModalityInput(
                modality=Modality.TEXT, seqs=torch.empty(1, 0), seq_lens=[0], loss=False
            )
        )

        # Create batch
        device = new_audio_embeddings.device
        B = new_audio_embeddings.size(0)
        batch = Seq2SeqBatch(
            source_seqs=new_audio_embeddings,  # Not used for streaming inference
            source_seq_lens=new_audio_embedding_seq_lens,  # Not used for streaming inference
            target_seqs=torch.tensor(
                [1] * B, dtype=torch.long, device=device
            ).unsqueeze(
                1
            ),  # Not used for inference
            target_seq_lens=list([1] * B),
            example={
                "audio_segments": previous_audio_embeddings,
                "token_segments": previous_text_tokens,
                "n_segments": n_total_segments,
            },
        )
        if langs is not None:
            batch.example["lang"] = langs  # type: ignore

        # Create syntax
        inputs = self.model.create_streaming_syntax(
            batch=batch,
            device=device,
            inference=True,
        )

        # Set audio embeddings and mark them as embedded
        loc = 0
        for inp in inputs:
            if inp.modality == Modality.AUDIO:
                inp.seqs = previous_audio_embeddings[loc].seqs
                inp.seq_lens = previous_audio_embeddings[loc].seq_lens
                loc += 1
                inp.embedded = True
        assert loc == len(previous_audio_embeddings)

        # Embed all other inputs except audio
        inputs = self.model.embed_inputs(
            inputs=inputs, dtype=new_audio_embeddings.dtype
        )

        # Concat decoder inputs
        decoder_inputs, decoder_input_layout, _, _, _ = self.model.concat_inputs(
            inputs=inputs
        )

        # Run beam search
        final_tokens, final_token_lens = self.generate_hypotheses_one_segment(
            decoder_context_inputs=decoder_inputs,
            decoder_context_seq_lens=list(decoder_input_layout.seq_lens),
        )
        return final_tokens, final_token_lens

    @torch.no_grad()
    def generate_hypotheses_one_segment(
        self,
        decoder_context_inputs: Tensor,
        decoder_context_seq_lens: List[int],
    ) -> Tuple[Tensor, List[int]]:
        """Generate hypotheses using beam search with context prefilling.

        Expands each batch element into nbest beams. Prefills decoder inputs with
        embedded context (audio, prompt, etc.) up to and including BOS token.

        Generation loop:
        - Decode latest token (BOS initially, then generated tokens)
        - Add log probabilities to beam scores
        - Select top nbest hypotheses
        - Freeze scores for beams that emitted EOS
        - Stop when all nbest beams in all batch elements have emitted EOS

        Returns top hypothesis tokens and their lengths per batch element.
        """
        # Some init
        B = decoder_context_inputs.size(0)
        device = decoder_context_inputs.device
        dtype = decoder_context_inputs.dtype
        nbest = self.config.nbest
        ex_separator = torch.arange(B, device=device).unsqueeze(1) * nbest

        # Prepare a decoder input matrix, prefill with context
        decoder_inputs = torch.zeros(
            [
                B * nbest,
                self.model.max_generation_length,
                self.model.model_dim,
            ],
            device=device,
            dtype=dtype,
        )
        decoder_inputs[:, : decoder_context_inputs.size(1)] = (
            decoder_context_inputs.repeat_interleave(nbest, dim=0)
        )
        context_lengths = torch.tensor(
            decoder_context_seq_lens, device=device
        ).repeat_interleave(nbest)

        # Prepare a token self matrix and a scores matrix
        assert self.pad_idx is not None, "`pad_idx` must be specified"
        out_tokens = torch.full_like(
            decoder_inputs[:, :, 0],
            fill_value=self.pad_idx,
            dtype=torch.int,
        )
        scores = torch.zeros_like(decoder_inputs[:, 0, 0], dtype=torch.float) - 1e6
        scores[::nbest] = 0.0

        # Prefill with shortest context, keep state
        state_bag = IncrementalStateBag(max_num_steps=self.model.max_generation_length)
        min_context_len = int(context_lengths.min()) - 1  # remove double BOS input
        prefill_seqs = decoder_inputs[:, :min_context_len]
        prefill_seq_lens = [min_context_len] * B * nbest
        sliced_decoder_layout = BatchLayout.of(prefill_seqs, prefill_seq_lens)
        _ = self.model.llama_decoder(
            seqs=prefill_seqs,
            seqs_layout=sliced_decoder_layout,
            state_bag=state_bag,
        )
        state_bag.increment_step_nr(min_context_len)

        # Iterative decoding:
        # Start decoding after the shortest context in the batch. Samples with longer
        # context will be ignored until their context has been consumed.
        # For each sample, choose either context, or emitted text embedding.
        # If EOS is emitted, the sample is non-active. Stop when there are no active samples.
        eos_mask = torch.zeros_like(context_lengths, dtype=torch.bool)
        done = False
        t = context_lengths.min() - 1
        while not done:
            # Run the decoder on mixed context and emitted text embeddings
            iterative_seqs = decoder_inputs[:, t : t + 1]
            iterative_seq_lens = [1] * B * nbest
            iterative_seqs_layout = BatchLayout.of(iterative_seqs, iterative_seq_lens)
            dec_out = self.model.llama_decoder(
                seqs=iterative_seqs,
                seqs_layout=iterative_seqs_layout,
                state_bag=state_bag,
            )
            state_bag.increment_step_nr(1)
            logits = self.model.final_proj(dec_out).squeeze(1)  # [B * nbest, V]
            log_probs = F.log_softmax(logits, dim=-1)

            # Choose nbest
            if self.config.length_norm:
                n_tokens = torch.logical_and(
                    out_tokens[:, :t] != self.pad_idx,
                    out_tokens[:, :t] != self.eos_idx,  # type: ignore[arg-type]
                ).sum(dim=1, keepdim=True)
                if n_tokens[0, 0] > 0:
                    candidate_scores = (scores.unsqueeze(1) * n_tokens + log_probs) / (
                        n_tokens + 1
                    )
                else:
                    candidate_scores = scores.unsqueeze(1) + log_probs
            else:
                candidate_scores = scores.unsqueeze(1) + log_probs  # [B * nbest, V]

            candidate_scores[eos_mask] = -torch.inf
            candidate_scores[eos_mask, self.eos_idx] = scores[
                eos_mask
            ]  # Don't change scores for ended hypos

            top_scores, top_idx = candidate_scores.view(B, -1).topk(
                k=nbest, dim=-1, sorted=True
            )
            top_idx_nbest, top_idx_v = self.idx_1d_to_2d(
                top_idx, candidate_scores.size(-1)
            )
            top_idx_b = (top_idx_nbest + ex_separator).view(-1)  # Parent hypos indices

            # Reorder some tensors based on parent hypos
            out_tokens = out_tokens[top_idx_b]
            eos_mask = eos_mask[top_idx_b]
            scores = scores[top_idx_b]
            state_bag.reorder(top_idx_b)
            scores = torch.where(eos_mask, scores, top_scores.view(-1))  # [N * nbest]
            out_tokens[:, t] = top_idx_v.view(-1)

            # For hypos that still don't emit tokens, set new tokens to pad_idx, score to 0.
            no_token_mask = t < context_lengths - 1
            out_tokens[no_token_mask, t] = self.pad_idx
            scores[no_token_mask] = 0.0

            # For hypos that had EOS previously, set new tokens to EOS. Scores don't change.
            # Set new EOS mask.
            assert self.eos_idx is not None, "`eos_idx` must be set"
            out_tokens[eos_mask, t] = self.eos_idx
            new_tokens = out_tokens[:, t : t + 1]
            eos_mask = (new_tokens == self.eos_idx).squeeze(1)

            # Run new tokens through frontend, set in decoder input
            new_tokens_embedded = self.model.embed_text(new_tokens, dtype=dtype)
            decoder_inputs[~no_token_mask, t + 1] = (
                new_tokens_embedded[~no_token_mask].to(decoder_inputs.dtype).squeeze(1)
            )  # Don't override audio encoder outputs

            # Early stopping if emitting repeating characters, use compression ratio
            # only every t, only when started emitting tokens more than T tokens ago
            if t % 250 == 0:
                cpu_tokens = (
                    out_tokens[:, t - self.config.compression_window : t].cpu().numpy()
                )
                ratios_floats = [
                    self.compression_ratio(
                        np.array_str(cpu_tokens[i]).replace("\n", "")
                    )
                    for i in range(B * nbest)
                ]
                ratios = torch.tensor(ratios_floats, device=device)
                early_stopping_mask = torch.logical_and(
                    ratios > self.config.compression_threshold,
                    t > context_lengths + self.config.compression_window,
                )
                eos_mask = torch.logical_or(eos_mask, early_stopping_mask)

            # Decide if we are done
            done = bool(
                torch.logical_or(
                    torch.all(eos_mask),
                    t == self.model.max_generation_length - 4,
                )
            )
            t += 1

        # Get final tokens, only use top hypo
        out_tokens = out_tokens[::nbest]
        valid_tokens_mask = torch.logical_and(
            torch.logical_and(
                out_tokens != self.pad_idx,
                out_tokens != self.bos_idx,  # type: ignore[arg-type]
            ),
            out_tokens != self.eos_idx,  # type: ignore[arg-type]
        )
        valid_tokens_count = valid_tokens_mask.sum(dim=1)
        final_tokens = torch.full(
            [B, int(valid_tokens_count.max())],
            fill_value=self.pad_idx,
            dtype=torch.int64,
            device=device,
        )
        for i in range(B):
            final_tokens[i, : valid_tokens_count[i]] = out_tokens[i][
                valid_tokens_mask[i]
            ]

        return final_tokens, valid_tokens_count.tolist()
