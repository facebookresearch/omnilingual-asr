# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import zlib
from typing import Tuple, final

import numpy as np
import torch
import torch.nn.functional as F
from fairseq2.logging import get_log_writer
from fairseq2.nn import BatchLayout, IncrementalStateBag
from fairseq2.recipe.base import RecipeContext
from torch import Tensor

from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaBeamSearchConfig
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

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
    ) -> None:
        self.model = model
        self.pad_idx = model.target_vocab_info.pad_idx
        self.eos_idx = model.target_vocab_info.eos_idx
        self.bos_idx = model.target_vocab_info.bos_idx
        self.config = config

    @classmethod
    def from_context(
        cls, context: RecipeContext
    ) -> "Wav2Vec2LlamaBeamSearchSeq2SeqGenerator":
        """Create generator from fairseq2 recipe context, extracting model and beam search config."""
        return cls(
            model=context.model.base_module,  # type: ignore
            config=context.model.base_module.beam_search_config,  # type: ignore
        )

    @staticmethod
    def idx_1d_to_2d(idx: Tensor, dim2: int) -> tuple[Tensor, Tensor]:
        return idx // dim2, idx % dim2

    @staticmethod
    def compression_ratio(text: str) -> float:
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))

    @torch.no_grad()
    def generate_hypotheses(
        self,
        decoder_context_inputs: Tensor,
        decoder_context_input_layout: BatchLayout,
    ) -> Tuple[Tensor, BatchLayout]:
        """
        Conducts a beam search to generate hypotheses for the LLM-ASR model. A high-level overview of the algorithm is as follows.
        The algorithm is described for a single example for simplicity, however the code may operate on a batch of examples for better perforamnce.

        Initialization and preparation:
        Prefill the decoder_input matrix with the embeddings of context already computed by the model, which is everything up to but
        exluding the <BOS> token. Feed the decoder the precomputed context.

        Generation Loop:
        Run the deocder on the latest generated token (or <BOS> at the first iterations), get emission scores.
        Add the log proability scores to the current scores of the hypotheses, choose the new `nbest` hypotheses. Don't change
        scores of hypos that already ended (emitted <EOS>). Decoding is done when all `nbest` hypotheses end with <EOS>.

        Args:
        decoder_context_inputs (Tensor):
            The input tensor containing the previously embded context / prompt for the decoder. Everything up to and including the <BOS> toekn.
        decoder_context_input_layout (BatchLayout):
            The layout object describing the above tensor, specifically, the lengths of each row in the batch.

        Returns:
            Tuple[Tensor, BatchLayout]:
                A tuple containing:
                    - hypotheses (Tensor): The generated tokens for the batch.
                    - batch_layout (BatchLayout): The layout object describing the length of the generated tokens in the batch.
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
        context_lengths = decoder_context_input_layout.seq_lens_pt.repeat_interleave(
            nbest
        )

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
                    out_tokens[:, :t] != self.pad_idx, out_tokens[:, :t] != self.eos_idx  # type: ignore[arg-type]
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
        final_layout = BatchLayout.of(final_tokens, valid_tokens_count.tolist())

        return final_tokens, final_layout
