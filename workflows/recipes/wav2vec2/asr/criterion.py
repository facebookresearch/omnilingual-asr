# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Tuple, final

from fairseq2.datasets import Seq2SeqBatch
from fairseq2.metrics import MetricBag
from fairseq2.nn import BatchLayout
from fairseq2.recipe.model import RecipeModel
from torch import Tensor

from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

# isort: split

from .metrics import add_asr_metrics, update_asr_batch_metrics, update_ctc_loss
from .wer_calculator import WerCalculator


@final
class Wav2Vec2AsrCriterion:
    """ASR training criterion with optional WER computation.

    Computes CTC loss for training. When `wer_calculator` is provided (validation/evaluation),
    also computes WER metrics by decoding model outputs.
    """

    _model: RecipeModel
    _wer_calculator: WerCalculator | None
    _llama_beam_search: Wav2Vec2LlamaBeamSearchSeq2SeqGenerator | None

    def __init__(
        self,
        model: RecipeModel,
        wer_calculator: WerCalculator | None = None,
        llama_beam_search: Wav2Vec2LlamaBeamSearchSeq2SeqGenerator | None = None,
    ) -> None:
        self._model = model
        self._wer_calculator = wer_calculator
        self._llama_beam_search = llama_beam_search

    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        add_asr_metrics(metric_bag)

    def __call__(
        self,
        batch: Seq2SeqBatch,
        metric_bag: MetricBag,
    ) -> Tuple[Tensor, int]:
        """
        Compute CTC loss and optionally WER for validation/evaluation.

        :param batch: The input batch containing audio and text.
        :param metric_bag: The metric bag to update.

        :returns: The CTC loss and batch size.
        """
        if self._wer_calculator is not None:
            # Validation path - compute both loss and WER
            ctc_loss, logits, logit_layout, context_logits, context_logit_layout = (
                self._forward_with_logits(batch)
            )
            self._wer_calculator.compute_wer(
                batch,
                logits,
                logit_layout,
                context_logits,
                context_logit_layout,
                metric_bag,
                self._llama_beam_search,
            )
        else:
            # Training path - only compute loss
            ctc_loss = self._forward(batch)  # type: ignore

        update_ctc_loss(metric_bag, ctc_loss, batch.batch_size)  # type: ignore
        update_asr_batch_metrics(metric_bag, batch)

        return ctc_loss, batch.batch_size  # type: ignore

    def _forward_with_logits(
        self, batch: Seq2SeqBatch
    ) -> Tuple[Tensor, Tensor, BatchLayout, Tensor | None, BatchLayout | None]:
        """
        :return: loss, logits, layout, context_logits (optional), context_layout (optional)
        """
        if isinstance(self._model.module, Wav2Vec2LlamaModel):
            # Llama model requires batch.extras for constructing the input batch
            return self._model.module(
                batch, return_logits=True
            )  # type: ignore[call-overload]
        else:
            source_seqs, source_seqs_layout = batch.as_source_input()  # Audio
            target_seqs, target_seqs_layout = batch.as_target_input()  # Text tokens

            loss, logits, logits_layout = self._model.module(
                source_seqs,
                source_seqs_layout,
                target_seqs,
                target_seqs_layout,
                return_logits=True,
            )
            return loss, logits, logits_layout, None, None

    def _forward(self, batch: Seq2SeqBatch) -> Tensor | Tuple[Tensor, BatchLayout]:
        """
        :return: Tensor (default) - CTC Loss
        :return: tuple[Tensor, BatchLayout] (if target_seqs == None) - logits, layout
        """
        if isinstance(self._model.module, Wav2Vec2LlamaModel):
            # Llama model requires batch.extras for constructing the input batch
            return self._model.module(batch)  # type: ignore
        else:
            source_seqs, source_seqs_layout = batch.as_source_input()  # Audio
            target_seqs, target_seqs_layout = batch.as_target_input()  # Text tokens

            return self._model.module(
                source_seqs,
                source_seqs_layout,
                target_seqs,
                target_seqs_layout,
            )

    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        if self._wer_calculator:
            self._wer_calculator.process_metric_values(values)

    @property
    def model(self) -> RecipeModel:
        return self._model
