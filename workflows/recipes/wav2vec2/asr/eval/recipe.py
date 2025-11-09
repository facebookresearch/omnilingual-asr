# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import final

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import Seq2SeqBatch, SyncMode
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.recipe.base import EvalRecipe, RecipeContext
from fairseq2.recipe.evaluator import Evaluator, EvalUnit
from fairseq2.recipe.model import RecipeModel
from fairseq2.runtime.dependency import DependencyContainer
from typing_extensions import override

from omnilingual_asr.datasets.impl.manifest_asr_dataset import (
    MANIFEST_ASR_DATASET,
    ManifestAsrDataset,
    ManifestAsrDatasetConfig,
    open_manifest_asr_dataset,
)
from omnilingual_asr.datasets.impl.mixture_parquet_asr_dataset import (
    MIXTURE_PARQUET_ASR_DATASET,
    MixtureParquetAsrDataset,
    MixtureParquetAsrDatasetConfig,
    open_mixture_parquet_asr_dataset,
)
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

from ..criterion import Wav2Vec2AsrCriterion
from ..dataset_selector import Wav2Vec2AsrDatasetSelector
from ..wer_calculator import WerCalculator
from .default_config import Wav2Vec2AsrEvalRecipeConfig


@final
class Wav2Vec2AsrEvalRecipe(EvalRecipe):
    @override
    def register(self, container: DependencyContainer) -> None:
        register_dataset_family(
            container,
            MANIFEST_ASR_DATASET,
            ManifestAsrDataset,
            ManifestAsrDatasetConfig,
            opener=open_manifest_asr_dataset,
        )
        register_dataset_family(
            container,
            MIXTURE_PARQUET_ASR_DATASET,
            MixtureParquetAsrDataset,
            MixtureParquetAsrDatasetConfig,
            opener=open_mixture_parquet_asr_dataset,
        )

    @override
    def create_evaluator(self, context: RecipeContext) -> Evaluator:
        config = context.config.as_(Wav2Vec2AsrEvalRecipeConfig)

        dataset, storage_config, task_config = (
            Wav2Vec2AsrDatasetSelector.get_dataset_and_configs(config, context)
        )

        if isinstance(context.model.base_module, Wav2Vec2LlamaModel):
            log.info("Detected LLama model, using beamsearch during evaluation.")
            llama_beam_search = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator.from_context(
                context
            )
        else:
            log.info(
                f"No LLama Model, instead {context.model.base_module}. Using greedy decoding during evaluation."
            )
            llama_beam_search = None

        valid_criterion = Wav2Vec2AsrCriterion(
            model=context.model,
            wer_calculator=WerCalculator.from_context(context),
            llama_beam_search=llama_beam_search,
        )

        if config.dataset.valid_split is None:
            raise ValueError(
                "Wav2Vec2AsrEvalDatasetConfig.valid_split must be defined for evaluation but is `None`."
            )

        # Initialize validation units
        eval_units = []
        eval_data_readers = []
        eval_splits = config.dataset.valid_split.split(",")

        for split in eval_splits:
            eval_unit = Wav2Vec2AsrEvalUnit(valid_criterion)
            eval_units.append(eval_unit)

            storage_config.sync_mode = SyncMode.UNTIL_LAST
            eval_data_reader = dataset.create_reader(
                split=split,
                tokenizer=context.default_tokenizer,
                gangs=context.gangs,
                dtype=config.evaluator.amp_dtype,
                num_accumulate=1,
                storage_config=storage_config,  # type: ignore
                task_config=task_config,  # type: ignore
            )

            eval_data_readers.append(eval_data_reader)

        return context.create_evaluator(eval_units, eval_data_readers)

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrEvalRecipeConfig


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """
    wav2vec2 ASR evaluation unit.
    """

    _criterion: Wav2Vec2AsrCriterion

    def __init__(self, criterion: Wav2Vec2AsrCriterion) -> None:
        self._criterion = criterion

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)
        metric_bag.add("wer", WerMetric())

    @override
    def process_batch(self, batch: Seq2SeqBatch, metric_bag: MetricBag) -> None:
        self._criterion(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        return self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model
