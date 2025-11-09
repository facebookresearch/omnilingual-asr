# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast, final

from fairseq2.composition import register_dataset_family
from fairseq2.datasets import Seq2SeqBatch, SyncMode
from fairseq2.error import OperationalError
from fairseq2.gang import GangError
from fairseq2.logging import log
from fairseq2.metrics import MetricBag
from fairseq2.metrics.text import WerMetric
from fairseq2.models.wav2vec2 import Wav2Vec2Model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.utils.module import freeze_parameters, share_parameters, to_device
from fairseq2.recipe.base import RecipeContext, TrainRecipe
from fairseq2.recipe.evaluator import EvalUnit
from fairseq2.recipe.model import RecipeModel
from fairseq2.recipe.trainer import Trainer, TrainUnit
from fairseq2.runtime.dependency import DependencyContainer
from torch import Tensor
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

from .criterion import Wav2Vec2AsrCriterion
from .dataset_selector import Wav2Vec2AsrDatasetSelector
from .default_config import Wav2Vec2AsrRecipeConfig
from .wer_calculator import WerCalculator


@final
class Wav2Vec2AsrRecipe(TrainRecipe):
    """wav2vec2 ASR training recipe."""

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

    def prepare_training_from_encoder(
        self, context: RecipeContext, asr_module: Wav2Vec2AsrModel
    ):
        w2v2_model = context.bootstrap_model("pretrained_encoder")
        w2v2_module = cast(Wav2Vec2Model, w2v2_model.module)

        share_parameters(w2v2_module.encoder_frontend, asr_module.encoder_frontend)  # type: ignore
        share_parameters(w2v2_module.encoder, asr_module.encoder)  # type: ignore
        if asr_module.masker is not None:
            share_parameters(w2v2_module.masker, asr_module.masker)  # type: ignore

        del w2v2_model

    @override
    def prepare_model(self, context: RecipeContext, model: RecipeModel) -> RecipeModel:
        """Initialize ASR model depending on the configuration.

        (1) model.name -> training from a pre-trained CTC/LLM model
        (2) pretrained_encoder.name + model.name=="" -> training with shared encoder_frontend, encoder and optional masker
        """
        config = context.config.as_(Wav2Vec2AsrRecipeConfig)
        asr_module = cast(Wav2Vec2AsrModel, model.module)

        # fmt:off
        # considering None and empty string to be undefined
        model_defined              = not (config.model.name == ""              or config.model.name is None)
        pretrained_encoder_defined = not (config.pretrained_encoder.name == "" or config.pretrained_encoder.name is None)
        training_fresh_asr_model       = pretrained_encoder_defined and not model_defined
        training_from_pretrained_model = model_defined

        if training_fresh_asr_model:
            log.info("Initializing the ASR model with pretrained SSL model (encoder only).")
            self.prepare_training_from_encoder(context=context, asr_module=asr_module)
        elif training_from_pretrained_model:
            log.info("Found ASR checkpoint, starting training.")
        else:
            raise ValueError("Recipe configuration is invalid. Supported configurations: [model.name | model.arch + pretrained_encoder.name]")
        # fmt: on

        # Make sure that the final projection layer is instantiated along with
        # the pretrained parameters if it was on the meta device.
        if context.gangs.dp.rank == 0:
            to_device(asr_module, context.gangs.root.device)
        try:
            context.gangs.root.barrier()
        except GangError as ex:
            raise OperationalError(
                "The collective barrier after the pretrained model load operation has failed. See the nested exception for details."
            ) from ex

        # Always freeze feature extractor
        freeze_parameters(asr_module.encoder_frontend.feature_extractor)

        return model

    @override
    def create_trainer(self, context: RecipeContext) -> Trainer:
        """Main recipe entry point."""
        config = context.config.as_(Wav2Vec2AsrRecipeConfig)

        dataset, storage_config, task_config = (
            Wav2Vec2AsrDatasetSelector.get_dataset_and_configs(config, context)
        )
        criterion = Wav2Vec2AsrCriterion(context.model)

        unit = Wav2Vec2AsrTrainUnit(
            criterion, config.trainer.freeze_encoder_for_n_steps
        )

        if config.dataset.train_split is None:
            raise ValueError(
                "Wav2Vec2AsrDatasetConfig.train_split must be defined for training but is `None`."
            )

        task_config.seed = config.common.seed

        data_reader = dataset.create_reader(
            split=config.dataset.train_split,
            tokenizer=context.default_tokenizer,
            gangs=context.gangs,
            dtype=config.trainer.mixed_precision.dtype,
            num_accumulate=config.trainer.grad_accumulation.num_batches,
            storage_config=storage_config,  # type: ignore
            task_config=task_config,  # type: ignore
        )

        valid_units = []
        valid_data_readers = []

        if config.dataset.valid_split is not None:
            # Support multiple validation splits
            valid_splits = config.dataset.valid_split.split(",")
            if isinstance(context.model.base_module, Wav2Vec2LlamaModel):
                log.info("Detected LLama model, using beamsearch during evaluation.")
                llama_beam_search = (
                    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator.from_context(context)
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
            for split in valid_splits:
                valid_unit = Wav2Vec2AsrEvalUnit(valid_criterion)
                valid_units.append(valid_unit)

                # validation specific settings
                task_config.batch_shuffle_window = 1
                task_config.seed = config.common.seed + 1
                storage_config.sync_mode = SyncMode.UNTIL_LAST

                valid_data_reader = dataset.create_reader(
                    split=split,
                    tokenizer=context.default_tokenizer,
                    gangs=context.gangs,
                    dtype=config.trainer.mixed_precision.dtype,
                    storage_config=storage_config,  # type: ignore
                    task_config=task_config,  # type: ignore
                )
                valid_data_readers.append(valid_data_reader)

        return context.create_trainer(
            unit, data_reader, valid_units, valid_data_readers
        )

    @property
    @override
    def config_kls(self) -> type[object]:
        return Wav2Vec2AsrRecipeConfig


@final
class Wav2Vec2AsrTrainUnit(TrainUnit[Seq2SeqBatch]):
    """ASR training unit with encoder freezing logic."""

    _criterion: Wav2Vec2AsrCriterion
    _freeze_encoder_for_n_steps: int
    _frozen: bool

    def __init__(
        self, criterion: Wav2Vec2AsrCriterion, freeze_encoder_for_n_steps: int
    ) -> None:
        """
        :param criterion: The ASR criterion holding the model.
        :param freeze_encoder_for_n_steps: The encoder will be frozen for this number of steps.
        """
        self._criterion = criterion
        self._freeze_encoder_for_n_steps = freeze_encoder_for_n_steps
        self._frozen = False

    @override
    def set_step_nr(self, step_nr: int) -> None:
        """Gradually unfreeze encoder during training for stability.
        Freezes encoder/masker for first N steps, then unfreezes while keeping feature extractor frozen.
        """
        base_module = cast(Wav2Vec2AsrModel, self._criterion.model.base_module)

        if step_nr <= self._freeze_encoder_for_n_steps:
            if self._frozen:
                return

            if step_nr == 1:
                log.info(
                    f"Freezing the encoder for the first {self._freeze_encoder_for_n_steps} steps."
                )

            # Freeze encoder components
            freeze_parameters(base_module.encoder_frontend)
            freeze_parameters(base_module.encoder)

            if base_module.masker is not None:
                freeze_parameters(base_module.masker)

            self._frozen = True
        else:
            if not self._frozen:
                return

            if step_nr == self._freeze_encoder_for_n_steps + 1:
                log.info(f"Unfreezing the encoder after step {step_nr - 1}.")

            # Unfreeze all parameters
            freeze_parameters(base_module, False)

            # Always keep feature extractor frozen
            freeze_parameters(base_module.encoder_frontend.feature_extractor)

            self._frozen = False

    @override
    def prepare_metric_bag(self, metric_bag: MetricBag) -> None:
        self._criterion.prepare_metric_bag(metric_bag)

    @override
    def process_batch(
        self, batch: Seq2SeqBatch, metric_bag: MetricBag
    ) -> tuple[Tensor, int]:
        return self._criterion(batch, metric_bag)

    @override
    def process_metric_values(self, values: MutableMapping[str, object]) -> None:
        """Accessor function to modify the logged values"""
        return self._criterion.process_metric_values(values)

    @property
    @override
    def model(self) -> RecipeModel:
        return self._criterion.model


@final
class Wav2Vec2AsrEvalUnit(EvalUnit[Seq2SeqBatch]):
    """ASR evaluation unit for validation during training."""

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
