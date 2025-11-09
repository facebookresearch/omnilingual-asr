# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

from fairseq2.models.llama import LLaMAConfig
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrConfig
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver

WAV2VEC2_LLAMA_FAMILY: Final = "wav2vec2_llama"


class ModelType(Enum):
    LLM_ASR = 1
    LLM_ASR_LID = 2
    ZERO_SHOT = 3


@dataclass(kw_only=True)
class Wav2Vec2LlamaBeamSearchConfig:
    """Contains the settings for the LLM-ASR beam search."""

    nbest: int = 5
    """The size of the beam."""

    length_norm: bool = False
    """Whether we apply length normalization when computing hypothesis score."""

    compression_window: int = 100
    """For early stopping during decoding when inputs are bad, we try to compress the last `compression_window` tokens, and it the compression ration is larger than `compression_threshold`, we stop decoding."""

    compression_threshold: float = 4.0
    """See `compression_window`."""


@dataclass(kw_only=True)
class Wav2Vec2LlamaConfig:
    """Model configuration for the Wav2Vec2 Llama-based model.
    Holds individual configuration of Wav2Vec2Asr (frontend + encoder), Llama (decoder).
    """

    wav2vec2_asr_config: Wav2Vec2AsrConfig = field()
    """Wav2Vec2Asr configuration (frontend + encoder)."""

    llama_config: LLaMAConfig = field()
    """Llama configuration (decoder)."""

    beam_search_config: Wav2Vec2LlamaBeamSearchConfig = field(
        default_factory=lambda: Wav2Vec2LlamaBeamSearchConfig()
    )
    """Beam search configuration for LLM-ASR decoding."""

    encoder_stacking: int = 1
    """The number audio embeddings frames to stack before the decoder calls."""

    frozen_encoder: int = 1
    """Set 0: frozen, 1: unfrozen, N: unfrozen every N calls."""

    lang_embeddings_p: float = 0.0
    """The probability of dropping the language embeddings."""

    language_column_name: str = "lang"
    """The name of the column containing the language information."""

    context_text_only: bool = False
    """Adapts the model input syntax. Must be set to `True` if the context is text-only (instead of audio + text)."""

    n_special_tokens: int = 0
    """How many additional special tokens to allocate in the vocab-embd mapping."""

    # Migration note: VocabularyInfo was refactored away from Wav2Vec2AsrConfig and is now determined at the tokenizer level (>=fs2:v0.5)
    # Vocabulary size is defined in `wav2vec2_asr_config.target_vocab_size`.
    # Adding missing parameters to re-create VocabularyInfo object in the factory to keep model API stable.
    # Source: https://github.com/facebookresearch/fairseq2/blob/main_w2v2_pretraining/src/fairseq2/models/wav2vec2/asr/_config.py#L39
    unk_idx: int = 3
    """Defines the index of UNK token in the the target vocabulary."""

    bos_idx: int = 0
    """Defines the index of BOS token in the the target vocabulary."""

    eos_idx: int = 2
    """Defines the index of EOS token in the the target vocabulary."""

    pad_idx: int = 1
    """Defines the index of PAD token in the the target vocabulary. Note that it must align with the Llama's `pad_idx`."""

    boh_idx: int | None = None
    """Defines the index of the BOH (beginning of header) token in the target vocabulary."""

    eoh_idx: int | None = None
    """Defines the index of the EOH (end of header) token in the target vocabulary."""

    model_type: ModelType = ModelType.LLM_ASR
    """Defines the high-level model type."""

    n_context_examples: int = 0
    """For the zero-shot model, the number of context examples to use."""

    def __post_init__(self) -> None:
        # Check that target_vocabulary is aligned with llama_config
        if self.wav2vec2_asr_config.target_vocab_size != self.llama_config.vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: wav2vec2_asr_config.target_vocab_size "
                f"({self.wav2vec2_asr_config.target_vocab_size}) != "
                f"llama_config.vocab_size ({self.llama_config.vocab_size})"
            )

        if self.pad_idx != self.llama_config.pad_idx:
            raise ValueError(
                f"PAD token index mismatch: pad_idx "
                f"({self.pad_idx}) != "
                f"llama_config.pad_idx ({self.llama_config.pad_idx})"
            )


def register_wav2vec2_llama_configs(container: DependencyContainer) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2LlamaConfig)

    # Explicit default due to almost no config reuse
    default_pad_idx = Wav2Vec2LlamaConfig.pad_idx

    @arch("7b", advanced=True)
    def _7b_llama(resolver: DependencyResolver) -> Wav2Vec2LlamaConfig:
        wav2vec2_asr_config = get_config(resolver, Wav2Vec2AsrConfig, "7b")
        llama_config = LLaMAConfig(
            model_dim=4096,
            max_seq_len=8192,
            vocab_size=wav2vec2_asr_config.target_vocab_size,
            pad_idx=default_pad_idx,
            num_layers=12,
            num_attn_heads=8,
            num_key_value_heads=8,
            ffn_inner_dim=4096,
            rope_theta=10_000.0,
            dropout_p=0.1,
        )
        target_vocab_size = 9818
        llama_config.vocab_size = target_vocab_size
        llama_config.pad_idx = 1
        wav2vec2_asr_config.target_vocab_size = target_vocab_size

        config = Wav2Vec2LlamaConfig(
            wav2vec2_asr_config=wav2vec2_asr_config, llama_config=llama_config
        )
        config.lang_embeddings_p = 0.5
        config.n_special_tokens = 1
        config.model_type = ModelType.LLM_ASR_LID
        return config

    @arch("300m", advanced=True)
    def _300m_llama(
        resolver: DependencyResolver,
    ) -> Wav2Vec2LlamaConfig:
        config = _7b_llama(resolver)
        config.wav2vec2_asr_config = get_config(resolver, Wav2Vec2AsrConfig, "300m")
        vocab_size = 9812
        config.llama_config.vocab_size = vocab_size
        config.wav2vec2_asr_config.target_vocab_size = vocab_size
        return config

    @arch("1b", advanced=True)
    def _1b_llama(
        resolver: DependencyResolver,
    ) -> Wav2Vec2LlamaConfig:
        config = _7b_llama(resolver)
        config.wav2vec2_asr_config = get_config(resolver, Wav2Vec2AsrConfig, "1b")
        vocab_size = 9812
        config.llama_config.vocab_size = vocab_size
        config.wav2vec2_asr_config.target_vocab_size = vocab_size
        return config

    @arch("3b", advanced=True)
    def _3b_llama(
        resolver: DependencyResolver,
    ) -> Wav2Vec2LlamaConfig:
        config = _7b_llama(resolver)
        config.wav2vec2_asr_config = get_config(resolver, Wav2Vec2AsrConfig, "3b")
        vocab_size = 9812
        config.llama_config.vocab_size = vocab_size
        config.wav2vec2_asr_config.target_vocab_size = vocab_size
        return config

    @arch("7b_zs", advanced=True)
    def _7b_llama_zs(resolver: DependencyResolver) -> Wav2Vec2LlamaConfig:
        config = _7b_llama(resolver)
        config.llama_config.max_seq_len = 16384
        config.encoder_stacking = 3
        config.n_special_tokens = 6
        config.model_type = ModelType.ZERO_SHOT
        config.n_context_examples = 10
        config.lang_embeddings_p = 0.0
        vocab_size = 9812
        config.llama_config.vocab_size = vocab_size
        config.wav2vec2_asr_config.target_vocab_size = vocab_size
        return config
