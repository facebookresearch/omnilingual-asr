# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2Config
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_omnilingual_asr_wav2vec2_asr_configs(
    container: DependencyContainer,
) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2AsrConfig)

    @arch("300m", advanced=True)
    def _300m_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # base_10h and large_lv60k are original wav2vec2 configurations living in fs2
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2475
        config.target_vocab_size = 9812

        return config

    @arch("1b", advanced=True)
    def _1b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config

    @arch("3b", advanced=True)
    def _3b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config

    @arch("7b", advanced=True)
    def _7b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "7b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config
