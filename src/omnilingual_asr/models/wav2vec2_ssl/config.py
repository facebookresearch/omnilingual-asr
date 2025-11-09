# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models.wav2vec2 import Wav2Vec2Config
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_omnilingual_asr_wav2vec2_ssl_configs(
    container: DependencyContainer,
) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2Config)

    @arch("xlsr_base", advanced=True)
    def xlsr_base(resolver: DependencyResolver) -> Wav2Vec2Config:
        # large_lv60k is an original wav2vec2 configuration living in fs2
        config = get_config(resolver, Wav2Vec2Config, "large_lv60k")
        config.encoder_config.attn_dropout_p = 0.0
        return config

    @arch("1b", advanced=True)
    def _1b_ssl(resolver: DependencyResolver) -> Wav2Vec2Config:
        config = xlsr_base(resolver)

        config.encoder_config.model_dim = 1280
        config.encoder_config.num_encoder_layers = 48
        config.encoder_config.ffn_inner_dim = 5120
        config.encoder_config.dropout_p = 0.0
        config.quantized_dim = 1024
        config.final_dim = 1024
        config.encoder_config.first_pass_dropout_p = 0.1

        return config

    @arch("3b", advanced=True)
    def _3b_ssl(resolver: DependencyResolver) -> Wav2Vec2Config:
        config = _1b_ssl(resolver)

        config.encoder_config.num_encoder_layers = 60
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192

        return config

    @arch("7b", advanced=True)
    def _7b_ssl(resolver: DependencyResolver) -> Wav2Vec2Config:
        config = _3b_ssl(resolver)

        config.encoder_config.num_encoder_layers = 128
        config.encoder_config.model_dim = 2048
        config.encoder_config.ffn_inner_dim = 8192
        config.encoder_config.num_encoder_attn_heads = 16
        config.quantized_dim = 1024
        config.final_dim = 1024

        return config
