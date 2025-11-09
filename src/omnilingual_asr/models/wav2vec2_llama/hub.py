# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.models import ModelHubAccessor

from omnilingual_asr.models.wav2vec2_llama.config import (
    WAV2VEC2_LLAMA_FAMILY,
    Wav2Vec2LlamaConfig,
)
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

get_wav2vec2_llama_model_hub = ModelHubAccessor(
    WAV2VEC2_LLAMA_FAMILY, Wav2Vec2LlamaModel, Wav2Vec2LlamaConfig
)
