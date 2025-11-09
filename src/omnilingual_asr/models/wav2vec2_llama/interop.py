# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Dict

from fairseq2.models.wav2vec2.asr import convert_wav2vec2_asr_state_dict

from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaConfig


def convert_wav2vec2_llama_state_dict(
    state_dict: Dict[str, object], config: Wav2Vec2LlamaConfig
) -> Dict[str, object]:
    """Using the fs2:wav2vec2_asr implementation until breaking changes arise. Replace with custom loading if necessary."""

    return convert_wav2vec2_asr_state_dict(
        state_dict=state_dict, config=config.wav2vec2_asr_config
    )
