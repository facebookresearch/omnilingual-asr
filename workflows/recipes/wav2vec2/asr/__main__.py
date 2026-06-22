# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from fairseq2.recipe.cli import train_main

import sys

from .recipe import Wav2Vec2AsrRecipe
from .recipe_lora import Wav2Vec2LoraAsrRecipe

if "--lora" in sys.argv:
  recipe = Wav2Vec2LoraAsrRecipe()
  sys.argv.remove("--lora")
else:
  recipe = Wav2Vec2AsrRecipe()

train_main(recipe)
