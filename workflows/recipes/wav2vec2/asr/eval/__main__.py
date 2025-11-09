# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.recipe.cli import eval_main

from .recipe import Wav2Vec2AsrEvalRecipe

recipe = Wav2Vec2AsrEvalRecipe()

eval_main(recipe)
