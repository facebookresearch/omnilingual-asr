# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs


def test_supported_langs():
    assert isinstance(supported_langs, list), "supported_langs should be a list"
    assert len(supported_langs) > 0, "supported_langs should not be empty"
    assert (
        len(supported_langs) >= 1600
    ), f"Expected at least 1600 languages, got {len(supported_langs)}"

    for lang in supported_langs:
        assert isinstance(lang, str), f"Language code {lang} is not a string"
        assert "_" in lang, f"Language code {lang} does not follow format 'code_script'"
        parts = lang.split("_")
        assert len(parts) in [
            2,
            3,
        ], f"Language code {lang} should have exactly one underscore"
        assert parts[0].islower(), f"Language code {parts[0]} should be lowercase"
        assert parts[1][
            0
        ].isupper(), f"Script code {parts[1]} should start with uppercase"

    expected_languages = [
        "eng_Latn",
        "spa_Latn",
        "fra_Latn",
        "deu_Latn",
        "cmn_Hans",
        "arb_Arab",
        "hin_Deva",
        "por_Latn",
        "rus_Cyrl",
        "jpn_Jpan",
    ]
    for lang in expected_languages:
        assert lang in supported_langs, f"Expected language {lang} not found"

    unique_langs = set(supported_langs)
    assert len(unique_langs) == len(supported_langs), "Duplicate language codes found"

    assert supported_langs == sorted(
        supported_langs
    ), "Language codes should be sorted alphabetically"
