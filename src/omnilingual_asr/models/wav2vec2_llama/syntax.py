# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum

import torch
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.device import Device
from fairseq2.logging import get_log_writer
from fairseq2.nn import BatchLayout
from torch import Tensor

log = get_log_writer(__name__)


class Modality(Enum):
    """Input modality types for multi-modal model."""

    AUDIO = "audio"
    TEXT = "text"
    LANG = "lang"


@dataclass
class ModalityInput:
    """Multi-modal input for concatenation into the decoder input.

    Represents a batched input where seqs is [B, S, D] and seq_lens has B elements.
    """

    modality: Modality
    seqs: Tensor
    seq_lens: list[int]
    loss: bool
    embedded: bool = False

    def to_batch_layout(self) -> BatchLayout:
        """Convert to BatchLayout for encoder/decoder calls."""
        return BatchLayout.of(self.seqs, self.seq_lens)

    @property
    def batch_size(self) -> int:
        """Number of examples in this batch."""
        return self.seqs.size(0)

    @property
    def device(self) -> Device:
        """Device where tensors live."""
        return self.seqs.device


def create_lang_inputs(
    batch: Seq2SeqBatch,
    lid_marker: int,
    lang_mapping: dict[str, int],
    lang_column_name: str,
    dropout_mask: Tensor | None,
    device: Device,
) -> tuple[ModalityInput, ModalityInput]:
    """Create language marker token and lang ID inputs.

    Returns special token marker and language ID with optional dropout applied.
    Dropout mask zeros lang_id where True (used during training).
    """

    batch_size = batch.source_seqs.size(0)

    # Create special token marker
    lid_marker_input = create_single_char_input(batch, lid_marker, device)

    # Create lang IDs
    lang_id = torch.zeros(batch_size, dtype=torch.int64, device=device)

    if isinstance(batch.example, dict) and lang_column_name in batch.example:
        langs = batch.example[lang_column_name]
        assert (
            len(langs) == batch_size
        ), f"lang_ids must match batch size ({batch_size})"

        for i, lang in enumerate(langs):
            lang_id[i] = lang_id_getter(lang_mapping, lang)

    # Apply dropout if mask provided (i.e. during training)
    if dropout_mask is not None:
        lang_id[dropout_mask] = 0

    lang_id_input = ModalityInput(
        modality=Modality.LANG,
        seqs=lang_id.unsqueeze(-1),
        seq_lens=[1] * batch_size,
        loss=False,
    )

    return lid_marker_input, lang_id_input


def lang_id_getter(lang_mapping: dict[str, int], lang: str) -> int:
    """Get the lang ID for a given language code (kept for BC)."""
    if lang.lower() in lang_mapping:
        return lang_mapping[lang.lower()]
    if lang in lang_mapping:
        return lang_mapping[lang]

    log.warning(f"lang not in mapping: {lang}")
    return 0


def create_single_char_input(
    batch: Seq2SeqBatch, char: int, device: Device, loss: bool = False
) -> ModalityInput:
    """Create text input containing a single special token per batch element.

    Returns [B, 1] tensor filled with the specified token index.
    """
    batch_size = batch.source_seqs.size(0)
    buffer_size = batch.target_seqs[:, :1]
    seq_lens = [1] * batch_size

    return ModalityInput(
        modality=Modality.TEXT,
        seqs=torch.full_like(buffer_size, fill_value=char, device=device),  # type: ignore
        seq_lens=seq_lens,
        loss=loss,
    )
