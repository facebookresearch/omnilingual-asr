# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from fairseq2.data.tokenizers import VocabularyInfo
from fairseq2.logging import get_log_writer
from fairseq2.models.llama import LLaMAConfig, LLaMAFactory
from fairseq2.models.llama.factory import _init_truncated_normal
from fairseq2.models.transformer import (
    CausalAttentionBias,
    FeedForwardNetwork,
    GLUFeedForwardNetwork,
    MultiheadAttention,
    StandardMultiheadAttention,
    TransformerEncoder,
    TransformerNormOrder,
    create_default_sdpa,
)
from fairseq2.models.transformer_lm import (
    StandardTransformerLMDecoder,
    StandardTransformerLMDecoderLayer,
    TransformerLMDecoder,
    TransformerLMDecoderLayer,
)
from fairseq2.models.wav2vec2 import (
    StandardWav2Vec2Masker,
    Wav2Vec2EncoderFactory,
    Wav2Vec2Frontend,
    Wav2Vec2Masker,
)
from fairseq2.models.wav2vec2.asr.factory import _init_final_projection
from fairseq2.nn import Linear, PositionEncoder, StandardEmbedding

from omnilingual_asr.models.wav2vec2_llama.config import Wav2Vec2LlamaConfig
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel

log = get_log_writer(__name__)


LANG_LOOKUP_TABLE_PATH = str(Path(__file__).parent / "languges_lookup_table.parquet")


def create_wav2vec2_llama_model(
    config: Wav2Vec2LlamaConfig,
) -> Wav2Vec2LlamaModel:
    return Wav2Vec2LlamaFactory(config).create_model()


class OmnilingualASRLLamaFactory(LLaMAFactory):
    """
    The behavior of how ``config.dropout_p`` is used was refactored since fs2:v0.4.5,
    this subclass restores that functionality. Annotated the changes with comments.

    Adding ``config.dropout_p`` to:
    - ``create_default_sdpa.dropout_p``
    - ``GLUFeedForwardNetwork.inner_dropout_p``
    - ``StandardTransformerLMDecoder.dropout_p``

    Removing ``config.dropout_p`` from:
    - ``StandardTransformerLMDecoderLayer.dropout_p``
    """

    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__(config)

    def create_self_attention(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> MultiheadAttention:
        config = self._config

        attn_bias = CausalAttentionBias()
        # omnilingual_asr-specific change
        sdpa = create_default_sdpa(attn_bias, dropout_p=config.dropout_p)

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        return StandardMultiheadAttention(
            config.model_dim,
            config.num_attn_heads,
            sdpa,
            num_key_value_heads=config.num_key_value_heads,
            qkv_proj_init_fn=init_projection,
            pos_encoder=pos_encoder,
            output_proj_init_fn=init_projection,
            bias=False,
        )

    def create_decoder_layer(
        self, layer_idx: int, pos_encoder: PositionEncoder
    ) -> TransformerLMDecoderLayer:
        self_attn = self.create_self_attention(layer_idx, pos_encoder)

        self_attn_layer_norm = self.create_layer_norm()

        ffn = self.create_ffn(layer_idx)

        ffn_layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoderLayer(
            self_attn,
            self_attn_layer_norm,
            ffn,
            ffn_layer_norm,
            norm_order=TransformerNormOrder.PRE,
            # dropout_p=config.dropout_p,
        )

    def create_decoder(self) -> TransformerLMDecoder:
        config = self._config

        pos_encoder = self.create_position_encoder()

        layers = []

        for idx in range(config.num_layers):
            layer = self.create_decoder_layer(idx, pos_encoder)

            layers.append(layer)

        layer_norm = self.create_layer_norm()

        return StandardTransformerLMDecoder(
            layers,
            layer_norm,
            # omnilingual-asr-specific change
            dropout_p=config.dropout_p,
        )

    def create_ffn(self, layer_idx: int) -> FeedForwardNetwork:
        config = self._config

        init_std = config.init_std

        std_scale_factor = self.get_std_scale_factor(layer_idx)

        def init_projection(proj: Linear) -> None:
            input_dim = proj.weight.shape[1]

            std = init_std or (input_dim**-0.5)

            _init_truncated_normal(proj.weight, proj.bias, std=std / std_scale_factor)

        ffn_inner_dim = int(config.ffn_inner_dim * config.ffn_inner_dim_multiplier)

        return GLUFeedForwardNetwork(
            config.model_dim,
            ffn_inner_dim,
            bias=False,
            inner_dim_scale=config.ffn_inner_dim_scale,
            inner_dim_to_multiple=config.ffn_inner_dim_multiple_of,
            proj_init_fn=init_projection,
            # omnilingual_asr-specific change
            inner_dropout_p=config.dropout_p,
        )


class Wav2Vec2LlamaFactory:
    _config: Wav2Vec2LlamaConfig

    def __init__(
        self,
        config: Wav2Vec2LlamaConfig,
    ) -> None:
        self._config = config

    def create_encoder(self) -> tuple[Wav2Vec2Frontend, TransformerEncoder]:
        factory = Wav2Vec2EncoderFactory(
            self._config.wav2vec2_asr_config.encoder_config
        )
        return factory.create_encoder_frontend(), factory.create_encoder()

    def create_masker(self) -> Wav2Vec2Masker:
        config = self._config.wav2vec2_asr_config
        return StandardWav2Vec2Masker(
            config.encoder_config.model_dim,
            config.temporal_mask_span_len,
            config.max_temporal_mask_prob,
            config.min_num_temporal_mask_spans,
            config.spatial_mask_span_len,
            config.max_spatial_mask_prob,
            config.min_num_spatial_mask_spans,
        )

    def create_model(self) -> Wav2Vec2LlamaModel:
        encoder_frontend, encoder = self.create_encoder()
        masker = (
            self.create_masker()
            if self._config.wav2vec2_asr_config.use_masking
            else None
        )

        encoder_proj = Linear(
            self._config.wav2vec2_asr_config.encoder_config.model_dim
            * self._config.encoder_stacking,
            self._config.llama_config.model_dim,
            bias=True,
        )

        # Reserve some extra room for special tokens to construct syntax (zero-shot)
        n_special_tokens = self._config.n_special_tokens

        text_frontend = StandardEmbedding(
            num_embeddings=self._config.llama_config.vocab_size + n_special_tokens,
            embed_dim=self._config.llama_config.model_dim,
        )

        llama_decoder = OmnilingualASRLLamaFactory(
            self._config.llama_config
        ).create_decoder()

        final_proj = Linear(
            self._config.llama_config.model_dim,
            self._config.llama_config.vocab_size,
            bias=False,
            init_fn=_init_final_projection,
        )

        lang_mapping = None
        lang_embeddings = None
        if self._config.lang_embeddings_p > 0.0:
            import pyarrow.parquet as pq

            langage_lookup_table = pq.read_table(LANG_LOOKUP_TABLE_PATH).to_pylist()

            log.info(
                f"Found {len(langage_lookup_table)} languages from {LANG_LOOKUP_TABLE_PATH} table"
            )
            lang_mapping = {
                row["lang"].lower(): row["index"] + 1 for row in langage_lookup_table
            }

            # Create embeddings
            lang_embeddings = StandardEmbedding(
                num_embeddings=len(langage_lookup_table) + 1,
                embed_dim=self._config.llama_config.model_dim,
            )
        else:
            lang_embeddings = None

        # constructing target vocabulary info with wav2vec2 size
        target_vocab_info = VocabularyInfo(
            size=self._config.wav2vec2_asr_config.target_vocab_size,
            unk_idx=self._config.unk_idx,
            bos_idx=self._config.bos_idx,
            eos_idx=self._config.eos_idx,
            pad_idx=self._config.pad_idx,
            boh_idx=self._config.boh_idx,
            eoh_idx=self._config.eoh_idx,
        )

        return Wav2Vec2LlamaModel(
            model_type=self._config.model_type,
            model_dim=self._config.llama_config.model_dim,
            encoder_frontend=encoder_frontend,
            encoder=encoder,
            encoder_proj=encoder_proj,
            text_frontend=text_frontend,
            llama_decoder=llama_decoder,
            final_proj=final_proj,
            target_vocab_info=target_vocab_info,
            masker=masker,
            max_generation_length=self._config.llama_config.max_seq_len,
            encoder_stacking=self._config.encoder_stacking,
            lang_embeddings_p=self._config.lang_embeddings_p,
            language_column_name=self._config.language_column_name,
            lang_embeddings=lang_embeddings,
            lang_mapping=lang_mapping,
            context_text_only=self._config.context_text_only,
            n_context_examples=self._config.n_context_examples,
        )
