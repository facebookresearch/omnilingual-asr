# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import types
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from fairseq2.nn import BatchLayout
from torch import Tensor

try:
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel
    from omnilingual_asr.models.wav2vec2_llama.syntax import lang_id_getter
except ImportError:
    pass  # Type checking only


# =============================================================================
# SCRIPT DETECTION & UTILS
# =============================================================================

# Regex for scripts that typically do not use whitespace for word separation.
# Covers: CJK Unified Ideographs, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar.
_NON_SPACED_SCRIPTS = re.compile(
    r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u309F\u30A0-\u30FF"  # CJK + Kana
    r"\u0E00-\u0E7F\u0E80-\u0EFF"  # Thai + Lao
    r"\u1780-\u17FF\u1000-\u109F]"  # Khmer + Myanmar
)


def detect_alignment_mode(text: str) -> str:
    """
    Language-agnostic detection of alignment granularity.
    Returns 'char' if the text contains characters from known non-spaced scripts
    (CJK, Thai, etc.), otherwise returns 'word'.
    """
    if _NON_SPACED_SCRIPTS.search(text):
        return "char"
    return "word"


def chunk_waveform(
    waveform: Tensor, sample_rate: int, max_duration_sec: float
) -> List[Tuple[Tensor, float]]:
    if waveform.ndim != 1:
        raise ValueError("waveform must be 1D")
    chunk_samples = int(max_duration_sec * sample_rate)
    T = waveform.shape[0]
    chunks: List[Tuple[Tensor, float]] = []
    for s in range(0, T, chunk_samples):
        e = min(T, s + chunk_samples)
        chunks.append((waveform[s:e], s / float(sample_rate)))
    return chunks


def split_text_units(transcript: str, mode: str) -> List[Tuple[str, int, int]]:
    """
    Split transcript into alignment units.
    Returns list of (unit_text, start_char_idx, end_char_idx).
    """
    units: List[Tuple[str, int, int]] = []
    if mode == "word":
        # Split on whitespace
        for m in re.finditer(r"\S+", transcript):
            units.append((m.group(0), m.start(), m.end()))
    elif mode == "char":
        # Split into individual characters (ignoring whitespace)
        for idx, ch in enumerate(transcript):
            if not ch.isspace():
                units.append((ch, idx, idx + 1))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return units


# =============================================================================
# CTC ALIGNMENT LOGIC
# =============================================================================


def _make_seqs_layout(x_batched: Tensor, seq_lens: Sequence[int]) -> Any:
    return BatchLayout.of(batch=x_batched, seq_lens=list(seq_lens))


def _ensure_tensor(maybe_ret: Any) -> Tensor:
    return maybe_ret[0] if isinstance(maybe_ret, (tuple, list)) else maybe_ret


def get_ctc_logits(
    asr_model: Any, waveform: Tensor, sample_rate: int
) -> Tuple[Tensor, float]:
    device = next(asr_model.parameters()).device

    if hasattr(asr_model, "encoder_frontend"):
        frontend = asr_model.encoder_frontend
    elif hasattr(asr_model, "frontend"):
        frontend = asr_model.frontend
    else:
        raise RuntimeError("Model has no encoder_frontend or frontend.")

    try:
        f_dtype = next(frontend.parameters()).dtype
    except Exception:
        f_dtype = next(asr_model.parameters()).dtype

    with torch.no_grad():
        wav = waveform.to(device=device, dtype=f_dtype)
        x = wav.unsqueeze(0)
        seqs_layout = _make_seqs_layout(x, [wav.shape[0]])

        ret = frontend.extract_features(x, seqs_layout)
        if isinstance(ret, (tuple, list)):
            feats, layout = ret[0], ret[1]
        else:
            feats, layout = ret, seqs_layout

        try:
            feats = _ensure_tensor(frontend.process_features(feats, layout, None))
        except Exception:
            pass

        if hasattr(asr_model, "encoder"):
            enc = _ensure_tensor(asr_model.encoder(feats, layout))
        elif hasattr(asr_model, "w2v2"):
            enc = _ensure_tensor(asr_model.w2v2(feats, layout))
        else:
            raise RuntimeError("CTC model missing encoder block")

        proj = None
        for name in ("final_proj", "ctc_head", "output_proj", "proj", "decoder"):
            if hasattr(asr_model, name):
                proj = getattr(asr_model, name)
                break
        if proj is None:
            raise RuntimeError("CTC model missing projection head.")

        logits = proj(enc)
        logits = logits.squeeze(0)

        s_frames = feats.shape[1]
        t_samples = waveform.shape[0]
        stride_samples = max(1, round(t_samples / s_frames))
        stride_seconds = stride_samples / float(sample_rate)

        return logits, stride_seconds


def align_ctc(
    model: Any,
    waveform: Tensor,
    sample_rate: int,
    transcript: str,
) -> List[dict]:
    """
    Aligns text to audio using CTC boundaries. Returns list of dicts.
    """
    if not transcript.strip():
        return []

    # 1. Get Logits & Boundaries
    logits, stride_seconds = get_ctc_logits(model, waveform, sample_rate)

    path = torch.argmax(logits, dim=-1).tolist()
    boundaries = [0]
    prev_token = -1
    for frame_idx, token_id in enumerate(path):
        if token_id != 0 and token_id != prev_token:
            boundaries.append(frame_idx)
            prev_token = token_id
        elif token_id != 0:
            prev_token = token_id
    boundaries.append(len(path))

    # 2. Detect Mode & Split
    mode = detect_alignment_mode(transcript)
    units = split_text_units(transcript, mode)

    if not boundaries or len(boundaries) < 2 or not units:
        return []

    # 3. Interpolate
    num_frames = len(boundaries) - 1
    U = len(units)
    results: List[dict] = []

    # Key name based on mode
    key_name = "word" if mode == "word" else "char"

    for u_idx, (tok, _s, _e) in enumerate(units):
        start_bin = int(round((u_idx) * num_frames / U))
        end_bin = int(round((u_idx + 1) * num_frames / U))

        start_bin = min(max(start_bin, 0), num_frames)
        end_bin = min(max(end_bin, start_bin + 1), num_frames)

        start_frame = boundaries[start_bin]
        end_frame = boundaries[end_bin]

        results.append(
            {
                key_name: tok,
                "start": start_frame * stride_seconds,
                "end": end_frame * stride_seconds,
            }
        )

    return results


# =============================================================================
# LLM ALIGNMENT LOGIC (DTW)
# =============================================================================


class AttentionStore:
    def __init__(self) -> None:
        self.weights: Dict[int, torch.Tensor] = {}

    def add_weights(self, layer_idx: int, attn: torch.Tensor) -> None:
        self.weights[layer_idx] = attn.detach().cpu()

    def clear(self) -> None:
        self.weights = {}


_attention_store = AttentionStore()


def make_patched_sdpa_forward(layer_idx: int, original_forward_func):
    def patched_sdpa_forward(self, *args, **kwargs):
        context, attn_weights_orig = original_forward_func(*args, **kwargs)
        try:
            if len(args) >= 3:
                query, key = args[0], args[2]
                if isinstance(query, torch.Tensor) and isinstance(key, torch.Tensor):
                    q = query.to(torch.float32)
                    k = key.to(torch.float32)
                    if q.ndim == 4 and k.ndim == 4 and q.shape[0] == 1:
                        B, T, H, D = q.shape
                        q_flat = q.reshape(T, H * D)
                        k_flat = k.reshape(T, H * D)
                        dim = max(1, H * D)
                        sim = (q_flat @ k_flat.T) / (float(dim) ** 0.5)
                        attn = F.softmax(sim, dim=-1)
                        _attention_store.add_weights(layer_idx, attn)
        except Exception:
            pass
        return context, attn_weights_orig

    return patched_sdpa_forward


def forced_alignment_dtw(similarity_matrix: np.ndarray) -> List[int]:
    """Pure Numpy DTW for forced alignment."""
    N_text, N_audio = similarity_matrix.shape
    score = np.full((N_text + 1, N_audio + 1), -1e9, dtype=np.float32)
    score[0, 0] = 0.0

    for i in range(1, N_text + 1):
        for j in range(1, N_audio + 1):
            s = similarity_matrix[i - 1, j - 1]
            score_diag = score[i - 1, j - 1]
            score_left = score[i, j - 1]
            score[i, j] = max(score_diag, score_left) + s

    path = []
    i, j = N_text, int(np.argmax(score[N_text, :]))

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        s_diag = score[i - 1, j - 1]
        s_left = score[i, j - 1]
        if s_diag >= s_left:
            i, j = i - 1, j - 1
        else:
            j -= 1

    path = path[::-1]
    token_frames: List[List[int]] = [[] for _ in range(N_text)]
    for t_idx, f_idx in path:
        token_frames[t_idx].append(f_idx)

    aligned_frames = []
    last_frame = 0
    for tf in token_frames:
        if tf:
            avg = int(np.mean(tf))
            aligned_frames.append(avg)
            last_frame = avg
        else:
            aligned_frames.append(last_frame)

    return aligned_frames


@torch.inference_mode()
def align_llm(
    pipeline: "ASRInferencePipeline",
    waveform: Tensor,
    transcript: str,
    lang: Optional[str] = None,
) -> List[dict]:
    if not transcript.strip():
        return []

    # Narrow type for MyPy
    model = cast("Wav2Vec2LlamaModel", pipeline.model)

    lang = lang if lang else "eng_Latn"

    # 1. Prepare Inputs
    try:
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)

        audio_data = [{"waveform": waveform, "sample_rate": 16000}]
        audio_candidate = next(
            iter(pipeline._build_audio_wavform_pipeline(audio_data).and_return())
        )
        audio_batch = pipeline._create_batch_simple([(audio_candidate, lang)])

        source_seq_lens = audio_batch.source_seq_lens
        if isinstance(source_seq_lens, torch.Tensor):
            seq_lens_list: List[int] = source_seq_lens.tolist()
        else:
            seq_lens_list = list(source_seq_lens)

        audio_features, _ = model.embed_audio(
            audio_batch.source_seqs.to(dtype=pipeline.dtype),
            seq_lens_list,
        )

        token_ids = pipeline.token_encoder(transcript)
        text_embeddings = model.embed_text(
            token_ids.unsqueeze(0).to(pipeline.device), pipeline.dtype
        )

        vocab_info = pipeline.tokenizer.vocab_info
        bos_emb = model.embed_text(
            torch.tensor([[vocab_info.bos_idx]], device=pipeline.device), pipeline.dtype
        )
        sep_emb = model.embed_text(
            torch.tensor([[vocab_info.size]], device=pipeline.device), pipeline.dtype
        )

        lang_emb = torch.zeros(
            1, 0, audio_features.shape[-1], device=pipeline.device, dtype=pipeline.dtype
        )
        lang_mapping = getattr(model, "lang_mapping", None)
        if model.lang_embeddings is not None and lang_mapping is not None:
            lid = lang_id_getter(lang_mapping, lang)
            lang_emb = model.lang_embeddings(
                torch.tensor([lid], device=pipeline.device).unsqueeze(0)
            )

        full_input = torch.cat(
            [audio_features, sep_emb, lang_emb, bos_emb, text_embeddings], dim=1
        )

    except Exception as e:
        print(f"Error preparing LLM alignment inputs: {e}")
        return []

    # 2. Run Decoder with Hook
    _attention_store.clear()
    original_forwards: Dict[int, Callable[..., Any]] = {}
    patched_layers: List[int] = []

    try:
        for i, layer in enumerate(model.llama_decoder.layers):
            sdpa = getattr(layer.self_attn, "sdpa", None)
            sdpa_forward = getattr(sdpa, "forward", None)
            if sdpa is None or sdpa_forward is None:
                continue

            original_forwards[i] = cast(Callable[..., Any], sdpa_forward)
            sdpa.forward = types.MethodType(
                make_patched_sdpa_forward(i, original_forwards[i]), sdpa
            )
            patched_layers.append(i)

        B, T_full, _ = full_input.shape
        layout = BatchLayout(
            shape=(B, T_full), seq_lens=[T_full], packed=False, device=full_input.device
        )

        model.llama_decoder(seqs=full_input, seqs_layout=layout, state_bag=None)

    except Exception as e:
        print(f"Error running LLM alignment pass: {e}")
        return []
    finally:
        for i in patched_layers:
            layer = model.llama_decoder.layers[i]
            sdpa = getattr(layer.self_attn, "sdpa", None)
            if sdpa is not None and i in original_forwards:
                sdpa.forward = original_forwards[i]

    if not _attention_store.weights:
        return []

    # 3. DTW
    L_audio = audio_features.shape[1]
    L_pre = sep_emb.shape[1] + lang_emb.shape[1] + bos_emb.shape[1]
    L_text = text_embeddings.shape[1]

    query_start = L_audio + L_pre

    sorted_layers = sorted(_attention_store.weights.keys())
    num_layers = len(sorted_layers)
    start_layer = int(num_layers * 0.4)
    end_layer = int(num_layers * 0.9)
    selected = [l for l in sorted_layers if start_layer <= l < end_layer]
    if not selected:
        selected = sorted_layers

    avg_attn = torch.zeros_like(_attention_store.weights[selected[0]])
    for l in selected:
        avg_attn += _attention_store.weights[l]
    avg_attn /= len(selected)

    cross_attn = avg_attn[query_start : query_start + L_text, :L_audio].numpy()

    aligned_frames = forced_alignment_dtw(cross_attn)

    # 4. Decode to Words/Chars (Mode detection logic)
    decoded_tokens = [
        pipeline.token_decoder(token_ids[i : i + 1]) for i in range(len(token_ids))
    ]

    mode = detect_alignment_mode(transcript)
    key_name = "word" if mode == "word" else "char"

    # Group tokens
    groups = []
    current_group: List[int] = []

    if mode == "word":
        # Group subwords into words (SentencePiece logic)
        for i, tk in enumerate(decoded_tokens):
            if (tk.startswith(" ") or tk in ".,!?") and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(i)
        if current_group:
            groups.append(current_group)
    else:
        # Char mode: We still have subword tokens. We must map tokens to text characters.
        # This is complex for LLM tokens vs unicode chars.
        # Simplified approach for LLM: One token usually contains multiple chars or one char.
        # We will output token-level timestamps but label them "char" if they are single chars.
        # Ideally, we would map token-frames to char-offsets, but that requires a char-to-token map.
        # Fallback: Just report token-level as "char" units for now, grouping strictly by token index.
        for i in range(len(decoded_tokens)):
            groups.append([i])

    results = []
    frame_dur = 0.02  # 20ms

    for indices in groups:
        s_idx, e_idx = indices[0], indices[-1]
        s_frame, e_frame = aligned_frames[s_idx], aligned_frames[e_idx]

        start = s_frame * frame_dur
        end = e_frame * frame_dur + frame_dur

        text_fragment = "".join([decoded_tokens[i] for i in indices])
        if mode == "word":
            text_fragment = text_fragment.replace(" ", "")

        if not text_fragment:
            continue

        results.append({key_name: text_fragment, "start": start, "end": end})

    return results
