# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations
import re
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
from fairseq2.data._memory import MemoryBlock
from fairseq2.data.audio import AudioDecoder
from fairseq2.data.data_pipeline import (
    CollateOptionsOverride,
    Collater,
    DataPipeline,
    DataPipelineBuilder,
    FileMapper,
    read_sequence,
)
from fairseq2.data.tokenizers import Tokenizer
from fairseq2.data.tokenizers.hub import load_tokenizer
from fairseq2.datasets.batch import Seq2SeqBatch
from fairseq2.logging import get_log_writer
from fairseq2.models.hub import load_model
from fairseq2.models.wav2vec2.asr import Wav2Vec2AsrModel
from fairseq2.nn.batch_layout import BatchLayout
from numpy.typing import NDArray
from torch import Tensor
from omnilingual_asr.datasets.utils.audio import add_waveform_processing
from omnilingual_asr.models.wav2vec2_llama.beamsearch import (
    Wav2Vec2LlamaBeamSearchSeq2SeqGenerator,
)
from omnilingual_asr.models.wav2vec2_llama.config import ModelType
from omnilingual_asr.models.wav2vec2_llama.model import (
    Wav2Vec2LlamaBeamSearchConfig,
    Wav2Vec2LlamaModel,
)
log = get_log_writer(__name__)
AudioInput = (
    List[Path]
    | List[str]
    | List[str | Path]
    | List[bytes]
    | List[NDArray[np.int8]]
    | List[bytes | NDArray[np.int8]]
    | List[Dict[str, Any]]
)
MAX_ALLOWED_AUDIO_SEC: Final = 40
@dataclass
class WordTimestamp:
    """Represents a single timestamped word/character."""
    word: str
    start: float
    end: float
@dataclass
class ContextExample:
    """
    Represents a single context example with audio and text.
    Args:
        audio: Audio input (same formats as AudioInput)
        text: Corresponding text transcription
    """
    audio: str | Path | bytes | NDArray[np.int8] | Dict[str, Any]
    text: str
def resample_to_16khz(
    audio_data: Dict[str, Any], target_sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    Resample audio waveform to target sample rate (16kHz by default).
    Args:
        audio_data: Dictionary containing 'waveform' and 'sample_rate' keys
        target_sample_rate: Target sample rate (default: 16000)
    Returns:
        Dictionary with resampled waveform and updated sample rate
    """
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]
    if current_sample_rate != target_sample_rate:
        # Resample the waveform using torchaudio functional
        log.debug(f"Resampling from {current_sample_rate}Hz to {target_sample_rate}Hz")

        # Different audio reading mechanisms can cause the shape to be either (channels, time)
        # or (time, channels). We go heuristically by the longer axis to enforce (channels, time)
        # is going to F.resample, and (time, channels) is returned
        assert len(waveform.shape) <= 2
        need_transpose = (
            len(waveform.shape) > 1 and waveform.shape[0] > waveform.shape[1]
        )
        if need_transpose:
            waveform = waveform.transpose(1, 0)
        waveform = F.resample(
            waveform,
            orig_freq=current_sample_rate,
            new_freq=target_sample_rate,
        )
        if need_transpose:
            waveform = waveform.transpose(1, 0)
        audio_data["sample_rate"] = target_sample_rate
        audio_data["waveform"] = waveform
    return audio_data
def repeat_to_max_len(
    lists: List[List[ContextExample]], max_len: int
) -> List[List[ContextExample]]:
    """Repeats each inner list of `lists` until it reaches the `max_len`.
    This is used to replicate context examples to fit the zero-shot model training setup, which always
    saw exactly `max_len` context examples for low-resource languages.
    If more than `max_len` examples are provided we trim them down to `max_len`.
    """
    def extend_list(lst):
        repetitions = (max_len // len(lst)) + 1
        return (lst * repetitions)[:max_len]
    return [extend_list(lst) for lst in lists]
def assert_max_length(
    audio_data: Dict[str, Any], target_sample_rate: int = 16000
) -> Dict[str, Any]:
    waveform = audio_data["waveform"]
    current_sample_rate = audio_data["sample_rate"]
    waveform_len_s = len(waveform) / current_sample_rate
    if waveform_len_s > MAX_ALLOWED_AUDIO_SEC:
        warnings.warn(f"Audio longer than {MAX_ALLOWED_AUDIO_SEC}s is not recommended and might fail.")
    return audio_data
class ASRInferencePipeline:
    def __init__(
        self,
        model_card: str | None,
        *,
        model: Wav2Vec2LlamaModel | Wav2Vec2AsrModel | None = None,
        tokenizer: Tokenizer | None = None,
        device: str | None | torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
        beam_search_config: Wav2Vec2LlamaBeamSearchConfig | None = None,
    ) -> None:
        """
        Initialize the inference pipeline.
        Args:
            model_card: Model card name to load from the hub (mutually exclusive with model/tokenizer)
                Recommended to use model_card for inference !
            model: Pre-loaded Wav2Vec2LlamaModel instance (mutually exclusive with model_card)
            tokenizer: Pre-loaded Tokenizer instance (mutually exclusive with model_card)
            device: Device to run inference on
            dtype: Data type for model inference
            beam_search_config: Optional beam search configuration
        Raises:
            ValueError: If both model_card and (model/tokenizer) are provided, or if model/tokenizer are provided without each other
        """
        # Validate mutually exclusive arguments
        if model_card is not None and (model is not None or tokenizer is not None):
            raise ValueError(
                "model_card is mutually exclusive with model/tokenizer. "
                "Provide either model_card OR both model and tokenizer."
            )
        if (model is None) != (tokenizer is None):
            raise ValueError(
                "Both model and tokenizer must be provided together when not using model_card"
            )
        if model_card is None and (model is None or tokenizer is None):
            raise ValueError(
                "Must provide either model_card OR both model and tokenizer"
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        # Load or use provided model and tokenizer
        if model_card is not None:
            # Load from model card (original behavior)
            log.info(f"Loading model from model card: {model_card}")
            self.model = load_model(model_card, device=self.device, dtype=self.dtype)
            # Load tokenizer from model card
            log.info(f"Loading tokenizer from model card: {model_card}")
            self.tokenizer = load_tokenizer(model_card)
        else:
            # Use provided model and tokenizer
            assert isinstance(tokenizer, Tokenizer)
            assert isinstance(model, (Wav2Vec2LlamaModel, Wav2Vec2AsrModel))
            log.info("Using provided model and tokenizer")
            self.model = model
            self.model = self.model.to(
                device=self.device
            ) # TODO avoid moving buffers to dtype
            self.tokenizer = tokenizer
        # Set model to evaluation mode
        self.model.eval()
        # Set up beam search
        if beam_search_config is None:
            beam_search_config = Wav2Vec2LlamaBeamSearchConfig(
                nbest=1,
                length_norm=False,
            )
        else:
            assert isinstance(beam_search_config, Wav2Vec2LlamaBeamSearchConfig)
        self.beam_search_generator = None
        if isinstance(self.model, Wav2Vec2LlamaModel):
            self.beam_search_generator = Wav2Vec2LlamaBeamSearchSeq2SeqGenerator(
                model=self.model, config=beam_search_config
            )
        # Set up tokenizer decoder
        assert self.tokenizer is not None # Should always be non-None at this point
        self.token_decoder = self.tokenizer.create_decoder(skip_special_tokens=True)
        self.token_encoder = self.tokenizer.create_encoder()
        # Set up audio processor
        self.audio_decoder = AudioDecoder(dtype=torch.float32)
        self.file_mapper = FileMapper(cached_fd_count=200)
        pad_idx = getattr(self.tokenizer.vocab_info, "pad_idx", 0)
        text_collate_opts = CollateOptionsOverride("text", pad_value=pad_idx)
        self.full_collater = Collater(
            pad_value=0, overrides=[text_collate_opts] # Default pad value for audio
        )
        self.collater_audio = Collater(pad_value=0)
        self.collater_text = Collater(pad_value=pad_idx)
        model_source = (
            f"model_card={model_card}" if model_card else "provided model/tokenizer"
        )
        log.info(
            f"Pipeline initialized on {self.device} with dtype {self.dtype} using {model_source}"
        )
    def _create_batch_simple(
        self, wavs_langs: List[Tuple[torch.Tensor, str | None]]
    ) -> Seq2SeqBatch:
        """Create a Seq2SeqBatch from audio tensors using fairseq2 utilities."""
        # Create audio data structure similar to ASR task
        audio_examples = []
        for item in wavs_langs:
            audio_examples.append(
                {
                    "audio_feature": item[0],
                    "text": torch.tensor(
                        [0], dtype=torch.int64
                    ), # Dummy text for inference
                }
            )
        # Use Collater with proper pad_value for audio (0) and text (pad_idx)
        # Following ASR task's approach with CollateOptionsOverride for text
        # Collate the examples
        collated_data = self.full_collater(audio_examples)
        # Extract audio and text data
        audio_data = collated_data["audio_feature"]
        text_data = collated_data["text"]
        example = {"lang": [item[1] for item in wavs_langs]}
        if all(x is None for x in example["lang"]):
            example = {}
        # Following ASR task's to_seq2seq_batch method
        return Seq2SeqBatch(
            source_seqs=audio_data["seqs"].to(self.device, self.dtype),
            source_seq_lens=audio_data["seq_lens"],
            target_seqs=text_data["seqs"].to(self.device),
            target_seq_lens=text_data["seq_lens"],
            example=example,
        )
    def _apply_model_wav2vec2asr(self, batch: Seq2SeqBatch) -> List[str]:
        batch_layout = BatchLayout.of(batch.source_seqs, batch.source_seq_lens)
        logits, bl_out = self.model(batch.source_seqs, batch_layout)
        pred_ids = torch.argmax(logits, dim=-1)
        transcriptions = []
        for i in range(pred_ids.shape[0]):
            # Create a mask for where consecutive elements differ (CTC decoding)
            # First element is always True, then compare with previous elements
            seq = pred_ids[i][: bl_out.seq_lens[i]]
            mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
            mask[1:] = seq[1:] != seq[:-1]
            # Use the mask to select non-duplicate tokens
            decoded_ids = seq[mask]
            transcriptions.append(self.token_decoder(decoded_ids))
        return transcriptions
    def _apply_model_wav2vec2llama(self, batch: Seq2SeqBatch) -> List[str]:
        context_logits, context_layout = self.model(batch, return_decoder_inputs=True)
        # Generate hypotheses using beam search
        assert self.beam_search_generator is not None
        hypothesis_tokens, hypothesis_layout = (
            self.beam_search_generator.generate_hypotheses(
                decoder_context_inputs=context_logits,
                decoder_context_input_layout=context_layout,
            )
        )
        # Decode tokens to text
        transcriptions = []
        for i in range(hypothesis_tokens.shape[0]):
            seq_len = hypothesis_layout.seq_lens[i]
            tokens = hypothesis_tokens[i, :seq_len]
            text = self.token_decoder(tokens)
            transcriptions.append(text)
        return transcriptions
    def _apply_model(self, batch: Seq2SeqBatch) -> List[str]:
        """Apply model forward pass to the batch."""
        if isinstance(self.model, Wav2Vec2LlamaModel):
            transcriptions = self._apply_model_wav2vec2llama(batch)
        elif isinstance(self.model, Wav2Vec2AsrModel):
            transcriptions = self._apply_model_wav2vec2asr(batch)
        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return transcriptions
    def _build_audio_wavform_pipeline(
        self,
        inp_list: AudioInput,
    ) -> DataPipelineBuilder:
        """Process audio inputs using fairseq2.data pipeline similar to ASR task."""
        # Build pipeline based on input type
        builder = read_sequence(inp_list)
        need_to_decode = True
        first_element = inp_list[0]
        if isinstance(first_element, (Path, str)):
            builder = builder.map(str)
            builder = builder.map(self.file_mapper)
        elif isinstance(
            first_element, (bytes, np.ndarray)
        ): # list of audio bytes in memory
            if isinstance(first_element, np.ndarray):
                assert first_element.dtype in [
                    np.uint8,
                    np.int8,
                ], "Only uint8 numpy arrays are supported"
            builder = builder.map(lambda x: {"data": MemoryBlock(x)})
        elif isinstance(first_element, dict): # list[dict([waveform, sample_rate])]
            need_to_decode = False
            log.info("Processing pre-decoded audio dictionaries")
            builder = builder.map(
                lambda x: {
                    "data": {
                        "waveform": torch.tensor(x["waveform"]),
                        "sample_rate": int(x["sample_rate"]),
                    }
                }
            )
        else:
            raise ValueError(f"Unsupported input type: {type(first_element)}")
        if need_to_decode:
            builder = builder.map(self.audio_decoder, selector="data")
            # Add resampling to 16kHz after audio decoding
        # Resample
        builder = builder.map(resample_to_16khz, selector="data")
        # Add waveform processing
        builder = add_waveform_processing(
            builder,
            normalize_audio=True,
            dtype=self.dtype,
            selector="data.waveform", # Process the waveform from decoded audio
            spec_aug_p=None, # No SpecAugment for inference
            spec_aug_freq_mask_param=0,
            spec_aug_time_mask_param=0,
        )
        return builder
    def _process_context_audio(
        self, context_examples: List[ContextExample]
    ) -> Dict[str, Any] | None:
        """
        Process context audio examples into tensors.
        Args:
            context_examples: List of context examples with audio and text
        Returns:
            Dictionary of collater audio tensors
            keys = (seqs, seq_lens, is_ragged)
        """
        if not context_examples:
            return None
        # Use the same audio processing pipeline as main inference
        builder = self._build_audio_wavform_pipeline([example.audio for example in context_examples]) # type: ignore[arg-type]
        audio_data = list(builder.and_return())
        context_audio_tensors = [d["data"]["waveform"] for d in audio_data]
        collated_audio = self.collater_audio(context_audio_tensors)
        collated_audio["seqs"] = collated_audio["seqs"].to(self.device, self.dtype)
        collated_audio["seq_lens"] = torch.tensor(
            collated_audio["seq_lens"], device=self.device
        )
        return collated_audio
    def _process_context_text(
        self, context_examples: List[ContextExample]
    ) -> List[torch.Tensor]:
        """
        Process context text examples into tokenized tensors.
        Args:
            context_examples: List of context examples with audio and text
        Returns:
            Dictionary of collater tokenized text
            keys = (seqs, seq_lens, is_ragged)
        """
        if not context_examples:
            return []
        context_text_tensors = []
        for example in context_examples:
            # Tokenize the text using the tokenizer's encoder
            text_tensor = self.token_encoder(example.text)
            context_text_tensors.append(text_tensor)
        collated_text = self.collater_text(context_text_tensors)
        collated_text["seqs"] = collated_text["seqs"].to(self.device)
        collated_text["seq_lens"] = torch.tensor(
            collated_text["seq_lens"], device=self.device
        )
        return collated_text
    def _create_batch_with_context(
        self, combined_batch: List[Tuple[torch.Tensor, List[ContextExample]]]
    ) -> Seq2SeqBatch:
        """
        Create a Seq2SeqBatch with zero-shot context support.
        Args:
            wavs: List of audio tensors for transcription
            per_audio_context_examples: List of context examples for each audio input
        Returns:
            Seq2SeqBatch with context information
        """
        # Start with the basic batch creation (with empty lang)
        batch = self._create_batch_simple([(item[0], None) for item in combined_batch]) # type: ignore[index]
        # Process context examples for each input audio
        context_audio = []
        context_text = []
        for combined_item in combined_batch:
            context_examples = combined_item[1] # type: ignore[index]
            context_audio_tensors = self._process_context_audio(context_examples)
            context_text_tensors = self._process_context_text(context_examples)
            context_audio.append(context_audio_tensors)
            context_text.append(context_text_tensors)
        batch.example["context_audio"] = context_audio # type: ignore[index]
        batch.example["context_text"] = context_text # type: ignore[index]
        return batch
    @torch.inference_mode()
    def transcribe(
        self,
        inp: AudioInput,
        *,
        lang: Optional[List[Optional[str]]] = None,
        batch_size: int = 2,
        chunk_len: int = 40,
        return_timestamps: bool = True,
    ) -> Union[List[str], Tuple[List[str], List[List[Dict[str, Any]]]]]:
        """
        Transcribes `AudioInput` into text by preprocessing (decoding, resample to 16kHz, converting to mono, normalizing)
        each input sample and performing inference with `self.model`.

        Args:
            `inp`: Audio input in different forms.
            `lang`: Language code for the input audios (e.g., 'eng_Latn', ...)
            `batch_size`: Number of audio samples to process in each batch.
            `chunk_len`: The length of audio chunks in seconds to process. Silently capped to [1, 40].
            `return_timestamps`: If True, returns timestamps along with transcriptions. Default True.

        Returns:
            - A list of transcribed texts if `return_timestamps` is False.
            - A tuple containing a list of transcriptions and a list of timestamp lists if `return_timestamps` is True.
              For non-CTC models, the timestamp list will be empty and a warning will be issued.
        """
        is_ctc_model = isinstance(self.model, Wav2Vec2AsrModel)
        if return_timestamps and not is_ctc_model:
             #warnings.warn("Timestamps are only available for CTC models. Returning empty timestamps.")
        if len(inp) == 0:
            return ([], []) if return_timestamps else []
        # silently cap chunk length between 1 and 40 seconds
        chunk_len = max(1, min(40, chunk_len))
        audio_pipeline_builder = self._build_audio_wavform_pipeline(inp)
        loaded_audios = list(audio_pipeline_builder.and_return())
        all_chunks = []
        chunk_info = []
        for i, audio_data in enumerate(loaded_audios):
            waveform = audio_data["data"]["waveform"]
            sample_rate = audio_data["data"]["sample_rate"]
            chunks = self._chunk_waveform(waveform, sample_rate, chunk_len)
            for chunk_wav, offset_sec in chunks:
                all_chunks.append((chunk_wav, lang[i] if lang else None))
                chunk_info.append({"original_idx": i, "offset": offset_sec})
        builder = read_sequence(all_chunks)
        builder = builder.bucket(batch_size)
        builder = builder.map(self._create_batch_simple)
        builder = builder.prefetch(1)
        chunk_transcriptions = []
        chunk_logits_info = []
        for batch in builder.and_return():
            if is_ctc_model:
                source_seqs_layout = BatchLayout.of(batch.source_seqs, batch.source_seq_lens)
                logits, bl_out = self.model(batch.source_seqs, source_seqs_layout)
                pred_ids = torch.argmax(logits, dim=-1)
                for i in range(pred_ids.shape[0]):
                    seq = pred_ids[i][: bl_out.seq_lens[i]]
                    mask = torch.ones(seq.shape[0], dtype=torch.bool, device=seq.device)
                    mask[1:] = seq[1:] != seq[:-1]
                    decoded_ids = seq[mask]
                    chunk_transcriptions.append(self.token_decoder(decoded_ids))
                # pair per-example logits slices with their seq lens
                chunk_logits_info.extend(list(zip(logits, bl_out.seq_lens)))
            else:
                transcriptions = self._apply_model(batch)
                chunk_transcriptions.extend(transcriptions)
        stitched_transcriptions = [""] * len(inp)
        for i, text in enumerate(chunk_transcriptions):
            info = chunk_info[i]
            original_idx = info["original_idx"]
            stitched_transcriptions[original_idx] += text + " "
       
        final_transcriptions = [t.strip() for t in stitched_transcriptions]
        if not return_timestamps:
            return final_transcriptions
        if not is_ctc_model:
            return final_transcriptions, [[] for _ in range(len(inp))]
        stitched_timestamps = [[] for _ in range(len(inp))]
        _, stride_seconds = self._get_encoder_stride_seconds(self.model)
        for i, (text, (logits, logit_len)) in enumerate(zip(chunk_transcriptions, chunk_logits_info)):
            info = chunk_info[i]
            original_idx = info["original_idx"]
            offset = info["offset"]
            if text.strip():
                frame_boundaries = self._get_ctc_frame_boundaries(logits[:logit_len])
                use_char_segments = self._is_non_space_script(text)
                split_mode = "char" if use_char_segments else "word"
                units = self._split_text_units(text, mode=split_mode)
                timestamps = self._align_units_to_ctc_frames(units, frame_boundaries, stride_seconds)
                for ts in timestamps:
                    ts.start += offset
                    ts.end += offset
               
                stitched_timestamps[original_idx].extend(timestamps)
       
        final_timestamps = []
        for idx, ts_list in enumerate(stitched_timestamps):
            sanitized_ts = self._sanitize_timestamps(ts_list)
            # decide key by the final transcript script
            use_char_segments_final = self._is_non_space_script(final_transcriptions[idx])
            key = "char" if use_char_segments_final else "word"
            converted = []
            for ts in sanitized_ts:
                val = ts["word"]
                converted.append({key: val, "start": round(ts["start"], 2), "end": round(ts["end"], 2)})
            final_timestamps.append(converted)
           
        return final_transcriptions, final_timestamps
    def _chunk_waveform(self, waveform: Tensor, sample_rate: int, max_duration_sec: float) -> List[Tuple[Tensor, float]]:
        if waveform.ndim != 1:
            waveform = waveform.squeeze(0)
        if waveform.ndim != 1:
            raise ValueError("Waveform must be 1D or squeezable to 1D")
        chunk_samples = int(max_duration_sec * sample_rate)
        T = waveform.shape[0]
        chunks: List[Tuple[Tensor, float]] = []
        for s in range(0, T, chunk_samples):
            e = min(T, s + chunk_samples)
            chunks.append((waveform[s:e], s / float(sample_rate)))
        return chunks
    def _get_encoder_stride_seconds(self, model: Any) -> Tuple[int, float]:
        stride_samples = 320
        sample_rate = 16000
        return stride_samples, stride_samples / sample_rate
    def _get_ctc_frame_boundaries(self, logits: Tensor) -> List[int]:
        path = torch.argmax(logits, dim=-1).tolist()
        boundaries = [0]
        prev_token = -1
        for i, tok in enumerate(path):
            if tok != 0 and tok != prev_token: # 0 is blank
                boundaries.append(i)
            if tok != 0:
                prev_token = tok
        boundaries.append(len(path))
        out = [boundaries[0]]
        for b in boundaries[1:]:
            if b > out[-1]:
                out.append(b)
        return out
    def _is_non_space_script(self, text: str) -> bool:
        _NON_SPACE_RANGES = [
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xAC00, 0xD7AF),
            (0x3040, 0x309F), (0x30A0, 0x30FF), (0x0E00, 0x0E7F),
            (0x0E80, 0x0EFF), (0x1780, 0x17FF), (0x1000, 0x109F),
        ]
        stripped_text = "".join(text.split())
        if not stripped_text: return False
        non_space_char_count = 0
        for char in stripped_text:
            code_point = ord(char)
            for start, end in _NON_SPACE_RANGES:
                if start <= code_point <= end:
                    non_space_char_count += 1
                    break
        return (non_space_char_count / len(stripped_text)) > 0.5
    def _is_punctuation(self, char: str) -> bool:
        return unicodedata.category(char).startswith('P')
    def _split_text_units(self, transcript: str, mode: str) -> List[Tuple[str, int, int]]:
        if mode == "char":
            return [(ch, i, i + 1) for i, ch in enumerate(transcript) if not ch.isspace()]
       
        units = []
        for match in re.finditer(r'\S+', transcript):
            word = match.group(0)
            start_idx, end_idx = match.span()
           
            leading_punct = ""
            while word and self._is_punctuation(word[0]):
                leading_punct += word[0]
                word = word[1:]
            trailing_punct = ""
            while word and self._is_punctuation(word[-1]):
                trailing_punct = word[-1] + trailing_punct
                word = word[:-1]
           
            current_offset = 0
            if leading_punct:
                units.append((leading_punct, start_idx, start_idx + len(leading_punct)))
                current_offset += len(leading_punct)
            if word:
                units.append((word, start_idx + current_offset, start_idx + current_offset + len(word)))
                current_offset += len(word)
            if trailing_punct:
                units.append((trailing_punct, start_idx + current_offset, start_idx + current_offset + len(trailing_punct)))
        return units
    def _align_units_to_ctc_frames(self, units: List[Tuple[str, int, int]], frame_boundaries: List[int], stride_seconds: float) -> List[WordTimestamp]:
        if not frame_boundaries or len(frame_boundaries) < 2 or not units:
            return []
       
        num_frames = len(frame_boundaries) - 1
        num_units = len(units)
        if num_units == 0:
            return []
        timestamps = []
       
        for i, (unit_text, _, _) in enumerate(units):
            start_bin = int(round(i * num_frames / num_units))
            end_bin = int(round((i + 1) * num_frames / num_units))
           
            start_bin = min(max(start_bin, 0), num_frames)
            end_bin = min(max(end_bin, start_bin + 1), num_frames)
            start_frame = frame_boundaries[start_bin]
            end_frame = frame_boundaries[end_bin]
            timestamps.append(WordTimestamp(word=unit_text, start=start_frame * stride_seconds, end=end_frame * stride_seconds))
           
        return timestamps
    def _sanitize_timestamps(self, timestamps: List[WordTimestamp]) -> List[Dict[str, Any]]:
        if not timestamps:
            return []
       
        merged = []
        i = 0
        while i < len(timestamps):
            current_ts = timestamps[i]
            is_current_punct = all(self._is_punctuation(c) for c in current_ts.word)
           
            if is_current_punct:
                punct_group_start = current_ts.start
                punct_group_end = current_ts.end
               
                j = i + 1
                while j < len(timestamps):
                    next_ts = timestamps[j]
                    is_next_punct = all(self._is_punctuation(c) for c in next_ts.word)
                    if is_next_punct:
                        punct_group_end = next_ts.end
                        j += 1
                    else:
                        break
               
                if j > i + 1:
                    merged.append({"word": "", "start": punct_group_start, "end": punct_group_end})
                else:
                    merged.append({"word": current_ts.word, "start": current_ts.start, "end": current_ts.end})
                i = j
            else:
                merged.append({"word": current_ts.word, "start": current_ts.start, "end": current_ts.end})
                i += 1
        return merged
    @torch.inference_mode()
    def transcribe_with_context(
        self,
        inp: AudioInput,
        context_examples: List[List[ContextExample]],
        *,
        batch_size: int = 1,
    ) -> List[str]:
        """
        Transcribes `AudioInput` into text by preprocessing (decoding, resample to 16kHz, converting to mono, normalizing)
        each input sample and its `context_examples` and performing inference with `self.model` by leveraging the context examples
        as a "reference" on how to transcribe.
        The zero-shot model was trained on up to 30s samples per context example, with **10** examples per training sample.
        Please provide at least a single context example per input sample which is replicated until we reach 10 in total.
        If >10 samples are provided we crop to the first ten.
        Warning: Only works for the `omniASR_LLM_7B_ZS` model.
        Args:
            `inp`: Audio input in different forms.
                - `List[ Path | str ]`: Audio file paths
                - `List[ bytes ]`: Raw audio data
                - `List[ np.ndarray ]`: Audio data as uint8 numpy array
                - `List[ dict[str, Any] ]`: Pre-decoded audio with 'waveform' and 'sample_rate' keys
            `context_examples`: A list of context examples for each input audio.
                - `List[ List[ContextExample] ]`: Each inner list contains context examples (audio-text pairs) for that specific audio.
            `batch_size`: Number of audio samples to process in each batch.
        Returns:
            `List[str]`: Transcribed texts.
        """
        if len(inp) == 0:
            return []
        # fmt: off
        is_ctc_model = isinstance(self.model, Wav2Vec2AsrModel)
        is_llm_model = isinstance(self.model, Wav2Vec2LlamaModel)
        if is_ctc_model:
            raise NotImplementedError("CTC models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")
        if is_llm_model and self.model.model_type != ModelType.ZERO_SHOT:
            raise NotImplementedError("LLM models do not support context conditioning. Please use `.transcribe()` instead of `.transcribe_with_context()`.")
        assert len(inp) == len(context_examples), f"Number of audio inputs ({len(inp)}) must match number of context examples {len(context_examples)}."
        for i, examples in enumerate(context_examples):
            assert len(examples) > 0, f"Input index {i} has no context examples, but needs at least one."
            if len(examples) > 10:
                log.info(f"Found {len(examples)} context examples for input index {i}, but can only process 10. Ignoring extra.")
        # fmt: on
        max_context_example_per_sample = 10
        context_examples = repeat_to_max_len(
            context_examples, max_len=max_context_example_per_sample
        )
        builder = self._build_audio_wavform_pipeline(inp)
        waveforms = [d["data"]["waveform"] for d in list(builder.and_return())]
        combined_builder = DataPipeline.zip(
            [
                read_sequence(waveforms).and_return(),
                read_sequence(context_examples).and_return(),
            ]
        )
        combined_builder = combined_builder.bucket(batch_size)
        combined_builder = combined_builder.map(self._create_batch_with_context)
        combined_builder = combined_builder.prefetch(1)
        combined_builder = combined_builder.map(self._apply_model)
        combined_builder = combined_builder.yield_from(
            lambda seq: read_sequence(seq).and_return()
        )
        transcriptions = list(combined_builder.and_return())
        return transcriptions
