# omnilingual_asr/data.py

import io
import pandas as pd
import pyarrow as pa
from pathlib import Path
from .constants import DATA_ROOT
import soundfile as sf
import numpy as np


def load_all_data(split: str = "test", data_root: str = DATA_ROOT) -> pd.DataFrame:
    """
    Load a specific split ('train', 'validation', 'test') from all idiom folders.
    Returns a DataFrame with columns: audio_path, sentence, idiom.
    """
    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_path}")

    all_rows = []
    for idiom_folder in sorted(data_path.iterdir()):
        if not idiom_folder.is_dir():
            continue
        tsv_file = idiom_folder / f"{split}.tsv"
        if not tsv_file.exists():
            continue

        df = pd.read_csv(tsv_file, sep="\t")
        # Give the idiom a readable name (remove "rm-" prefix)
        idiom_name = idiom_folder.name.replace("rm-", "")
        clips_dir = idiom_folder / "clips"

        # Build absolute audio paths
        df["audio_path"] = df["path"].apply(lambda rel: str(clips_dir / rel))
        df["idiom"] = idiom_name
        all_rows.append(df)

    if not all_rows:
        raise ValueError(f"No data found for split '{split}' in {data_root}")

    return pd.concat(all_rows, ignore_index=True)

def compress_audio_to_ogg(audio_array, sample_rate):
    """
    Compresses a raw audio NumPy array into OGG format entirely in-memory.

    Args:
        audio_array (np.ndarray): The raw audio waveform data.
        sample_rate (int): The sampling rate of the audio (e.g., 16000 for 16kHz).

    Returns:
        bytes: The raw compressed OGG byte stream, ready for transmission or saving.
    """
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format='ogg')
    return buffer.getvalue()

def binary_to_list_int8(binary_array: pa.Array | pa.ChunkedArray) -> pa.Array:
    """
    Efficiently convert a pyarrow BinaryArray to a ListArray of int8.
    Each binary value becomes a list of int8 values (that's copy-less method)
    Nulls are preserved.
    """
    if not pa.types.is_binary(binary_array.type):
        raise ValueError("Input array must be of binary type.")
    if isinstance(binary_array, pa.ChunkedArray):
        binary_array = binary_array.combine_chunks()

    # Get buffers: [null_bitmap, offsets, data]
    buffers = binary_array.buffers()
    offsets = buffers[1]
    data = buffers[2]
    offset = binary_array.offset

    # Offsets as numpy array
    offsets_np = np.frombuffer(offsets, dtype="int32")[  # type: ignore
        offset : offset + len(binary_array) + 1
    ]

    data_np = np.frombuffer(data, dtype="int8")[offsets_np[0] :]  # type: ignore
    offsets_np -= offsets_np[0]
    values_array = pa.array(data_np, type=pa.int8())

    list_array = pa.ListArray.from_arrays(
        offsets_np, values_array, mask=binary_array.is_null()
    )
    return list_array
