# omnilingual_asr/data.py

import pandas as pd
from pathlib import Path
from typing import List
from .constants import DATA_ROOT

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