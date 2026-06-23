import os
from pathlib import Path

SUBMODULE_ROOT = Path(__file__).parent.parent.parent.resolve()
PROJECT_ROOT = SUBMODULE_ROOT.parent.parent.resolve()
DATA_ROOT = PROJECT_ROOT / "data" / "clean-data"
MODELS_ROOT = PROJECT_ROOT / "omnilingual" / "models"
PARQUET_DATA_ROOT = PROJECT_ROOT / "data" / "parquet-data" / "version=0"
LANG_DIST_FILE_ROOT = PROJECT_ROOT / "data" / "parquet-data" / "language_distribution_0.tsv"
FOLDER_NAMES = [folder for folder in os.listdir(DATA_ROOT) if "cc" in folder]
SPLITS = ["train", "validated", "test", "validation"]
SPLIT_FILES = ["train.tsv", "validated.tsv", "test.tsv", "validation.tsv"]
RANDOM_SEED = 42