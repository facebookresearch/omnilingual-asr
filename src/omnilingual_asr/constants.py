import os

DATA_ROOT = "../../romansh-data"
FOLDER_NAMES = [folder for folder in os.listdir(DATA_ROOT) if "cc" in folder]
SPLITS = ["train", "validated", "test", "validation"]
SPLIT_FILES = ["train.tsv", "validated.tsv", "test.tsv", "validation.tsv"]
RANDOM_SEED = 42