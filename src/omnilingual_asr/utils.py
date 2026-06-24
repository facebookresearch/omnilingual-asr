import subprocess
import re
import unicodedata
import yaml
from pathlib import Path
from .constants import LANG_DIST_FILE_ROOT, PARQUET_DATA_ROOT

def get_idiom_name_by_folder(folder_name):
  """Returns the name of the idiom given its folder name"""
  name = (folder_name.split("-")[0])[2:]
  match name:
    case "sursilv":
      return "Sursilvan"
    case "sursiv":
      return "Surmiran"
    case "sutsilv":
      return "Sutsilvan"
    case "puter":
      return "Puter"
    case "vallader":
      return "Vallader"
    case _:
      return "RG"
     
def get_language_code_by_folder(folder_name):
  """Returns the language code of the idiom given its folder name"""
  name = (folder_name.split("-")[0])[2:]
  match name:
    case "sursilv":
      return "roh_Latn_surs1244"
    case "sursiv":
      return "roh_Latn_surm1243"
    case "sutsilv":
      return "roh_Latn_suts1235"
    case "puter":
      return "roh_Latn_uppe1396"
    case "vallader":
      return "roh_Latn_lowe1386"
    case _:
      return "roh_Latn_ruma1247"

def get_best_gpu():
    """Finds GPU with most free memory available and returns the index"""
    try:
        cmd = ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                idx, free_mem = [int(x.strip()) for x in line.split(',')]
                gpus.append((idx, free_mem))
        
        if not gpus:
            return 0
        
        max_free = max(free for _, free in gpus)
        best_gpus = [idx for idx, free in gpus if free == max_free]
        selected = best_gpus[len(best_gpus)-1]
        
        print(f"Selected GPU {selected} with {max_free} MiB free memory")
        return selected
        
    except Exception as e:
        print(f"Error detecting best GPU: {e}")
        print("Defaulting to GPU 0")
        return 0

def normalize_romansh_text(text: str) -> str:
    """Normalize text for Romansh ASR:
    - Unicode NFD → remove combining characters → NFC
    - Lowercase
    - Remove punctuation (keep letters and whitespace)
    - Collapse multiple spaces
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def set_config_paths(config_path: Path, dataset_card_path: Path, ):
    """Reads the config yaml, updates paths dynamically, and saves the runtime config."""
    if not config_path.exists():
        print(f"Error: Configuration not found at {config_path}")
        raise FileNotFoundError(f"Configuration not found at: {config_path}")
        
    print("Generating runtime YAML configuration dynamically...")
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)
    
    resolved_path = LANG_DIST_FILE_ROOT.resolve()
    if "dataset" in config_data and "mixture_parquet_storage_config" in config_data["dataset"]:
        config_data["dataset"]["mixture_parquet_storage_config"]["dataset_summary_path"] = str(resolved_path)
    else:
        print("Error: The config YAML structure does not match the expected nested keys.")
        raise KeyError(f"The config is missing 'dataset_summary_path'")
        
    with open(config_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)
        
    print(f"Runtime config written to {config_path}")
    print(f"   dataset_summary_path dynamically set to: {resolved_path}")

    if not dataset_card_path.exists():
        print(f"Error: Dataset card not found at {dataset_card_path}")
        raise FileNotFoundError(f"Dataset card not found at: {dataset_card_path}")
        
    with open(dataset_card_path, "r") as f:
        dataset_card_data = yaml.safe_load(f)
        
    if "dataset_config" in dataset_card_data:
        dataset_card_data["dataset_config"]["data"] = str(PARQUET_DATA_ROOT)
    else:
        print("Error: The dataset card YAML structure does not match the expected keys.")
        raise KeyError(f"The dataset card is missing 'dataset_config'")
        
    with open(dataset_card_path, "w") as f:
        yaml.dump(dataset_card_data, f, default_flow_style=False)
    print(f"Dataset card config written to: {dataset_card_path}")
    print(f"    data path set to: {PARQUET_DATA_ROOT}")