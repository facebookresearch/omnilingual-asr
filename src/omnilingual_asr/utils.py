import subprocess

def get_idiom_name_by_folder(folder_name):
  name = (folder_name.split("-")[0])[2:]
  match name:
    case "sursilv":
      return "Sursilvan"
    case "surmiran":
      return "Surmiran"
    case "sutsilv":
      return "Sutsilvan"
    case "puter":
      return "Puter"
    case "vallader":
      return "Vallader"
    case _:
      return "RG"

def get_best_gpu():
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