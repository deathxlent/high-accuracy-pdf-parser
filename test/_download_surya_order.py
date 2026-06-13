import os
import json
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import hf_hub_download
from pathlib import Path

MODELS_DIR = Path(r"c:\ws\high accuracy pdf parser\models")
SURYA_ORDER_DIR = MODELS_DIR / "surya_order"
SURYA_ORDER_DIR.mkdir(exist_ok=True)

repo_id = "vikp/surya_order"
files = ["config.json", "generation_config.json", "model.safetensors", "preprocessor_config.json"]

for f in files:
    print(f"Downloading {f}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=f,
        repo_type="model",
        local_dir=str(SURYA_ORDER_DIR),
    )
    print(f"  Saved to: {local_path}")

print("\n=== Config ===")
with open(SURYA_ORDER_DIR / "config.json") as f:
    config = json.load(f)
    print(json.dumps(config, indent=2))

print("\n=== Preprocessor Config ===")
with open(SURYA_ORDER_DIR / "preprocessor_config.json") as f:
    preproc = json.load(f)
    print(json.dumps(preproc, indent=2))

print("\n=== Generation Config ===")
with open(SURYA_ORDER_DIR / "generation_config.json") as f:
    gen_config = json.load(f)
    print(json.dumps(gen_config, indent=2))
