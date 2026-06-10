import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

print("Testing import order...")
print("1. First, try to import torch (may fail)...")
try:
    import torch
    print(f"   torch imported successfully: {torch.__version__}")
except OSError as e:
    print(f"   torch import failed (expected): {e}")

print("\n2. Now import paddle...")
import paddle
paddle.device.set_device("cpu")
print(f"   paddle imported: {paddle.__version__}")
print(f"   default device: {paddle.device.get_device()}")

print("\n3. Now try to import paddleocr...")
try:
    import paddleocr
    print(f"   paddleocr imported: {paddleocr.__version__}")
except Exception as e:
    print(f"   paddleocr import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone.")
