import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

from paddleocr import PaddleOCRVL

IMAGE_DIR = PROJECT_ROOT / "tmp/42e59745cdb54b6fb2c635d7c11dbd43"
img_path = str(IMAGE_DIR / "page_1.jpg")
print(f"Image exists: {Path(img_path).exists()}")

print("Initializing PaddleOCRVL...")
pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    device="cpu",
)
print("Pipeline initialized!")

print(f"Predicting: {img_path}")
try:
    output = pipeline.predict(img_path)
    print("Success! Output type:", type(output))
    print("Output length:", len(output) if hasattr(output, '__len__') else "N/A")
    for i, res in enumerate(output):
        print(f"  Result {i}: type={type(res)}")
        if hasattr(res, 'save_to_markdown'):
            print("  Has save_to_markdown")
except Exception as e:
    import traceback
    traceback.print_exc()
