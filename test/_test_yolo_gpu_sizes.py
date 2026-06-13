import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from ultralytics import YOLO
from backend.config import MODELS_DIR, YOLO_MODEL_FILE

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.mem_get_info(): {torch.cuda.mem_get_info() if hasattr(torch.cuda, 'mem_get_info') else 'N/A'}")

model_path = MODELS_DIR / YOLO_MODEL_FILE
print(f"\nLoading model from {model_path}...")
model = YOLO(str(model_path))

test_img = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"

for imgsz in [640, 800, 1024, 1280]:
    print(f"\n--- Testing imgsz={imgsz} on GPU ---")
    try:
        torch.cuda.empty_cache()
        start = time.time()
        results = model(test_img, imgsz=imgsz, device='cuda', verbose=False, half=True)
        elapsed = time.time() - start
        print(f"  SUCCESS! Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            print(f"  GPU Memory: allocated={allocated:.1f}MB, reserved={reserved:.1f}MB")
    except Exception as e:
        print(f"  FAILED: {e}")
        torch.cuda.empty_cache()

print("\n--- Testing imgsz=640 on CPU for comparison ---")
try:
    start = time.time()
    results = model(test_img, imgsz=640, device='cpu', verbose=False)
    elapsed = time.time() - start
    print(f"  CPU Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
except Exception as e:
    print(f"  CPU FAILED: {e}")
