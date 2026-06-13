import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from ultralytics import YOLO
from backend.config import MODELS_DIR, YOLO_MODEL_FILE

model_path = MODELS_DIR / YOLO_MODEL_FILE
model = YOLO(str(model_path))

test_img = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"

print("=== Test 1: imgsz=1280, device='cuda', no half ===")
try:
    torch.cuda.empty_cache()
    start = time.time()
    results = model(test_img, imgsz=1280, device='cuda', verbose=False)
    elapsed = time.time() - start
    print(f"  SUCCESS! Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test 2: imgsz=1280, device='cuda', half=True ===")
try:
    torch.cuda.empty_cache()
    start = time.time()
    results = model(test_img, imgsz=1280, device='cuda', verbose=False, half=True)
    elapsed = time.time() - start
    print(f"  SUCCESS! Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n=== Test 3: imgsz=1280, device=0 (GPU id), half=True ===")
try:
    torch.cuda.empty_cache()
    start = time.time()
    results = model(test_img, imgsz=1280, device=0, verbose=False, half=True)
    elapsed = time.time() - start
    print(f"  SUCCESS! Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n=== Test 4: Check image size ===")
from PIL import Image
img = Image.open(test_img)
print(f"  Image size: {img.size}")
print(f"  Image mode: {img.mode}")

print("\n=== Test 5: imgsz=1280 with explicit model.to('cuda') first ===")
try:
    model.to('cuda')
    torch.cuda.empty_cache()
    start = time.time()
    results = model(test_img, imgsz=1280, verbose=False)
    elapsed = time.time() - start
    print(f"  SUCCESS! Time: {elapsed:.2f}s, {len(results[0].boxes)} boxes")
except Exception as e:
    print(f"  FAILED: {e}")
