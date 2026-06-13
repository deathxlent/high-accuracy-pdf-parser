import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import torch
from backend.services.layout_service import _get_model, detect_layout
from backend.config import YOLO_DEVICE, YOLO_MODEL_FILE, MODELS_DIR

print(f"YOLO_DEVICE config: {YOLO_DEVICE}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")

print("\nLoading model...")
start = time.time()
model = _get_model()
print(f"Model loaded in {time.time()-start:.2f}s")

print(f"\nModel device: {model.device}")

test_img = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"
print(f"\nTesting inference on {test_img}...")
start = time.time()
results = detect_layout(test_img)
print(f"Inference done in {time.time()-start:.2f}s")
print(f"Found {len(results)} elements")
for i, r in enumerate(results[:3]):
    print(f"  #{i}: {r['element_type']} conf={r['confidence']:.3f} bbox={r['bbox']}")

print("\nDone!")
