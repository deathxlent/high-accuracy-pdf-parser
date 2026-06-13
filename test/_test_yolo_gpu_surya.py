import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(level=logging.INFO)

from backend.services.layout_service import detect_layout, _get_model

img_path = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"
print(f"Testing YOLO with image: {img_path}")
print(f"Image exists: {Path(img_path).exists()}")

import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

model = _get_model()
print(f"Model device: {model.device}")

elements = detect_layout(img_path)
print(f"Found {len(elements)} elements")
for i, e in enumerate(elements[:5]):
    print(f"  {i}: {e['element_type']} conf={e['confidence']:.3f} bbox=({e['bbox'][0]:.0f},{e['bbox'][1]:.0f})-({e['bbox'][2]:.0f},{e['bbox'][3]:.0f})")

print()
print("Testing surya-ocr detection...")
try:
    from PIL import Image
    from surya.detection import DetectionPredictor
    from surya.settings import settings
    print(f"DetectionPredictor found")
    
    # Check detection output schema
    from surya.detection.schema import TextDetectionResult
    print(f"TextDetectionResult: {dir(TextDetectionResult)}")
    
except Exception as e:
    print(f"Surya detection test failed: {e}")
    import traceback
    traceback.print_exc()
