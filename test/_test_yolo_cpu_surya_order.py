import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from backend.config import YOLO_DEVICE
from backend.services.layout_service import _get_model, detect_layout
from backend.services.order_service import assign_reading_order

print(f"=== Configuration Check ===")
print(f"YOLO_DEVICE from config: '{YOLO_DEVICE}'")
print()

print(f"=== YOLO Model Load Check ===")
model = _get_model()
print(f"Model device: {model.device}")
print(f"Model type: {type(model).__name__}")
print()

import torch
print(f"Torch CUDA available: {torch.cuda.is_available()}")
print()

print(f"=== Test Layout Detection (CPU) ===")
test_image = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
if Path(test_image).exists():
    print(f"Test image: {test_image}")
    elements = detect_layout(test_image)
    print(f"Found {len(elements)} layout elements:")
    for i, elem in enumerate(elements):
        print(f"  {i}: {elem['element_type']} bbox={elem['bbox']} conf={elem['confidence']:.3f}")
    print()
    
    print(f"=== Test Reading Order ===")
    ordered_elements = assign_reading_order(elements, test_image)
    print(f"After ordering ({len(ordered_elements)} elements):")
    for elem in ordered_elements:
        print(f"  #{elem['reading_order']}: {elem['element_type']} bbox={elem['bbox']}")
else:
    print(f"Test image not found: {test_image}")

print("\n=== All checks passed! ===")
