import sys
import os
import time
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from backend.services.layout_service import detect_layout, _get_model, _get_inference_kwargs
from backend.services.order_service import assign_reading_order, _get_ordering_model_and_processor

print("=" * 60)
print("Test 1: YOLO GPU")
print("=" * 60)

jpg_path = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
print(f"Test image: {jpg_path}")

print(f"\nLoading YOLO model...")
model = _get_model()
print(f"Model device: {model.device}")

kwargs = _get_inference_kwargs()
print(f"Inference kwargs: {kwargs}")

print(f"\nRunning layout detection on {jpg_path}...")
start = time.time()
elements = detect_layout(jpg_path)
elapsed = time.time() - start
print(f"Found {len(elements)} elements in {elapsed:.2f}s")

for i, elem in enumerate(elements[:5]):
    print(f"  [{i}] {elem['element_type']}: bbox={elem['bbox']}, conf={elem['confidence']:.3f}")

print(f"\n" + "=" * 60)
print("Test 2: Surya Ordering")
print("=" * 60)

print(f"\nLoading Surya ordering model...")
surya_model, surya_processor = _get_ordering_model_and_processor()
if surya_model is None:
    print("FAILED: Surya model not loaded")
    sys.exit(1)

print(f"Surya model device: {surya_model.device}")
print(f"Surya model dtype: {surya_model.dtype}")

print(f"\nRunning reading order assignment...")
start = time.time()
ordered_elements = assign_reading_order(elements, jpg_path)
elapsed = time.time() - start
print(f"Ordering completed in {elapsed:.2f}s")

print(f"\nOrdered elements ({len(ordered_elements)} total):")
for i, elem in enumerate(ordered_elements[:10]):
    print(f"  order={elem['reading_order']:2d}: {elem['element_type']:15s} bbox=({elem['bbox'][0]:.0f}, {elem['bbox'][1]:.0f}, {elem['bbox'][2]:.0f}, {elem['bbox'][3]:.0f})")

print(f"\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
