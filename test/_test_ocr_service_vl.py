import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services import ocr_service_vl as ocr_vl
import time

image_path = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"

print("Testing ocr_service_vl...")
print(f"Image: {image_path}")

# Test 1: Full page OCR
print("\n--- Test 1: Full page OCR ---")
start = time.time()
text = ocr_vl.ocr_region(image_path)
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.1f}s")
print(f"Output ({len(text)} chars):")
print(text[:300])

# Test 2: Table extraction with bounding box
print("\n--- Test 2: Table extraction ---")
# Approximate table bbox from page 1
bbox = (100, 500, 1500, 1000)
start = time.time()
table_result = ocr_vl.extract_table_with_vl(image_path, bbox)
elapsed = time.time() - start
print(f"Elapsed: {elapsed:.1f}s")
print(f"Rows: {table_result['rows']}, Cols: {table_result['cols']}")
print(f"HTML size: {len(table_result['html'])}")
print("HTML preview:")
print(table_result['html'][:500])
