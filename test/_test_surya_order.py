import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from backend.services.order_service import _get_ordering_model_and_processor, assign_reading_order

print("=== Testing Surya Order Model ===")
model, processor = _get_ordering_model_and_processor()

if model is None:
    print("FAILED: Model not loaded!")
    sys.exit(1)

print("SUCCESS: Model loaded!")
print(f"  Model type: {type(model).__name__}")
print(f"  Processor type: {type(processor).__name__}")

print("\n=== Testing with a simple image and bboxes ===")

from PIL import Image

test_img_path = None
import sqlite3
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

page = cur.execute("""
    SELECT p.id, p.page_number, p.jpg_path, p.document_id
    FROM pdf_pages p
    JOIN pdf_documents d ON p.document_id = d.id
    WHERE d.id = 10 AND p.page_number = 1
""").fetchone()

if page:
    print(f"Found doc10 page1 jpg:", page[2])
    test_img_path = page[2]
else:
    print("Could not find page10 page1")

conn.close()

if test_img_path and Path(test_img_path).exists():
    from backend.services.layout_service import detect_layout
    elements = detect_layout(test_img_path)
    print(f"Layout detected: {len(elements)} elements")
    for e in elements[:5]:
        print(f"  {e['element_type']} at bbox={e['bbox']} conf={e['confidence']:.3f}")
    
    print("\nCalling assign_reading_order...")
    ordered = assign_reading_order(elements, test_img_path)
    
    print(f"\nOrdered {len(ordered)} elements after ordering:")
    for e in ordered:
        print(f"  #{e['reading_order']} {e['element_type']}")
    
    print("\n=== Surya Order Test PASSED ===")
else:
    print("No test image available, skipping full test.")
    print("=== Model Load Test PASSED ===")
