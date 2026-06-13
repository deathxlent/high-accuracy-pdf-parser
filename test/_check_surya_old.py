import sys
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=== Testing surya-ocr 0.4.5 imports...")

try:
    from surya.ordering import batch_ordering
    print("✓ surya.ordering.batch_ordering imported")
except Exception as e:
    print(f"✗ surya.ordering.batch_ordering failed: {e}")

try:
    from surya.model.ordering.model import load_model as order_load_model
    print("✓ surya.model.ordering.model imported")
except Exception as e:
    print(f"✗ surya.model.ordering.model failed: {e}")

try:
    from surya.model.ordering.processor import load_processor as order_load_processor
    print("✓ surya.model.ordering.processor imported")
except Exception as e:
    print(f"✗ surya.model.ordering.processor failed: {e}")

try:
    import surya.ordering
    print(f"✓ surya.ordering module available")
    contents = [x for x in dir(surya.ordering) if not x.startswith('_')]
    print(f"  Contents: {contents}")
except Exception as e:
    print(f"✗ surya.ordering failed: {e}")
