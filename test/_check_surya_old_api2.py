import sys
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from surya.ordering import OrderResult
from surya.model.ordering.model import load_model
from surya.model.ordering.processor import load_processor
import inspect

print("=== OrderResult fields ===")
print(OrderResult.model_fields.keys() if hasattr(OrderResult, 'model_fields') else dir(OrderResult))
for field_name, field_info in OrderResult.model_fields.items():
    print(f"  {field_name}: {field_info.annotation}")

print("\n=== load_model signature ===")
print(inspect.signature(load_model))

print("\n=== load_processor signature ===")
print(inspect.signature(load_processor))

print("\n=== rank_elements ===")
from surya.ordering import rank_elements
print(inspect.signature(rank_elements))
try:
    source = inspect.getsource(rank_elements)
    print(source[:1000])
except:
    pass
