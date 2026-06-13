import sys
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from surya.ordering import batch_ordering, OrderBox, OrderResult
import inspect

print("=== batch_ordering signature ===")
print(inspect.signature(batch_ordering))

print("\n=== batch_ordering source (first 50 lines) ===")
try:
    source = inspect.getsource(batch_ordering)
    lines = source.split('\n')
    for i, line in enumerate(lines[:50]):
        print(f"{i+1:3d}: {line}")
except:
    print("Could not get source")

print("\n=== OrderBox ===")
print(inspect.signature(OrderBox.__init__))
try:
    print(OrderBox.__doc__)
except:
    pass

print("\n=== OrderResult ===")
try:
    print(OrderResult.__doc__)
except:
    pass
print(dir(OrderResult))
