import sys
import os
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import inspect
from surya.model.ordering.model import load_model

print("=== load_model source ===")
print(inspect.getsource(load_model))
