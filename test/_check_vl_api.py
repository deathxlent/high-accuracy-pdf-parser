"""Check PaddleOCRVL API"""
import os
os.environ["PADDLE_PDX_CACHE_HOME"] = "C:\\paddlex_cache"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL
import inspect

sig = inspect.signature(PaddleOCRVL.__init__)
print("PaddleOCRVL.__init__ parameters:")
for name, param in sig.parameters.items():
    if name == "self":
        continue
    default = param.default if param.default is not inspect.Parameter.empty else "(required)"
    print(f"  {name}: {default}")
