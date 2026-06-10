import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import torch
except OSError:
    pass

import paddle

print(f"Paddle version: {paddle.__version__}")
print(f"Default device: {paddle.device.get_device()}")

_original_prod = paddle.Tensor.prod

def _patched_prod(self, *args, **kwargs):
    if self.place.is_gpu_place():
        cpu_tensor = self.cpu()
        result = _original_prod(cpu_tensor, *args, **kwargs)
        return result
    return _original_prod(self, *args, **kwargs)

paddle.Tensor.prod = _patched_prod

print("\n--- Test patched prod() on GPU ---")
test_tensor = paddle.to_tensor([2, 3, 4])
print(f"Tensor: {test_tensor}")
print(f"Tensor place: {test_tensor.place}")
print(f"Patched prod(): {test_tensor.prod()}")
print(f"Expected: 24")

print("\n--- Test patched prod() on CPU ---")
cpu_tensor = paddle.to_tensor([2, 3, 4], place=paddle.CPUPlace())
print(f"Tensor: {cpu_tensor}")
print(f"Tensor place: {cpu_tensor.place}")
print(f"Patched prod(): {cpu_tensor.prod()}")
print(f"Expected: 24")

print("\n--- Test 2D tensor ---")
test_2d = paddle.to_tensor([[1, 2, 3], [4, 5, 6]])
print(f"Tensor:\n{test_2d}")
print(f"prod() all: {test_2d.prod()}")
print(f"Expected: 720")

print("\nAll tests done!")
