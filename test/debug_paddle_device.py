import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
print(f"Set CUDA_VISIBLE_DEVICES = '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

import paddle
print(f"\nPaddle version: {paddle.__version__}")
print(f"Compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")
print(f"CUDA_VISIBLE_DEVICES after import: '{os.environ.get('CUDA_VISIBLE_DEVICES')}'")

print(f"\nDefault device: {paddle.device.get_device()}")

test_tensor = paddle.to_tensor([2, 3, 4])
print(f"\nTensor: {test_tensor}")
print(f"Tensor place: {test_tensor.place}")
print(f"Tensor prod(): {test_tensor.prod()}")
print(f"Tensor numpy().prod(): {test_tensor.numpy().prod()}")

print(f"\n--- Test with explicit CPU place ---")
cpu_tensor = paddle.to_tensor([2, 3, 4], place="cpu")
print(f"CPU Tensor: {cpu_tensor}")
print(f"CPU Tensor place: {cpu_tensor.place}")
print(f"CPU Tensor prod(): {cpu_tensor.prod()}")
print(f"CPU Tensor numpy().prod(): {cpu_tensor.numpy().prod()}")

print(f"\n--- Try to set default device to cpu ---")
paddle.device.set_device("cpu")
print(f"Default device after set_device: {paddle.device.get_device()}")

test_tensor2 = paddle.to_tensor([2, 3, 4])
print(f"Tensor after set_device: {test_tensor2}")
print(f"Tensor place: {test_tensor2.place}")
print(f"Tensor prod(): {test_tensor2.prod()}")
