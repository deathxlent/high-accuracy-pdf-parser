import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import paddle

print("Testing Paddle Tensor behavior on CPU vs GPU:")
print()

data = np.array([[1, 84, 60]], dtype=np.int64)
print(f"Input data: {data}")
print(f"Expected prod: {data[0].prod()}")
print(f"Expected sum: {data[0].sum()}")
print()

# Force CPU
print("=" * 60)
print("Testing on CPU place:")
print("=" * 60)
with paddle.device_guard("cpu"):
    tensor_cpu = paddle.to_tensor(data, place=paddle.CPUPlace())
    print(f"Tensor place: {tensor_cpu.place}")
    
    r1 = tensor_cpu[0].prod()
    print(f"tensor_cpu[0].prod() = {int(r1)} (expected: 5040)")
    
    r2 = tensor_cpu[0].sum()
    print(f"tensor_cpu[0].sum() = {int(r2)} (expected: 145)")
    
    r3 = tensor_cpu[0].numpy().prod()
    print(f"tensor_cpu[0].numpy().prod() = {int(r3)} (expected: 5040)")

print()
print("=" * 60)
print("Testing float64 dtype:")
print("=" * 60)
data_float = np.array([[1.0, 84.0, 60.0]], dtype=np.float64)
with paddle.device_guard("cpu"):
    tensor_f64 = paddle.to_tensor(data_float, place=paddle.CPUPlace())
    r = tensor_f64[0].prod()
    print(f"float64 tensor[0].prod() = {float(r)} (expected: 5040.0)")
    
    r = tensor_f64[0].sum()
    print(f"float64 tensor[0].sum() = {float(r)} (expected: 145.0)")

print()
print("=" * 60)
print("Testing int32 dtype:")
print("=" * 60)
data_int32 = np.array([[1, 84, 60]], dtype=np.int32)
with paddle.device_guard("cpu"):
    tensor_i32 = paddle.to_tensor(data_int32, place=paddle.CPUPlace())
    r = tensor_i32[0].prod()
    print(f"int32 tensor[0].prod() = {int(r)} (expected: 5040)")
    
    r = tensor_i32[0].sum()
    print(f"int32 tensor[0].sum() = {int(r)} (expected: 145)")
