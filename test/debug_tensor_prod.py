import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import paddle

print("Testing Paddle Tensor prod() behavior:")
print()

# Simulate the image_grid_thw
data = np.array([[1, 84, 60]], dtype=np.int64)
print(f"Input data: {data}")
print(f"Data shape: {data.shape}")
print(f"Data prod (numpy): {data[0].prod()}")
print()

tensor = paddle.to_tensor(data)
print(f"Tensor: {tensor}")
print(f"Tensor shape: {tensor.shape}")
print()

# Method 1: tensor[0].prod()
result1 = tensor[0].prod()
print(f"Method 1: tensor[0].prod()")
print(f"  Result: {result1}")
print(f"  Result type: {type(result1)}")
print(f"  Result shape: {result1.shape}")
try:
    print(f"  int(result): {int(result1)}")
except Exception as e:
    print(f"  int(result) error: {e}")
print()

# Method 2: paddle.prod(tensor[0])
result2 = paddle.prod(tensor[0])
print(f"Method 2: paddle.prod(tensor[0])")
print(f"  Result: {result2}")
print(f"  Result type: {type(result2)}")
print(f"  Result shape: {result2.shape}")
try:
    print(f"  int(result): {int(result2)}")
except Exception as e:
    print(f"  int(result) error: {e}")
print()

# Method 3: tensor[0].numpy().prod()
result3 = tensor[0].numpy().prod()
print(f"Method 3: tensor[0].numpy().prod()")
print(f"  Result: {result3}")
print(f"  Result type: {type(result3)}")
print(f"  int(result): {int(result3)}")
print()

# Method 4: tensor[0].sum() to check if sum also has the same issue
result4 = tensor[0].sum()
print(f"Method 4: tensor[0].sum() (should be 1+84+60=145)")
print(f"  Result: {result4}")
try:
    print(f"  int(result): {int(result4)}")
except Exception as e:
    print(f"  int(result) error: {e}")
print()

# Method 5: check prod with axis parameter
result5 = tensor[0].prod(axis=0)
print(f"Method 5: tensor[0].prod(axis=0)")
print(f"  Result: {result5}")
try:
    print(f"  int(result): {int(result5)}")
except Exception as e:
    print(f"  int(result) error: {e}")
