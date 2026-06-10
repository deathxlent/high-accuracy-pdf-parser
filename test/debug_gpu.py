import paddle

print("Paddle compiled with CUDA:", paddle.device.is_compiled_with_cuda())
print("Paddle version:", paddle.__version__)

if paddle.device.is_compiled_with_cuda():
    print("CUDA version:", paddle.version.cuda())
    gpus = paddle.device.cuda.device_count()
    print("GPU count:", gpus)
    if gpus > 0:
        print("GPU name:", paddle.device.cuda.get_device_name(0))
        print("GPU capability:", paddle.device.cuda.get_device_capability(0))
