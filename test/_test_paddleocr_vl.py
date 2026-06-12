"""Test PaddleOCRVL with GGUF model"""
import os, sys, time
os.environ["PADDLE_PDX_CACHE_HOME"] = "C:\\paddlex_cache"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCRVL

IMG_PATH = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
MODEL_PATH = r"C:\ws\high accuracy pdf parser\models\PaddleOCR-VL-1.6.Q4_K_M.gguf"

# Approach 1: Use pipeline_version with local path
print("=" * 60)
print("Test 1: PaddleOCRVL with GGUF model via pipeline_version")
print("=" * 60)
try:
    t0 = time.time()
    pipeline = PaddleOCRVL(
        pipeline_version=MODEL_PATH,
        device="gpu",
    )
    t1 = time.time()
    print(f"Init: {t1-t0:.1f}s")
    
    result = pipeline.predict(IMG_PATH)
    t2 = time.time()
    print(f"Predict: {t2-t1:.1f}s")
    
    for i, res in enumerate(result):
        res.print()
        res.save_to_markdown(save_path=r"C:\ws\high accuracy pdf parser\test\output_vl_gguf")
        res.save_to_json(save_path=r"C:\ws\high accuracy pdf parser\test\output_vl_gguf")
except Exception as e:
    print(f"Test 1 failed: {e}")

print()
print("=" * 60)
print("Test 2: PaddleOCRVL with default v1.6")
print("=" * 60)
try:
    t0 = time.time()
    pipeline = PaddleOCRVL(
        pipeline_version="v1.6",
        device="gpu",
    )
    t1 = time.time()
    print(f"Init: {t1-t0:.1f}s")
    
    result = pipeline.predict(IMG_PATH)
    t2 = time.time()
    print(f"Predict: {t2-t1:.1f}s")
    
    for i, res in enumerate(result):
        res.print()
        res.save_to_markdown(save_path=r"C:\ws\high accuracy pdf parser\test\output_vl_default")
        res.save_to_json(save_path=r"C:\ws\high accuracy pdf parser\test\output_vl_default")
except Exception as e:
    print(f"Test 2 failed: {e}")
