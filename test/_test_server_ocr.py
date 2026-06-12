import urllib.request
import json
import base64
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

image_path = r"C:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"
if not Path(image_path).exists():
    image_path = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"

print(f"Testing with image: {image_path}")
print(f"Image exists: {Path(image_path).exists()}")

base64_image = encode_image(image_path)
print(f"Image encoded: {len(base64_image)} chars")

# Test 1: OCR mode
url = "http://127.0.0.1:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

payload_ocr = {
    "model": "PaddleOCR-VL-1.6.Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ],
    "temperature": 0,
    "max_tokens": 500,
    "stream": False
}

print("\n--- Test 1: OCR mode ---")
start = time.time()
try:
    data = json.dumps(payload_ocr).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"\nContent ({len(content)} chars):")
            print(content[:300])
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Table Recognition mode
print("\n--- Test 2: Table Recognition mode ---")
payload_table = {
    "model": "PaddleOCR-VL-1.6.Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Table Recognition:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ],
    "temperature": 0,
    "max_tokens": 800,
    "stream": False
}

start = time.time()
try:
    data = json.dumps(payload_table).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"Content ({len(content)} chars):")
            print(content[:500])
            print(f"\nContains <fcel>: {'<fcel>' in content}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
