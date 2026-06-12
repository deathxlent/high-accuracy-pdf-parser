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

image_path = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
print(f"Testing with image: {image_path}")

base64_image = encode_image(image_path)

# Test: Use /completion endpoint
url = "http://127.0.0.1:8080/completion"
headers = {"Content-Type": "application/json"}

# Try with image_data in completion request
payload = {
    "prompt": "Table Recognition:",
    "image_data": [{"data": base64_image, "id": 1}],
    "temperature": 0,
    "n_predict": 800,
    "stream": False
}

print("--- Test: /completion endpoint with Table Recognition prompt ---")
start = time.time()
try:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")
        content = result.get('content', '')
        print(f"Content ({len(content)} chars):")
        print(content[:800])
        print(f"\nContains <fcel>: {'<fcel>' in content}")
        print(f"Timings: {result.get('timings', {})}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: /v1/chat/completions with proper chat template
url2 = "http://127.0.0.1:8080/v1/chat/completions"
payload2 = {
    "model": "PaddleOCR-VL-1.6.Q4_K_M.gguf",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": "Table Recognition:"}
            ]
        }
    ],
    "temperature": 0,
    "max_tokens": 800,
    "stream": False
}

print("\n--- Test: /v1/chat/completions (image first, then text) ---")
start = time.time()
try:
    data = json.dumps(payload2).encode('utf-8')
    req = urllib.request.Request(url2, data=data, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=180) as resp:
        result = json.loads(resp.read().decode('utf-8'))
        elapsed = time.time() - start
        print(f"Elapsed: {elapsed:.1f}s")
        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"Content ({len(content)} chars):")
            print(content[:800])
            print(f"\nContains <fcel>: {'<fcel>' in content}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
