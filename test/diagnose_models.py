"""Diagnose which model/mmproj combinations work."""
import subprocess, sys, os, time, re

CLI = r"G:\llamacpp\llama-cli.exe"
IMAGE = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
BASE_MODEL = r"C:\ws\high accuracy pdf parser\models\PaddleOCR-VL-1.6.Q4_K_M.gguf"
F16 = r"G:\llamacpp\models\PaddleOCR-VL-1.6.f16.gguf"
Q8 = r"G:\llamacpp\models\PaddleOCR-VL-1.6.Q8_0.gguf"
MMPROJ_16 = r"G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
MMPROJ_MS = r"G:\llamacpp\models\PaddleOCR-VL-GGUF-mmproj.gguf"

tests = [
    ("Q4_K_M (project) + 1.6 mmproj", BASE_MODEL, MMPROJ_16),
    ("F16 (official?) + 1.6 mmproj", F16, MMPROJ_16),
    ("F16 + ModelScope mmproj", F16, MMPROJ_MS),
    ("Q8_0 + 1.6 mmproj", Q8, MMPROJ_16),
    ("Q8_0 + ModelScope mmproj", Q8, MMPROJ_MS),
]

for label, model, mmproj in tests:
    if not os.path.exists(model):
        print(f"SKIP {label}: model not found")
        continue
    if not os.path.exists(mmproj):
        print(f"SKIP {label}: mmproj not found")
        continue

    print(f"\n=== {label} ===")
    sys.stdout.flush()

    cmd = [CLI, "-m", model, "--mmproj", mmproj,
           "--image", IMAGE, "--temp", "0", "-p", "OCR:",
           "-n", "200", "--no-display-prompt"]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        elapsed = time.time() - start
        std = result.stdout
        print(f"  OK: {elapsed:.1f}s, {len(std)} bytes, rc={result.returncode}")

        txt = std.decode("utf-8", errors="replace")
        txt = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", txt)
        resp = txt.split("OCR:", 1)[1].strip() if "OCR:" in txt else txt.strip()
        resp = re.sub(r"\n\[ Prompt:.*$", "", resp, flags=re.DOTALL).strip()
        resp = re.sub(r"\n>.*$", "", resp, flags=re.DOTALL).strip()

        gc = resp.count("0" * 6)
        if gc > 5:
            print(f"  GARBAGE: {gc} x '000000' sequences")
        elif len(resp) > 10:
            print(f"  VALID: {len(resp)} chars, first 200: {resp[:200]}")
        else:
            print(f"  SHORT: '{resp}'")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 300s")
    except Exception as e:
        print(f"  ERROR: {e}")
