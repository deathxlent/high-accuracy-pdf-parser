"""Test official PaddleOCR-VL-1.6 F16 and Q8_0 models with correct mmproj."""
import subprocess, sys, os, time, re

MMPROJ = r"G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf"
IMAGE = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
CLI = r"G:\llamacpp\llama-cli.exe"
BASE_OUT = r"C:\ws\high accuracy pdf parser\test"

TESTS = [
    ("F16", r"G:\llamacpp\models\PaddleOCR-VL-1.6.f16.gguf",
     "output_ocr_vl_f16", "892 MB", "F16"),
    ("Q8_0", r"G:\llamacpp\models\PaddleOCR-VL-1.6.Q8_0.gguf",
     "output_ocr_vl_q8", "475 MB", "Q8_0"),
]

for name, model_path, out_sub, size, quant in TESTS:
    out_dir = os.path.join(BASE_OUT, out_sub)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== Testing {name} ({size}) ===")
    sys.stdout.flush()

    cmd = [CLI, "-m", model_path, "--mmproj", MMPROJ,
           "--image", IMAGE, "--temp", "0", "-p", "OCR:",
           "--no-display-prompt"]

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        elapsed = time.time() - start
        stdout = result.stdout
        rc = result.returncode
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        continue
    except Exception as e:
        print(f"  ERROR: {e}")
        continue

    # Save raw output
    with open(os.path.join(out_dir, "llama_cli_raw.txt"), "wb") as f:
        f.write(stdout)
    with open(os.path.join(out_dir, "elapsed.txt"), "w") as f:
        f.write(f"{elapsed:.1f}")

    print(f"  Done in {elapsed:.1f}s (rc={rc}), {len(stdout)} bytes")

    # Parse OCR response
    text = stdout.decode("utf-8", errors="replace")
    clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
    if "OCR:" in clean:
        resp = clean.split("OCR:", 1)[1].strip()
    else:
        resp = clean.strip()
    resp = re.sub(r"\n\[ Prompt:.*$", "", resp, flags=re.DOTALL).strip()
    resp = re.sub(r"\n>.*$", "", resp, flags=re.DOTALL).strip()

    # Quality check
    garbage_count = resp.count("000000")
    if garbage_count > 10:
        print(f"  WARNING: Garbage output ({garbage_count} x '000000')")
    else:
        print(f"  OK: {len(resp)} chars of output")
        print(f"  Preview: {resp[:200]}")
