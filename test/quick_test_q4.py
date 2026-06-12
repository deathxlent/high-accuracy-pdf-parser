"""Quick test: run Q4_K_M baseline and check it works."""
import subprocess, time, re, sys, os

cmd = [
    r"G:\llamacpp\llama-cli.exe",
    "-m", r"C:\ws\high accuracy pdf parser\models\PaddleOCR-VL-1.6.Q4_K_M.gguf",
    "--mmproj", r"G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf",
    "--image", r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg",
    "--temp", "0", "-p", "OCR:", "-n", "200", "--no-display-prompt",
]
print("Running Q4_K_M baseline...")
sys.stdout.flush()
start = time.time()
result = subprocess.run(cmd, capture_output=True, timeout=300)
elapsed = time.time() - start
print(f"Done: {elapsed:.1f}s, {len(result.stdout)} bytes, rc={result.returncode}")

text = result.stdout.decode("utf-8", errors="replace")
clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", text)
resp = clean.split("OCR:", 1)[1].strip() if "OCR:" in clean else clean.strip()
resp = re.sub(r"\n\[ Prompt:.*$", "", resp, flags=re.DOTALL).strip()
resp = re.sub(r"\n>.*$", "", resp, flags=re.DOTALL).strip()
print(f"Response ({len(resp)} chars):")
print(resp[:300])
print("---")
if "000000" in resp[:100]:
    print("GARBAGE DETECTED")
else:
    print("OUTPUT LOOKS VALID")
