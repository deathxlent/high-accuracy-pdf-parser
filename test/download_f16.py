"""Download PaddleOCR-VL-1.6-GGUF.gguf (F16, ~936MB) from hf-mirror.com"""
import urllib.request, sys, os, time

url = "https://hf-mirror.com/PaddlePaddle/PaddleOCR-VL-1.6-GGUF/resolve/main/PaddleOCR-VL-1.6-GGUF.gguf"
dest = r"G:\llamacpp\models\PaddleOCR-VL-1.6.f16.gguf"
tmp = dest + ".downloading"

# Remove stale partial
if os.path.exists(tmp):
    os.remove(tmp)

start = time.time()
last_report = [0]

def reporthook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    elapsed = time.time() - start
    if total_size > 0 and downloaded > 0:
        pct = downloaded * 100 // total_size
        speed = downloaded / (1024*1024) / elapsed if elapsed > 0 else 0
        eta = (total_size - downloaded) / (downloaded / elapsed) if downloaded > 0 and elapsed > 0 else 0
        sys.stdout.write(f"\r{pct}%  {downloaded//1024//1024}MB/{total_size//1024//1024}MB  {speed:.1f}MB/s  ETA {eta:.0f}s")
        sys.stdout.flush()
    elif downloaded > 0:
        sys.stdout.write(f"\r{downloaded//1024//1024} MB downloaded...")
        sys.stdout.flush()

print(f"Downloading to: {dest}")
print(f"URL: {url}")
urllib.request.urlretrieve(url, tmp, reporthook)
os.rename(tmp, dest)
elapsed = time.time() - start
size_mb = os.path.getsize(dest) / (1024*1024)
print(f"\nDone! {size_mb:.0f} MB in {elapsed:.0f}s ({size_mb/elapsed:.1f} MB/s)")
