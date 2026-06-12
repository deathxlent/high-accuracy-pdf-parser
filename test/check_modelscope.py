"""Check what files are on ModelScope for PaddleOCR-VL-GGUF"""
import urllib.request, json, re, sys

# Try API to list files
url = "https://www.modelscope.cn/api/v1/models/megemini/PaddleOCR-VL-GGUF/revisions/master/files"
try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        for f in data.get("Data", []):
            print(f.get("Path", "?"))
except Exception as e:
    print(f"API error: {e}", file=sys.stderr)
    # Fallback: scrape the page
    url2 = "https://www.modelscope.cn/models/megemini/PaddleOCR-VL-GGUF"
    try:
        req2 = urllib.request.Request(url2, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req2, timeout=30) as resp2:
            html = resp2.read().decode()
            files = re.findall(r'["\']([^"\']+\.gguf[^"\']*)["\']', html)
            for f in files:
                print(f"  {f}")
    except Exception as e2:
        print(f"HTML fallback error: {e2}", file=sys.stderr)

# Also check megemini direct file listing
print("---")
url3 = "https://www.modelscope.cn/models/megemini/PaddleOCR-VL-GGUF/file/view/master?file_name=README.md"
try:
    req3 = urllib.request.Request(url3, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req3, timeout=30) as resp3:
        text = resp3.read().decode()
        print(text[:2000])
except Exception as e3:
    print(f"README error: {e3}", file=sys.stderr)
