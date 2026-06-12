"""Check Q8_0 output and generate report."""
import re, sys, os

RAW = r'test\output_ocr_vl_gguf\q8_raw.txt'
OUT = 'output_ocr_vl_q8'

with open(RAW, 'rb') as f:
    raw = f.read()

cleaned = re.sub(rb'\x1b\[[0-9;]*[a-zA-Z]', b'', raw)
parts = cleaned.split(b'Table Recognition:')
resp = parts[1].strip() if len(parts) >= 2 else cleaned
resp = re.sub(b'\n\\[ Prompt:.*$', b'', resp, flags=re.DOTALL)

for enc in ['utf-8', 'gb18030', 'gbk', 'cp936']:
    try:
        text = resp.decode(enc)
        break
    except UnicodeDecodeError:
        continue

perf = re.search(b'\\[ Prompt:\\s*([\\d.]+)\\s*t/s \\| Generation:\\s*([\\d.]+)\\s*t/s', cleaned)
if perf:
    print(f"Q8_0 perf: Prompt {perf.group(1).decode()} t/s | Gen {perf.group(2).decode()} t/s")

print(f"Response: {len(text)} chars")
print(f"Has fcel: {'<fcel>' in text}")
print(f"Has 表5-10: {'表5-10' in text}")
print(f"Has G75: {'G75' in text}")
print(f"Has 崇遵: {'崇遵' in text}")
print()
# Save the extracted text for report generation
os.makedirs(os.path.join('test', OUT), exist_ok=True)
with open(os.path.join('test', OUT, 'extracted.txt'), 'w', encoding='utf-8') as f:
    f.write(text)

# Also save the performance info
p = {'prompt': None, 'gen': None}
if perf:
    p['prompt'] = perf.group(1).decode()
    p['gen'] = perf.group(2).decode()
import json
with open(os.path.join('test', OUT, 'perf.json'), 'w') as f:
    json.dump(p, f)

print("Saved extracted text and perf info")
print()
# Show beginning and end of structured output
idx = text.find('<fcel>')
if idx >= 0:
    print("=== START ===")
    print(text[idx:idx+400])
    print("...")
    print("=== END ===")
    print(text[-400:])
