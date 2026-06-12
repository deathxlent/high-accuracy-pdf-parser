import re

with open(r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\report.html', 'r', encoding='utf-8') as f:
    content = f.read()

m = re.search(r'<pre>(.+?)</pre>', content, re.DOTALL)
if m:
    text = m.group(1)
    print(f'Length: {len(text)} chars')
    print(f'First 100 chars: {text[:100]}')
    # Check specific Chinese chars
    targets = ['贵州', '高速', '公路', '集团']
    for t in targets:
        print(f'Contains "{t}": {t in text}')
    # Print all unique non-ASCII chars at the start
    print()
    print('First 300 chars raw:')
    print(repr(text[:300]))
