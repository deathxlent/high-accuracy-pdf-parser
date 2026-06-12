"""
PaddleOCR-VL-1.6 GGUF 测试报告生成器 v3
- 支持 "OCR:" 输出的纯文本表格（空格分隔检测）
- 支持 "Table Recognition:" 输出的 <fcel> 结构化表格
- 跨行跨列合并自动渲染
"""

import re, os
from html import escape

# ─────────────────────────────────────────────
# 1. 原始输出提取
# ─────────────────────────────────────────────

def extract_ocr_and_perf(raw_bytes: bytes):
    """Extract response text and performance metrics from raw llama-cli output."""
    cleaned = re.sub(rb'\x1b\[[0-9;]*[a-zA-Z]', b'', raw_bytes)
    cleaned = cleaned.replace(b'\r\n', b'\n').replace(b'\r', b'\n')

    used_prompt = 'OCR:'
    response_bytes = cleaned
    for pp in [b'Table Recognition:', b'OCR:']:
        parts = cleaned.split(pp)
        if len(parts) >= 2:
            response_bytes = parts[1].strip()
            used_prompt = pp.decode('utf-8')
            break

    response_bytes = re.sub(b'\n\\[ Prompt:.*$', b'', response_bytes, flags=re.DOTALL)
    response_bytes = re.sub(b'\n>.*$', b'', response_bytes, flags=re.DOTALL)

    perf = re.search(b'\\[ Prompt:\\s*([\\d.]+)\\s*t/s \\| Generation:\\s*([\\d.]+)\\s*t/s', cleaned)

    # Decode: try UTF-8 first, fallback to GB18030 for Windows console redirect
    for enc in ['utf-8', 'gb18030', 'gbk', 'cp936']:
        try:
            text = response_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = response_bytes.decode('utf-8', errors='replace')

    return text.strip(), perf, used_prompt


# ─────────────────────────────────────────────
# 2. 结构化 <fcel> 表格解析
# ─────────────────────────────────────────────

def parse_fcel_structured(text: str) -> str:
    """Parse PaddleOCR-VL <fcel>/<nl>/<ucel> structured table format into HTML."""
    if '<fcel>' not in text:
        return parse_tables_plaintext(text)

    parts = text.split('<nl>')
    rows = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        is_ucel = part.startswith('<ucel>')
        if is_ucel:
            part = part[len('<ucel>'):]

        cells = []
        remaining = part
        while '<fcel>' in remaining:
            _, after = remaining.split('<fcel>', 1)
            next_pos = len(after)
            for tag in ['<fcel>', '<lcel>', '<ecel>']:
                pos = after.find(tag)
                if 0 <= pos < next_pos:
                    next_pos = pos
            cells.append(after[:next_pos].strip())
            remaining = after[next_pos:]

        has_lcel = '<lcel>' in part or '<ecel>' in part
        non_empty = [c for c in cells if c and not c.startswith('<')]
        is_real_table = len(non_empty) >= 2 and not (has_lcel and len(non_empty) == 1)

        rows.append({
            'cells': cells,
            'is_ucel': is_ucel,
            'is_table_row': is_real_table,
        })

    # Split into table groups
    groups = []
    cur = []
    for r in rows:
        if r['is_table_row']:
            cur.append(r)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = []
    if len(cur) >= 2:
        groups.append(cur)

    # Build output
    out = []
    ri = 0
    while ri < len(rows):
        r = rows[ri]
        if not r['is_table_row']:
            out.append(escape(r['cells'][0] if r['cells'] else ''))
            ri += 1
            continue

        # Collect group
        group = []
        while ri < len(rows) and rows[ri]['is_table_row']:
            group.append(rows[ri])
            ri += 1

        if len(group) < 2:
            for gr in group:
                out.append(escape(gr['cells'][0] if gr['cells'] else ''))
            continue

        # Determine max columns
        max_cols = max(len(r['cells']) for r in group)

        # Compute rowspan for column 0
        rowspan0 = 1
        if len(group) >= 2:
            first = group[1]
            if not first['is_ucel'] and first['cells'] and first['cells'][0]:
                ucel_span = sum(1 for kg in range(2, len(group)) if group[kg]['is_ucel'])
                if ucel_span > 0:
                    rowspan0 = 1 + ucel_span

        html_table = '<table class="ocr-table">\n'
        for gi, gr in enumerate(group):
            cells = list(gr['cells'])
            is_ucel = gr['is_ucel']
            is_header = (gi == 0)

            while len(cells) < max_cols:
                cells.append('')

            html_table += '<tr>'
            for ci in range(max_cols):
                if is_header:
                    html_table += f'<th>{escape(cells[ci])}</th>'
                elif is_ucel:
                    if ci == 0:
                        html_table += '<td></td>'
                    else:
                        html_table += f'<td>{escape(cells[ci - 1])}</td>'
                else:
                    if ci == 0 and rowspan0 > 1:
                        html_table += f'<td rowspan="{rowspan0}">{escape(cells[0])}</td>'
                    else:
                        html_table += f'<td>{escape(cells[ci])}</td>'
            html_table += '</tr>\n'

        html_table += '</table>'
        out.append(html_table)

    return '\n'.join(out)


# ─────────────────────────────────────────────
# 3. 纯文本表格检测（OCR: 模式备用）
# ─────────────────────────────────────────────

def parse_tables_plaintext(text: str) -> str:
    """Detect whitespace-separated tables in plain OCR output."""
    lines = text.split('\n')
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]

        def is_table_line(l):
            toks = l.strip().split()
            if len(toks) < 3:
                return False
            non_first = [t for t in toks[1:]]
            short = sum(1 for t in non_first if len(t) <= 20)
            return len(non_first) >= 2 and short >= len(non_first) * 0.5

        if is_table_line(lines[i]):
            rows_data = [lines[i].strip().split()]
            j = i + 1
            while j < len(lines) and is_table_line(lines[j]):
                rows_data.append(lines[j].strip().split())
                j += 1
            if len(rows_data) >= 2:
                cols = len(rows_data[0])
                if all(len(r) == cols for r in rows_data):
                    html = '<table class="ocr-table">\n'
                    for ri2, row in enumerate(rows_data):
                        tag = 'th' if ri2 == 0 else 'td'
                        cells = ''.join(f'<{tag}>{escape(t)}</{tag}>' for t in row)
                        html += f'<tr>{cells}</tr>\n'
                    html += '</table>'
                    out.append(html)
                    i = j
                    continue
        out.append(escape(lines[i]))
        i += 1

    return '\n'.join(out)


# ─────────────────────────────────────────────
# 4. 主入口
# ─────────────────────────────────────────────

def build_report(ocr_text: str, perf, used_prompt: str, output_dir: str, raw_path: str = None, model_name: str = 'Q4_K_M'):
    p_speed = 'N/A'
    g_speed = 'N/A'
    if perf:
        p_speed = perf.group(1).decode('utf-8') if isinstance(perf.group(1), bytes) else perf.group(1)
        g_speed = perf.group(2).decode('utf-8') if isinstance(perf.group(2), bytes) else perf.group(2)

    # Parse tables
    rendered_text = parse_fcel_structured(ocr_text)
    table_count = rendered_text.count('<table')

    # Plain text version (strip tags)
    plain_text = re.sub(r'<[a-z]+>', ' ', ocr_text)
    plain_text = re.sub(r'\s+', ' ', plain_text)

    if model_name == 'Q8_0':
        llm_desc = 'PaddleOCR-VL-1.6.Q8_0 (475 MB)'
        subtitle_quant = 'Q8_0 475 MB'
    else:
        llm_desc = 'PaddleOCR-VL-1.6.Q4_K_M (284 MB, 由本地 F16 重新量化)'
        subtitle_quant = 'Q4_K_M 284 MB'

    title = f'PaddleOCR-VL-1.6 GGUF ({model_name}) 测试报告'
    subtitle = f'llama.cpp · {subtitle_quant} + mmproj 882MB · GPU offload · 提示词: "{used_prompt}"'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f0f2f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #7c4dff, #4a148c); color: white; padding: 40px 0; text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header p {{ opacity: 0.85; font-size: 14px; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 32px; font-weight: 700; color: #7c4dff; }}
.card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; margin-bottom: 16px; color: #7c4dff; border-left: 4px solid #7c4dff; padding-left: 12px; }}
pre {{ background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; font-size: 13px; line-height: 1.7; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; font-family: 'Cascadia Code', 'Fira Code', 'Source Han Mono', monospace; }}
.params-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
.params-table td {{ padding: 10px 12px; border: 1px solid #e8e8e8; }}
.params-table td:first-child {{ background: #fafafa; font-weight: 600; width: 140px; }}
.ocr-table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; }}
.ocr-table th {{ background: #7c4dff; color: white; padding: 8px 12px; text-align: left; font-weight: 600; white-space: nowrap; }}
.ocr-table td {{ padding: 6px 12px; border-bottom: 1px solid #e0e0e0; white-space: nowrap; }}
.ocr-table tr:hover td {{ background: #f5f0ff; }}
.ocr-table th:first-child, .ocr-table td:first-child {{ white-space: normal; min-width: 100px; }}</style>
</head>
<body>
<div class="header">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
<div class="container">

    <div class="cards">
        <div class="card"><div class="num">{p_speed}</div><div class="label">提示处理速度</div></div>
        <div class="card"><div class="num">{g_speed}</div><div class="label">生成速度</div></div>
        <div class="card"><div class="num">0.5B</div><div class="label">模型参数量</div></div>
        <div class="card"><div class="num">{table_count}</div><div class="label">检测到表格</div></div>
    </div>

    <div class="section">
        <h2>模型配置</h2>
        <table class="params-table">
            <tr><td>LLM 主干</td><td>{llm_desc}</td></tr>
            <tr><td>视觉编码器</td><td>PaddleOCR-VL-1.6-GGUF-mmproj.gguf (841 MB)</td></tr>
            <tr><td>运行工具</td><td>llama-cli (b9571)</td></tr>
            <tr><td>提示词</td><td><code>{escape(used_prompt)}</code></td></tr>
            <tr><td>温度</td><td>0 (确定性输出)</td></tr>
            <tr><td>GPU</td><td>NVIDIA Quadro P1000 (4GB VRAM)</td></tr>
        </table>
    </div>

    <div class="section">
        <h2>解析输出（表格已渲染为 HTML）</h2>
        {rendered_text}
    </div>

    <div class="section">
        <h2>解析输出（纯文本原文）</h2>
        <pre>{escape(plain_text.strip())}</pre>
    </div>

</div>
</body>
</html>'''

    os.makedirs(os.path.join('test', output_dir), exist_ok=True)
    path = os.path.join('test', output_dir, 'report.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report saved: {path}')
    print(f'Tables rendered: {table_count}')
    print(f'Report size: {len(html)} bytes')
    return html, table_count


def detect_model_from_path(raw_path: str) -> str:
    """Guess model name from raw file path. Default: Q4_K_M."""
    fname = os.path.basename(raw_path).lower()
    if 'q8' in fname:
        return 'Q8_0'
    if 'q4_k_m' in fname or 'q4km' in fname:
        return 'Q4_K_M'
    if 'f16' in fname:
        return 'F16'
    return 'Q4_K_M'  # default


def generate_report(raw_path: str, output_dir: str, model_name: str = None):
    """Full pipeline: read raw → extract → parse → build report."""
    if model_name is None:
        model_name = detect_model_from_path(raw_path)
    with open(raw_path, 'rb') as f:
        raw = f.read()
    ocr_text, perf, used_prompt = extract_ocr_and_perf(raw)
    print(f'Prompt: {used_prompt} | OCR text: {len(ocr_text)} chars')
    return build_report(ocr_text, perf, used_prompt, output_dir, raw_path, model_name)


# ─────────────────────────────────────────────
# 5. 运行入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    # Usage: python _gen_vl_report3.py [raw_path] [output_dir] [model_name]
    # model_name is auto-detected from raw_path if not given
    default_raw = r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\llama_cli_raw_v2.txt'
    raw_path = sys.argv[1] if len(sys.argv) > 1 else default_raw
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'output_ocr_vl_gguf'
    model_name = sys.argv[3] if len(sys.argv) > 3 else None
    generate_report(raw_path, output_dir, model_name)
