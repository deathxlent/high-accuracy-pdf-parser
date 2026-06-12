"""
Parse llama-cli raw output with structured <fcel> tags (Table Recognition mode)
and generate HTML report with proper cross-row/cross-column table rendering.
"""

import re, os, sys
from html import escape

def extract_ocr_and_perf(raw_bytes: bytes):
    """Extract OCR response and performance metrics from raw llama-cli output."""
    cleaned = re.sub(rb'\x1b\[[0-9;]*[a-zA-Z]', b'', raw_bytes)
    cleaned = cleaned.replace(b'\r\n', b'\n').replace(b'\r', b'\n')

    # Split on prompt boundary - support both "OCR:" and "Table Recognition:"
    prompt_patterns = [b'Table Recognition:', b'OCR:']
    response_bytes = cleaned
    used_prompt = 'OCR:'
    for pp in prompt_patterns:
        parts = cleaned.split(pp)
        if len(parts) >= 2:
            response_bytes = parts[1].strip()
            used_prompt = pp.decode('utf-8')
            break

    # Remove trailing prompt/shell lines
    response_bytes = re.sub(b'\n\\[ Prompt:.*$', b'', response_bytes, flags=re.DOTALL)
    response_bytes = re.sub(b'\n>.*$', b'', response_bytes, flags=re.DOTALL)

    # Extract performance
    perf = re.search(b'\\[ Prompt:\\s*([\\d.]+)\\s*t/s \\| Generation:\\s*([\\d.]+)\\s*t/s',
                     cleaned)

    return response_bytes, perf, used_prompt


def decode_flexibly(data: bytes) -> str:
    """Decode bytes trying multiple encodings (CP936 for Chinese Windows console redirect)."""
    for enc in ['utf-8', 'gb18030', 'gbk', 'cp936', 'gb2312']:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    # Fallback: decode with replacement
    return data.decode('utf-8', errors='replace')


def parse_fcel_table(text: str) -> str:
    """
    Parse <fcel>...</fcel> / <nl> / <ucel> tags into HTML table.
    
    Format from model:
      <fcel>header1<fcel>header2<fcel>header3<nl>
      <fcel>data1<fcel>data2<fcel>data3<nl>
      <ucel><fcel>data_cont<fcel>...
    
    <ucel> indicates a continuation of a merged cell from the row above.
    """
    if '<fcel>' not in text:
        return text  # No structured table tags, return as-is

    # Step 1: split by <nl> to get rows
    rows_raw = re.split(r'<nl>', text)
    
    html_fragments = []
    table_buffer = []      # (cells[], has_ucel) for current table
    
    for row_text in rows_raw:
        row_text = row_text.strip()
        if not row_text:
            continue
        
        # Check for <ucel> (continuation from merged cell above)
        has_ucel = row_text.startswith('<ucel>')
        if has_ucel:
            row_text = row_text[len('<ucel>'):]
        
        # Extract cell content between <fcel> tags
        # Pattern: <fcel>content may have embedded <fcel> tags
        cell_text = row_text
        
        # Remove leading <fcel> if present
        cells = []
        while '<fcel>' in cell_text:
            # Split at <fcel>: everything before is prefix (usually empty), after is rest
            parts = cell_text.split('<fcel>', 1)
            prefix = parts[0]
            if prefix.strip():
                # Text before first <fcel> tag - could be leading content
                pass  # Usually empty
            rest = parts[1]
            
            # Find next <fcel> or end of string
            next_tag = rest.find('<fcel>')
            other_tag = rest.find('<lcel>')
            
            if next_tag >= 0:
                cell_content = rest[:next_tag].strip()
                cells.append(cell_content)
                cell_text = rest[next_tag:]
            elif other_tag >= 0:
                cell_content = rest[:other_tag].strip()
                cells.append(cell_content)
                # <lcel> is a void cell marker - skip remaining
                break
            else:
                # Last cell - everything until end
                cell_content = rest.strip()
                cells.append(cell_content)
                break
        
        # Also handle <lcel> markers (void/empty cells at end of row)
        # These appear as <lcel><lcel><lcel>...<nl>
        # They represent empty cells filling the rest of the row
        non_empty = [c for c in cells if c and c != '<lcel>']
        
        if len(cells) >= 3:
            table_buffer.append((cells, has_ucel))
        elif cells and not table_buffer:
            # Standalone line, not part of a table
            html_fragments.append(escape(row_text))
        else:
            # Could be continuation text after a table
            if row_text:
                html_fragments.append(escape(row_text))
    
    # Now render any accumulated tables
    if table_buffer:
        html_table = '<table class="ocr-table">\n'
        
        # Determine max columns
        max_cols = max(len(cells) for cells, _ in table_buffer)
        
        row_idx = 0
        while row_idx < len(table_buffer):
            cells, has_ucel = table_buffer[row_idx]
            
            # Pad to max_cols
            while len(cells) < max_cols:
                cells.append('')
            
            if row_idx == 0:
                # Header row
                tag = 'th'
                rowspan_map = {}
            else:
                tag = 'td'
            
            html_row = '<tr>'
            col_idx = 0
            while col_idx < len(cells):
                if row_idx > 0:
                    # Check if this column was already counted as rowspan
                    pass  # Simple rendering for now
                
                content = cells[col_idx].strip() if col_idx < len(cells) else ''
                
                # Escape HTML
                content = escape(content)
                
                if has_ucel and col_idx == 0:
                    # First cell is empty (merged from above) - skip (rowspan)
                    # We don't render it
                    col_idx += 1
                    continue
                
                html_row += f'<{tag}>{content}</{tag}>'
                col_idx += 1
            
            html_row += '</tr>\n'
            html_table += html_row
            row_idx += 1
        
        html_table += '</table>'
        html_fragments.append(html_table)
    
    return '\n'.join(html_fragments)


def parse_fcel_structured(text: str) -> str:
    """
    Parse <fcel> structured table format from PaddleOCR-VL "Table Recognition:" mode.
    
    The format uses:
      <fcel>cell1<fcel>cell2...<nl>     — standard row
      <ucel><fcel>cell2<fcel>cell3...<nl> — merged row (col1 continues from above)
      <lcel>                              — empty/padding cell
    """
    if '<fcel>' not in text:
        # No structured tags. Try whitespace-based detection for plain OCR output.
        return parse_tables_plaintext(text)
    
    # Split by <nl> to get individual rows
    parts = text.split('<nl>')
    
    # Classify rows into "real table rows" vs "text rows with lcel padding"
    # A real table row has data in at least 2 cells without <lcel>
    # A text row with <lcel> has one fcel with content followed by <lcel><lcel>...
    rows = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        is_ucel = part.startswith('<ucel>')
        if is_ucel:
            part = part[len('<ucel>'):]
        
        # Extract cells
        cells = []
        remaining = part
        while '<fcel>' in remaining:
            _, after = remaining.split('<fcel>', 1)
            # Find the end of this cell (next <fcel>, <lcel>, or end of string)
            next_pos = len(after)
            for tag in ['<fcel>', '<lcel>']:
                pos = after.find(tag)
                if pos >= 0 and pos < next_pos:
                    next_pos = pos
            content = after[:next_pos].strip()
            cells.append(content)
            remaining = after[next_pos:]
        
        # Check if this row is a "text row" (has <lcel> padding)
        has_lcel = '<lcel>' in part
        non_empty = [c for c in cells if c and not c.startswith('<')]
        
        # A real table row: >= 2 non-empty content cells
        is_real_table = len(non_empty) >= 2 and not (
            # Exclude: single text paragraphs padded with <lcel>
            (has_lcel and len(non_empty) == 1)
        )
        
        rows.append({
            'cells': cells,
            'is_ucel': is_ucel,
            'is_table_row': is_real_table,
            'non_empty_count': len(non_empty),
        })
    
    # Split into table groups (consecutive table rows)
    table_groups = []
    current = []
    for r in rows:
        if r['is_table_row']:
            current.append(r)
        else:
            if len(current) >= 2:
                table_groups.append(current)
            current = []
    if len(current) >= 2:
        table_groups.append(current)
    
    # Build output
    out_parts = []
    row_idx = 0
    while row_idx < len(rows):
        r = rows[row_idx]
        if not r['is_table_row']:
            # Non-table row: render as escaped text
            # Reconstruct original text from cells
            text_content = r['cells'][0] if r['cells'] else ''
            out_parts.append(escape(text_content))
            row_idx += 1
            continue
        
        # Find table group starting at this row
        group = []
        gj = row_idx
        while gj < len(rows) and rows[gj]['is_table_row']:
            group.append(rows[gj])
            gj += 1
        
        if len(group) >= 2:
            # Determine max columns
            max_cols = max(len(r['cells']) for r in group)
            
            html = '<table class="ocr-table">\n'
            
            # Pre-compute rowspan for column 0 (merged cell detection)
            # Row 0 is always header
            colspan0 = 1  # No colspan by default
            rowspan0 = 1
            if len(group) >= 2:
                # Check if first data row has content in col 0 and subsequent rows have ucel
                first_data = group[1]
                if not first_data['is_ucel'] and len(first_data['cells']) > 0 and first_data['cells'][0]:
                    # Count consecutive ucel rows after first data row
                    span = 0
                    for kg in range(2, len(group)):
                        if group[kg]['is_ucel']:
                            span += 1
                        else:
                            break
                    if span > 0:
                        rowspan0 = 1 + span
            
            for gi, r in enumerate(group):
                cells = list(r['cells'])
                is_ucel = r['is_ucel']
                is_header = (gi == 0)
                
                # Pad cells
                while len(cells) < max_cols:
                    cells.append('')
                
                html += '<tr>'
                
                for ci in range(max_cols):
                    if is_header:
                        html += f'<th>{escape(cells[ci])}</th>'
                    elif is_ucel:
                        if ci == 0:
                            # Column 0 is merged (rowspan from row above)
                            html += '<td></td>'
                        else:
                            # ucel row: cells start from col 1, so cells[0] → col 1, cells[1] → col 2, etc.
                            cell_idx = ci - 1
                            content = cells[cell_idx] if cell_idx < len(cells) else ''
                            html += f'<td>{escape(content)}</td>'
                    else:
                        # Normal data row
                        if ci == 0 and gi > 0 and rowspan0 > 1:
                            html += f'<td rowspan="{rowspan0}">{escape(cells[0])}</td>'
                        else:
                            content = cells[ci] if ci < len(cells) else ''
                            html += f'<{ "th" if is_header else "td" }>{escape(content)}</{ "th" if is_header else "td" }>'
                
                html += '</tr>\n'
            
            html += '</table>'
            out_parts.append(html)
            row_idx = gj
        else:
            # Single table row - treat as text
            text_content = group[0]['cells'][0] if group[0]['cells'] else ''
            out_parts.append(escape(text_content))
            row_idx += 1
    
    return '\n'.join(out_parts)


def parse_tables_plaintext(text: str) -> str:
    """Whitespace-based table detection for plain OCR output (no <fcel> tags)."""
    lines = text.split('\n')
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        tokens = line.strip().split()
        
        def is_table_line(l):
            toks = l.strip().split()
            if len(toks) < 3:
                return False
            non_first = [t for t in toks[1:]]
            if len(non_first) >= 2:
                short = sum(1 for t in non_first if len(t) <= 20)
                return short >= len(non_first) * 0.5
            return False
        
        if is_table_line(lines[i]):
            rows_data = [line.strip().split()]
            j = i + 1
            while j < len(lines) and is_table_line(lines[j]):
                rows_data.append(lines[j].strip().split())
                j += 1
            
            if len(rows_data) >= 2:
                # Check if all rows have same column count
                col_count = len(rows_data[0])
                consistent = all(len(r) == col_count for r in rows_data)
                if consistent:
                    html = '<table class="ocr-table">\n'
                    for ri, row in enumerate(rows_data):
                        tag = 'th' if ri == 0 else 'td'
                        cells = ''.join(f'<{tag}>{escape(t)}</{tag}>' for t in row)
                        html += f'<tr>{cells}</tr>\n'
                    html += '</table>'
                    out.append(html)
                    i = j
                    continue
        
        out.append(escape(lines[i]))
        i += 1
    
    return '\n'.join(out)


def parse_fcel_simple(text: str) -> str:
    """Main entry: detect format and parse accordingly."""
    if '<fcel>' in text:
        return parse_fcel_structured(text)
    else:
        return parse_tables_plaintext(text)


def build_report(ocr_text: str, perf, used_prompt: str, output_dir: str):
    """Build HTML report with table rendering."""
    p_speed = 'N/A'
    g_speed = 'N/A'
    if perf:
        p_speed = perf.group(1).decode('utf-8') if isinstance(perf.group(1), bytes) else perf.group(1)
        g_speed = perf.group(2).decode('utf-8') if isinstance(perf.group(2), bytes) else perf.group(2)
    
    # Parse tables
    rendered_text = parse_fcel_simple(ocr_text)
    
    # Count tables
    table_count = rendered_text.count('<table')
    
    # Plain text version
    # Clean up the text for display - remove <fcel>, <nl>, <lcel>, <ucel> tags
    plain_text = ocr_text
    plain_text = re.sub(r'<[a-z]+>', ' ', plain_text)
    plain_text = re.sub(r'\s+', ' ', plain_text)
    # But preserve newlines at sentence boundaries
    plain_text = re.sub(r'(?<=[。，；])', '\n', plain_text)
    ocr_text_escaped = escape(plain_text)
    
    # Title based on prompt
    title = f"PaddleOCR-VL-1.6 GGUF (Q4_K_M) 测试报告"
    subtitle = f'llama.cpp · Q4_K_M 284 MB + mmproj 882MB · GPU offload · 提示词: "{used_prompt}"'
    
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
.ocr-table th:first-child, .ocr-table td:first-child {{ white-space: normal; min-width: 100px; }}
.compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.compare-box {{ padding: 16px; border-radius: 8px; }}
.compare-box.gpu {{ background: #e8f0fe; border: 1px solid #c5d9f7; }}
.compare-box.vl {{ background: #f3e8ff; border: 1px solid #dcc7ff; }}
.compare-box h3 {{ font-size: 14px; margin-bottom: 8px; }}
.compare-box ul {{ padding-left: 20px; font-size: 13px; line-height: 1.8; }}
.used-prompt {{ display: inline-block; background: #e8f0fe; color: #1a73e8; padding: 2px 10px; border-radius: 14px; font-size: 12px; font-weight: 600; }}
</style>
</head>
<body>
<div class="header">
    <h1>{title}</h1>
    <p>{subtitle}</p>
</div>
<div class="container">

    <div class="cards">
        <div class="card">
            <div class="num">{p_speed}</div>
            <div class="label">提示处理速度</div>
        </div>
        <div class="card">
            <div class="num">{g_speed}</div>
            <div class="label">生成速度</div>
        </div>
        <div class="card">
            <div class="num">0.5B</div>
            <div class="label">模型参数量</div>
        </div>
        <div class="card">
            <div class="num">{table_count}</div>
            <div class="label">检测到表格</div>
        </div>
    </div>

    <div class="section">
        <h2>模型配置</h2>
        <table class="params-table">
            <tr><td>LLM 主干</td><td>PaddleOCR-VL-1.6.Q4_K_M (284 MB, 由本地 F16 重新量化)</td></tr>
            <tr><td>视觉编码器</td><td>PaddleOCR-VL-1.6-GGUF-mmproj.gguf (841 MB)</td></tr>
            <tr><td>运行工具</td><td>llama-cli (b9571)</td></tr>
            <tr><td>提示词</td><td><span class="used-prompt">{escape(used_prompt)}</span></td></tr>
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
        <pre>{ocr_text_escaped}</pre>
    </div>

</div>
</body>
</html>'''
    
    os.makedirs(os.path.join('test', output_dir), exist_ok=True)
    path = os.path.join('test', output_dir, 'report_v2.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Report saved: {path}')
    print(f'Tables rendered: {table_count}')
    print(f'Report size: {len(html)} bytes')
    return html, table_count


# === Main ===
raw_path = r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\llama_cli_raw_v2.txt'
with open(raw_path, 'rb') as f:
    raw = f.read()

response_bytes, perf, used_prompt = extract_ocr_and_perf(raw)
ocr_text = decode_flexibly(response_bytes)

print(f'Used prompt: {used_prompt}')
print(f'Response size: {len(ocr_text)} chars')
print(f'Contains <fcel>: {"<fcel>" in ocr_text}')

build_report(ocr_text, perf, used_prompt, 'output_ocr_vl_gguf')
