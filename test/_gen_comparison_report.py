import sys
import sqlite3
import difflib
from pathlib import Path
from html import escape
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

DOC_ID_9 = 9    # Native PDF + Original PaddleOCR
DOC_ID_11 = 11  # Scanned PDF + PaddleOCR-VL Q4_K_M GGUF

def get_doc_data(conn, doc_id):
    cur = conn.cursor()
    doc = cur.execute("SELECT * FROM pdf_documents WHERE id = ?", (doc_id,)).fetchone()
    pages = cur.execute("""
        SELECT id, page_number, is_scanned, jpg_path, status, width, height, jpg_width, jpg_height
        FROM pdf_pages WHERE document_id = ? ORDER BY page_number
    """, (doc_id,)).fetchall()
    
    result = {
        'doc_id': doc_id,
        'filename': doc[2],
        'status': doc[7],
        'pages': []
    }
    
    for p in pages:
        pid, pnum, is_scanned, jpg_path, status, w, h, jw, jh = p
        elements = cur.execute("""
            SELECT id, element_type, bbox_x0, bbox_y0, bbox_x1, bbox_y1,
                   confidence, reading_order, content, content_format, cross_page_group
            FROM page_elements WHERE page_id = ? ORDER BY reading_order
        """, (pid,)).fetchall()
        
        elem_list = []
        for e in elements:
            elem_list.append({
                'id': e[0],
                'type': e[1],
                'bbox': (e[2], e[3], e[4], e[5]),
                'confidence': e[6],
                'order': e[7],
                'content': e[8] or '',
                'format': e[9],
                'cross_page_group': e[10]
            })
        
        result['pages'].append({
            'id': pid,
            'number': pnum,
            'is_scanned': is_scanned,
            'jpg_path': jpg_path,
            'status': status,
            'width': w, 'height': h,
            'jpg_width': jw, 'jpg_height': jh,
            'elements': elem_list
        })
    
    return result


def extract_text_content(elem):
    if elem['format'] == 'html' and elem['type'] == 'Table':
        import re
        text = re.sub(r'<[^>]+>', ' ', elem['content'])
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return elem['content'].strip()


def compare_elements(page9, page11):
    elems9 = page9['elements']
    elems11 = page11['elements']
    
    types9 = {}
    for e in elems9:
        types9[e['type']] = types9.get(e['type'], 0) + 1
    
    types11 = {}
    for e in elems11:
        types11[e['type']] = types11.get(e['type'], 0) + 1
    
    all_types = sorted(set(list(types9.keys()) + list(types11.keys())))
    
    type_comparison = []
    for t in all_types:
        c9 = types9.get(t, 0)
        c11 = types11.get(t, 0)
        status = "✓ 匹配" if c9 == c11 else ("⚠ 偏差" if abs(c9 - c11) <= 2 else "✗ 显著差异")
        type_comparison.append({
            'type': t,
            'count9': c9,
            'count11': c11,
            'status': status,
            'diff': c11 - c9
        })
    
    text9_parts = []
    for e in elems9:
        t = extract_text_content(e)
        if t:
            text9_parts.append(f"[{e['type']}] {t}")
    
    text11_parts = []
    for e in elems11:
        t = extract_text_content(e)
        if t:
            text11_parts.append(f"[{e['type']}] {t}")
    
    text9 = "\n".join(text9_parts)
    text11 = "\n".join(text11_parts)
    
    similarity = 0.0
    if text9 and text11:
        sm = difflib.SequenceMatcher(None, text9[:2000], text11[:2000])
        similarity = sm.ratio()
    
    return type_comparison, text9, text11, similarity


def build_comparison_report(data9, data11, output_path):
    pages9 = data9['pages']
    pages11 = data11['pages']
    max_pages = max(len(pages9), len(pages11))
    
    overall_similarities = []
    page_reports = []
    
    for i in range(max_pages):
        p9 = pages9[i] if i < len(pages9) else None
        p11 = pages11[i] if i < len(pages11) else None
        
        if p9 and p11:
            type_comp, text9, text11, sim = compare_elements(p9, p11)
            overall_similarities.append(sim)
            
            diff_html = difflib.HtmlDiff(wrapcolumn=60).make_table(
                text9.splitlines(), text11.splitlines(),
                fromdesc='ID=9 (原生PDF+PaddleOCR)',
                todesc='ID=11 (扫描版+VL Q4_K_M)',
                context=True, numlines=2
            )
            
            page_reports.append({
                'page_num': i + 1,
                'p9': p9, 'p11': p11,
                'type_comp': type_comp,
                'similarity': sim,
                'text9': text9, 'text11': text11,
                'diff_html': diff_html
            })
        else:
            page_reports.append({
                'page_num': i + 1,
                'p9': p9, 'p11': p11,
                'error': '页面数量不匹配'
            })
    
    avg_similarity = sum(overall_similarities) / len(overall_similarities) if overall_similarities else 0
    
    total_elems9 = sum(len(p['elements']) for p in pages9)
    total_elems11 = sum(len(p['elements']) for p in pages11)
    
    def sim_color(sim):
        if sim >= 0.9: return '#10b981'
        if sim >= 0.75: return '#3b82f6'
        if sim >= 0.6: return '#f59e0b'
        return '#ef4444'
    
    def sim_label(sim):
        if sim >= 0.9: return '极高'
        if sim >= 0.75: return '高'
        if sim >= 0.6: return '中'
        return '低'
    
    pages_html_parts = []
    for pr in page_reports:
        pnum = pr['page_num']
        if 'error' in pr:
            pages_html_parts.append(f'''
            <div class="page-report">
                <h2>第 {pnum} 页 - {pr['error']}</h2>
            </div>
            ''')
            continue
        
        p9 = pr['p9']
        p11 = pr['p11']
        sim = pr['similarity']
        tc = pr['type_comp']
        
        type_rows = ""
        for t in tc:
            if t['diff'] == 0:
                diff_cell = f"<td class='match'>{t['diff']:+d}</td>"
            elif abs(t['diff']) <= 2:
                diff_cell = f"<td class='warn'>{t['diff']:+d}</td>"
            else:
                diff_cell = f"<td class='miss'>{t['diff']:+d}</td>"
            
            type_rows += f'''
            <tr>
                <td><strong>{t['type']}</strong></td>
                <td class="count9">{t['count9']}</td>
                <td class="count11">{t['count11']}</td>
                {diff_cell}
                <td>{t['status']}</td>
            </tr>
            '''
        
        elements_details9 = ""
        for e in p9['elements']:
            content_preview = (e['content'][:100] + '...') if len(e['content']) > 100 else e['content']
            content_preview = escape(content_preview.replace('\n', '\\n'))
            elements_details9 += f'''
            <div class="elem-card">
                <div class="elem-header">
                    <span class="elem-type type-{e['type'].replace(' ', '-').lower()}">{e['type']}</span>
                    <span class="elem-order">顺序 #{e['order']}</span>
                    <span class="elem-conf">置信度: {e['confidence']:.2f}</span>
                </div>
                <div class="elem-bbox">BBox: ({e['bbox'][0]:.0f}, {e['bbox'][1]:.0f}) - ({e['bbox'][2]:.0f}, {e['bbox'][3]:.0f})</div>
                <div class="elem-format">格式: {e['format']}</div>
                <div class="elem-content">{content_preview}</div>
            </div>
            '''
        
        elements_details11 = ""
        for e in p11['elements']:
            content_preview = (e['content'][:100] + '...') if len(e['content']) > 100 else e['content']
            content_preview = escape(content_preview.replace('\n', '\\n'))
            elements_details11 += f'''
            <div class="elem-card">
                <div class="elem-header">
                    <span class="elem-type type-{e['type'].replace(' ', '-').lower()}">{e['type']}</span>
                    <span class="elem-order">顺序 #{e['order']}</span>
                    <span class="elem-conf">置信度: {e['confidence']:.2f}</span>
                </div>
                <div class="elem-bbox">BBox: ({e['bbox'][0]:.0f}, {e['bbox'][1]:.0f}) - ({e['bbox'][2]:.0f}, {e['bbox'][3]:.0f})</div>
                <div class="elem-format">格式: {e['format']}</div>
                <div class="elem-content">{content_preview}</div>
            </div>
            '''
        
        pages_html_parts.append(f'''
        <div class="page-report" id="page-{pnum}">
            <div class="page-header">
                <h2>第 {pnum} 页</h2>
                <div class="similarity-badge" style="background: {sim_color(sim)}">
                    相似度: {sim*100:.1f}% ({sim_label(sim)})
                </div>
            </div>
            
            <div class="page-meta">
                <div class="meta-col">
                    <h3>ID=9 (原生PDF+PaddleOCR)</h3>
                    <p>页面ID: {p9['id']} | 状态: {p9['status']} | 尺寸: {p9['width']:.0f}x{p9['height']:.0f}pt</p>
                    <p>扫描版: {'是' if p9['is_scanned'] else '否'} | 图片尺寸: {p9['jpg_width']:.0f}x{p9['jpg_height']:.0f}px</p>
                    <p>元素总数: <strong>{len(p9['elements'])}</strong></p>
                </div>
                <div class="meta-col">
                    <h3>ID=11 (扫描版+VL Q4_K_M)</h3>
                    <p>页面ID: {p11['id']} | 状态: {p11['status']} | 尺寸: {p11['width']:.0f}x{p11['height']:.0f}pt</p>
                    <p>扫描版: {'是' if p11['is_scanned'] else '否'} | 图片尺寸: {p11['jpg_width']:.0f}x{p11['jpg_height']:.0f}px</p>
                    <p>元素总数: <strong>{len(p11['elements'])}</strong></p>
                </div>
            </div>
            
            <h3>元素类型统计对比</h3>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>元素类型</th>
                        <th>ID=9 数量</th>
                        <th>ID=11 数量</th>
                        <th>差值</th>
                        <th>状态</th>
                    </tr>
                </thead>
                <tbody>
                    {type_rows}
                </tbody>
            </table>
            
            <h3>文本内容差异对比</h3>
            <div class="diff-container">
                {pr['diff_html']}
            </div>
            
            <h3>ID=9 元素详情 ({len(p9['elements'])})</h3>
            <div class="elements-grid">
                {elements_details9}
            </div>
            
            <h3>ID=11 元素详情 ({len(p11['elements'])})</h3>
            <div class="elements-grid">
                {elements_details11}
            </div>
        </div>
        ''')
    
    nav_links = " ".join(
        f'<a href="#page-{i+1}" class="nav-link">第{i+1}页 ({page_reports[i]["similarity"]*100:.0f}%)</a>'
        if 'similarity' in page_reports[i]
        else f'<a href="#page-{i+1}" class="nav-link">第{i+1}页</a>'
        for i in range(len(page_reports))
    )
    
    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF解析对比报告 - ID=9 (原生PaddleOCR) vs ID=11 (VL Q4_K_M GGUF)</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    background: #f0f2f5; color: #333; line-height: 1.6;
}}
.header {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white; padding: 48px 24px; text-align: center;
}}
.header h1 {{ font-size: 28px; margin-bottom: 12px; }}
.header p {{ opacity: 0.9; font-size: 15px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px 16px; }}
.nav {{
    background: white; border-radius: 12px; padding: 16px 24px;
    margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
}}
.nav-label {{ font-weight: 600; color: #666; margin-right: 12px; }}
.nav-link {{
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    background: #f3e8ff; color: #7c3aed; text-decoration: none;
    font-size: 13px; font-weight: 500; transition: all 0.2s;
}}
.nav-link:hover {{ background: #7c3aed; color: white; }}
.overview-cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin-bottom: 32px;
}}
.card {{
    background: white; border-radius: 12px; padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center;
}}
.card .num {{ font-size: 36px; font-weight: 700; margin-bottom: 8px; }}
.card .label {{ font-size: 14px; color: #666; }}
.card.green .num {{ color: #10b981; }}
.card.blue .num {{ color: #3b82f6; }}
.card.purple .num {{ color: #8b5cf6; }}
.card.orange .num {{ color: #f59e0b; }}
.page-report {{
    background: white; border-radius: 12px; padding: 32px;
    margin-bottom: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.page-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 24px; padding-bottom: 16px;
    border-bottom: 2px solid #f0f0f0;
}}
.page-header h2 {{ font-size: 22px; color: #1f2937; }}
.similarity-badge {{
    padding: 8px 20px; border-radius: 24px; color: white;
    font-weight: 600; font-size: 14px;
}}
.page-meta {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    margin-bottom: 24px;
}}
.meta-col {{
    background: #f9fafb; border-radius: 8px; padding: 16px;
}}
.meta-col h3 {{ font-size: 15px; margin-bottom: 8px; color: #374151; }}
.meta-col p {{ font-size: 13px; color: #6b7280; margin: 2px 0; }}
.page-report h3 {{
    font-size: 16px; color: #4f46e5; margin: 24px 0 12px 0;
    padding-left: 12px; border-left: 4px solid #4f46e5;
}}
.comparison-table {{
    width: 100%; border-collapse: collapse; margin-bottom: 16px; font-size: 14px;
}}
.comparison-table th, .comparison-table td {{
    padding: 10px 14px; border: 1px solid #e5e7eb; text-align: center;
}}
.comparison-table th {{
    background: #f3f4f6; font-weight: 600; color: #374151;
}}
.comparison-table td.match {{ background: #d1fae5; color: #065f46; font-weight: 600; }}
.comparison-table td.warn {{ background: #fef3c7; color: #92400e; font-weight: 600; }}
.comparison-table td.miss {{ background: #fee2e2; color: #991b1b; font-weight: 600; }}
.comparison-table td.count9 {{ color: #2563eb; font-weight: 500; }}
.comparison-table td.count11 {{ color: #7c3aed; font-weight: 500; }}
.diff-container {{
    overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px;
    font-family: 'Cascadia Code', 'Fira Code', Consolas, monospace; font-size: 12px;
}}
.diff-container table {{ width: 100%; border-collapse: collapse; }}
.diff-container td {{ padding: 4px 8px; vertical-align: top; white-space: pre-wrap; word-break: break-all; }}
.diff_add {{ background: #dcfce7; }}
.diff_sub {{ background: #fee2e2; }}
.diff_chg {{ background: #fef9c3; }}
.elements-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 12px;
}}
.elem-card {{
    background: #f9fafb; border-radius: 8px; padding: 12px;
    border: 1px solid #e5e7eb; transition: all 0.2s;
}}
.elem-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); transform: translateY(-2px); }}
.elem-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 8px; flex-wrap: wrap; gap: 4px;
}}
.elem-type {{
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; color: white;
}}
.type-title {{ background: #ef4444; }}
.type-section-header {{ background: #f97316; }}
.type-text {{ background: #3b82f6; }}
.type-list-item {{ background: #8b5cf6; }}
.type-table {{ background: #10b981; }}
.type-formula {{ background: #ec4899; }}
.type-picture {{ background: #6366f1; }}
.type-caption {{ background: #14b8a6; }}
.type-page-header {{ background: #64748b; }}
.type-page-footer {{ background: #64748b; }}
.type-footnote {{ background: #a855f7; }}
.elem-order {{ font-size: 11px; color: #6b7280; }}
.elem-conf {{ font-size: 11px; color: #6b7280; }}
.elem-bbox {{ font-size: 11px; color: #9ca3af; margin: 4px 0; }}
.elem-format {{ font-size: 11px; color: #9ca3af; margin: 2px 0; }}
.elem-content {{
    font-size: 12px; color: #374151; margin-top: 8px;
    padding-top: 8px; border-top: 1px dashed #e5e7eb;
    overflow: hidden; text-overflow: ellipsis;
    word-break: break-word;
}}
.footer {{
    text-align: center; padding: 32px; color: #9ca3af;
    font-size: 13px;
}}
.method-section {{
    background: white; border-radius: 12px; padding: 24px;
    margin-bottom: 32px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}
.method-section h2 {{ color: #4f46e5; margin-bottom: 16px; font-size: 20px; }}
.method-section ul {{ padding-left: 24px; line-height: 2; }}
.method-section li {{ margin-bottom: 4px; }}
code {{
    background: #f3f4f6; padding: 2px 6px; border-radius: 4px;
    font-family: 'Cascadia Code', Consolas, monospace; font-size: 12px;
}}
</style>
</head>
<body>
<div class="header">
    <h1>📊 PDF 解析对比报告</h1>
    <p>ID=9 (原生PDF + PaddleOCR v4) vs ID=11 (扫描版 + PaddleOCR-VL 1.6 Q4_K_M GGUF via llama.cpp)</p>
    <p style="margin-top: 8px; font-size: 13px;">
        生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
</div>

<div class="container">
    <div class="nav">
        <span class="nav-label">快速导航：</span>
        {nav_links}
    </div>

    <div class="method-section">
        <h2>🔬 测试方案说明</h2>
        <ul>
            <li><strong>基线方案 (ID=9)</strong>: 原生PDF + 传统PaddleOCR v4（CPU模式），处理非扫描版PDF</li>
            <li><strong>新方案 (ID=11)</strong>: 扫描版PDF + PaddleOCR-VL 1.6 Q4_K_M量化 (284MB) + mmproj (841MB)，通过 <code>llama-server</code> HTTP API调用</li>
            <li><strong>模型部署</strong>: llama.cpp llama-server，监听 127.0.0.1:8080，GPU offload (-ngl 99)，8线程，ctx=4096</li>
            <li><strong>OCR提示词</strong>: 文本识别用 <code>OCR:</code>，表格识别用 <code>Table Recognition:</code></li>
            <li><strong>表格输出</strong>: PaddleOCR-VL输出结构化标签 <code>&lt;fcel&gt;</code>/<code>&lt;nl&gt;</code>/<code>&lt;ucel&gt;</code>，解析为带跨行跨列的HTML表格</li>
            <li><strong>量化类型</strong>: Q4_K_M (4-bit k-quant, Medium方案) - 模型体积从~800MB降至284MB</li>
        </ul>
    </div>

    <div class="overview-cards">
        <div class="card green">
            <div class="num">{avg_similarity*100:.1f}%</div>
            <div class="label">平均文本相似度</div>
        </div>
        <div class="card blue">
            <div class="num">{len(pages9)}</div>
            <div class="label">总页数 (ID=9)</div>
        </div>
        <div class="card purple">
            <div class="num">{len(pages11)}</div>
            <div class="label">总页数 (ID=11)</div>
        </div>
        <div class="card blue">
            <div class="num">{total_elems9}</div>
            <div class="label">元素总数 (ID=9)</div>
        </div>
        <div class="card purple">
            <div class="num">{total_elems11}</div>
            <div class="label">元素总数 (ID=11)</div>
        </div>
        <div class="card orange">
            <div class="num">{total_elems11-total_elems9:+d}</div>
            <div class="label">元素数差值</div>
        </div>
    </div>

    {"".join(pages_html_parts)}

    <div class="footer">
        <p>报告生成工具: high accuracy pdf parser · Q4量化方案验证</p>
        <p>模型: PaddleOCR-VL-1.6.Q4_K_M.gguf + PaddleOCR-VL-1.6-GGUF-mmproj.gguf</p>
        <p>推理引擎: llama.cpp llama-server (b9571) · GPU: NVIDIA Quadro P1000 4GB</p>
    </div>
</div>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report saved: {output_path}")
    print(f"Report size: {len(html)} bytes")
    print(f"\n=== Summary ===")
    print(f"Average text similarity: {avg_similarity*100:.1f}%")
    print(f"Total elements ID=9: {total_elems9}")
    print(f"Total elements ID=11: {total_elems11}")
    for i, pr in enumerate(page_reports):
        if 'similarity' in pr:
            print(f"  Page {i+1}: {pr['similarity']*100:.1f}%")

    return avg_similarity


def main():
    conn = sqlite3.connect(str(DB_PATH))
    
    data9 = get_doc_data(conn, DOC_ID_9)
    data11 = get_doc_data(conn, DOC_ID_11)
    
    print(f"Document 9: {data9['filename']} ({len(data9['pages'])} pages)")
    print(f"Document 11: {data11['filename']} ({len(data11['pages'])} pages)")
    
    output_dir = Path(r"c:\ws\high accuracy pdf parser\test\output_comparison")
    output_dir.mkdir(exist_ok=True)
    output_path = str(output_dir / "comparison_report_id9_vs_id11.html")
    
    build_comparison_report(data9, data11, output_path)
    
    conn.close()


if __name__ == "__main__":
    main()
