"""
完整验证脚本 - 包含layout检测和表格解析
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
import asyncio
from backend.services.layout_service import detect_layout_batch
from backend.services.table_service import extract_table_from_native, _find_valid_table
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI, prepare_pages
from backend.services.parse_service import _is_continuation_table
from pathlib import Path

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc_dir = str(Path(pdf_path).parent / Path(pdf_path).stem)

print("=" * 100)
print("完整验证表格解析修复")
print("=" * 100)

# 准备页面数据
print("\n准备页面数据...")
pages_info = prepare_pages(pdf_path, doc_dir)
print(f"共 {len(pages_info)} 页")

# 批量检测layout
print("\n批量检测layout...")
jpg_paths = [p["jpg_path"] for p in pages_info]
layout_results = detect_layout_batch(jpg_paths)
print(f"layout检测完成")

# 逐页处理
prev_table_info = None

for page_idx, (page_info, layout) in enumerate(zip(pages_info, layout_results)):
    page_num = page_info["page_number"]
    print(f"\n{'=' * 100}")
    print(f"第{page_num}页")
    print("=" * 100)
    
    # 打开PDF页面
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    
    # 过滤出表格元素
    tables = [elem for elem in layout if elem.get("element_type") == "Table"]
    
    if not tables:
        print(f"  未检测到表格")
        doc.close()
        # 没有表格时，清空上一页表格信息
        prev_table_info = None
        continue
    
    print(f"  检测到 {len(tables)} 个表格")
    
    # 记录当前页列数最多的表格
    current_max_col_table = None
    
    for t_idx, t in enumerate(tables):
        t_bbox = t["bbox"]
        conf = t.get("confidence", 0)
        print(f"\n  表格{t_idx}: bbox={t_bbox}, conf={conf:.2f}")
        print(f"  {'-' * 60}")
        
        # 检测是否是跨页接续
        force_no_header = False
        if prev_table_info is not None:
            pdf_bbox = jpg_bbox_to_pdf_bbox(t_bbox, DEFAULT_DPI)
            rect = fitz.Rect(pdf_bbox)
            preview_table, preview_data, strategy = _find_valid_table(page, rect)
            
            if preview_table and preview_data:
                is_cont = _is_continuation_table(
                    preview_table.col_count,
                    preview_data[0],
                    prev_table_info["col_count"],
                    prev_table_info["last_row"]
                )
                if is_cont:
                    print(f"  ✅ 检测到跨页接续表格（上一页：{prev_table_info['col_count']}列，当前页：{preview_table.col_count}列，策略={strategy}）")
                    force_no_header = True
                else:
                    print(f"  ❌ 不是跨页接续（上一页：{prev_table_info['col_count']}列，当前页：{preview_table.col_count}列，策略={strategy}）")
                    print(f"     上一页最后一行: {prev_table_info['last_row'][:3]}")
                    print(f"     当前页第一行: {preview_data[0][:3]}")
        
        # 提取表格
        result = extract_table_from_native(page, t_bbox, True, DEFAULT_DPI, force_no_header)
        html = result.get("html", "")
        
        # 显示一些统计信息
        if html:
            # 提取表格信息
            import re
            row_count = html.count("<tr>")
            header_count = html.count("<th>")
            data_count = html.count("<td>")
            colspan_count = html.count('colspan=')
            rowspan_count = html.count('rowspan=')
            
            print(f"  行数: {row_count}, 表头单元格: {header_count}, 数据单元格: {data_count}")
            print(f"  colspan: {colspan_count}, rowspan: {rowspan_count}")
            
            # 显示前3行和后2行的预览
            rows = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)
            if rows:
                print(f"\n  前3行预览:")
                for i, row in enumerate(rows[:3]):
                    cells = re.findall(r'<t[hd].*?>(.*?)</t[hd]>', row, re.DOTALL)
                    clean_cells = [re.sub(r'<br\s*/?>', '\\n', c).strip()[:20] for c in cells]
                    print(f"    行{i}: {clean_cells}")
                if len(rows) > 3:
                    print(f"  ...")
                    print(f"  最后1行预览:")
                    cells = re.findall(r'<t[hd].*?>(.*?)</t[hd]>', rows[-1], re.DOTALL)
                    clean_cells = [re.sub(r'<br\s*/?>', '\\n', c).strip()[:20] for c in cells]
                    print(f"    行{len(rows)-1}: {clean_cells}")
        
        # 记录表格信息用于下一页跨页检测（只记录列数最多的）
        pdf_bbox = jpg_bbox_to_pdf_bbox(t_bbox, DEFAULT_DPI)
        rect = fitz.Rect(pdf_bbox)
        info_table, info_data, _ = _find_valid_table(page, rect)
        if info_table and info_data:
            if current_max_col_table is None or info_table.col_count > current_max_col_table["col_count"]:
                current_max_col_table = {
                    "col_count": info_table.col_count,
                    "last_row": info_data[-1],
                    "page_number": page_num,
                }
                print(f"\n  📝 记录为当前页最大列表格（{info_table.col_count}列）")
    
    # 更新上一页表格信息
    if current_max_col_table:
        prev_table_info = current_max_col_table
        print(f"\n  🔄 上一页表格信息更新为：{prev_table_info['col_count']}列（来自第{prev_table_info['page_number']}页）")
    else:
        prev_table_info = None
    
    doc.close()

print(f"\n{'=' * 100}")
print("验证完成")
print("=" * 100)
