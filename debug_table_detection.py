"""
调试表格检测流程
1. 检查layout检测是否能找到第1、3、4页的表格
2. 检查PyMuPDF本身能检测到哪些表格
"""
import fitz
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

from backend.services.layout_service import detect_layout

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

# 先看每一页用PyMuPDF能检测到多少表格
print("=" * 80)
print("PyMuPDF 原生表格检测结果")
print("=" * 80)
for page_num in range(len(doc)):
    page = doc[page_num]
    finder = page.find_tables(strategy="lines_strict")
    print(f"\n第 {page_num + 1} 页: 检测到 {len(finder.tables)} 个表格")
    for i, table in enumerate(finder.tables):
        print(f"  表格 {i}: {table.row_count}行 x {table.col_count}列, bbox={table.bbox}")
        if table.header:
            print(f"    有表头: {len(table.header.cells)} 个单元格")
    # 也试试其他策略
    finder2 = page.find_tables(strategy="text")
    if len(finder2.tables) > len(finder.tables):
        print(f"  text策略检测到 {len(finder2.tables)} 个表格")

doc.close()

# 检查layout检测结果
print("\n" + "=" * 80)
print("YOLO Layout 检测结果 (Table类型)")
print("=" * 80)

import tempfile
import os
from pathlib import Path

doc = fitz.open(pdf_path)
for page_num in range(len(doc)):
    page = doc[page_num]
    
    # 渲染成图片供layout检测
    with tempfile.TemporaryDirectory() as tmpdir:
        pix = page.get_pixmap(dpi=150)
        jpg_path = os.path.join(tmpdir, f"page_{page_num+1}.jpg")
        pix.save(jpg_path)
        
        layout = detect_layout(jpg_path)
        tables = [elem for elem in layout if elem["element_type"] == "Table"]
        print(f"\n第 {page_num + 1} 页: Layout检测到 {len(tables)} 个Table")
        for i, t in enumerate(tables):
            print(f"  表格 {i}: bbox={t['bbox']}, conf={t['confidence']:.2f}")

doc.close()
