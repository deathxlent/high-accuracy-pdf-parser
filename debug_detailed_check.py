"""
详细对比验证脚本
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
import re
from backend.services.table_service import extract_table_from_native, _find_valid_table
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI, prepare_pages
from backend.services.layout_service import detect_layout_batch
from pathlib import Path

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc_dir = str(Path(pdf_path).parent / Path(pdf_path).stem)

# 准备页面数据
pages_info = prepare_pages(pdf_path, doc_dir)
jpg_paths = [p["jpg_path"] for p in pages_info]
layout_results = detect_layout_batch(jpg_paths)

print("=" * 100)
print("详细对比验证")
print("=" * 100)

# 第3页检查
print("\n" + "=" * 100)
print("第3页检查（应该是3列）")
print("=" * 100)
page = fitz.open(pdf_path)[2]
page_layout = layout_results[2]
tables = [elem for elem in page_layout if elem.get("element_type") == "Table"]

for t_idx, t in enumerate(tables):
    print(f"\n表格{t_idx}:")
    bbox = t["bbox"]
    pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, DEFAULT_DPI)
    rect = fitz.Rect(pdf_bbox)
    
    # 用不同策略检测
    for strategy in ["lines", "lines_strict", "text"]:
        try:
            finder = page.find_tables(clip=rect, strategy=strategy, snap_tolerance=2, join_tolerance=2)
            for tbl in finder.tables:
                data = tbl.extract()
                print(f"  {strategy}: {tbl.row_count}x{tbl.col_count}")
                if data:
                    print(f"    行0: {data[0]}")
                    print(f"    行1: {data[1]}")
        except Exception as e:
            print(f"  {strategy}: 错误 {e}")

# 第7页检查
print("\n" + "=" * 100)
print("第7页检查（应该是14行）")
print("=" * 100)
page = fitz.open(pdf_path)[6]
page_layout = layout_results[6]
tables = [elem for elem in page_layout if elem.get("element_type") == "Table"]

for t_idx, t in enumerate(tables):
    print(f"\n表格{t_idx}:")
    bbox = t["bbox"]
    pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, DEFAULT_DPI)
    rect = fitz.Rect(pdf_bbox)
    
    print(f"  Layout bbox: {bbox}")
    print(f"  PDF rect: {rect}")
    
    # 用不同策略检测
    for strategy in ["lines", "lines_strict", "text"]:
        try:
            finder = page.find_tables(clip=rect, strategy=strategy, snap_tolerance=2, join_tolerance=2)
            for tbl in finder.tables:
                data = tbl.extract()
                print(f"\n  {strategy}: {tbl.row_count}x{tbl.col_count}")
                print(f"    表格bbox: {tbl.bbox}")
                if data:
                    print(f"    第一行: {data[0]}")
                    print(f"    最后一行: {data[-1]}")
                    
                    # 输出所有行
                    print(f"\n    所有{len(data)}行:")
                    for i, row in enumerate(data):
                        clean_row = [str(c).strip()[:15] if c else "" for c in row]
                        print(f"    {i:2d}: {clean_row}")
        except Exception as e:
            print(f"  {strategy}: 错误 {e}")

# 扩大第7页的检测区域看看
print("\n" + "=" * 100)
print("第7页检查 - 扩大检测区域")
print("=" * 100)
# 尝试扩大bbox向下延伸
expanded_bbox = list(bbox)
expanded_bbox[3] += 100  # 向下扩大100像素
print(f"  扩大后的bbox: {tuple(expanded_bbox)}")
expanded_pdf_bbox = jpg_bbox_to_pdf_bbox(tuple(expanded_bbox), DEFAULT_DPI)
expanded_rect = fitz.Rect(expanded_pdf_bbox)
print(f"  扩大后的PDF rect: {expanded_rect}")

for strategy in ["lines", "lines_strict"]:
    try:
        finder = page.find_tables(clip=expanded_rect, strategy=strategy, snap_tolerance=2, join_tolerance=2)
        for tbl in finder.tables:
            data = tbl.extract()
            print(f"\n  {strategy}: {tbl.row_count}x{tbl.col_count}")
            print(f"    表格bbox: {tbl.bbox}")
            if data:
                print(f"    最后一行: {data[-1]}")
    except Exception as e:
        print(f"  {strategy}: 错误 {e}")

# 直接检测整页
print("\n" + "=" * 100)
print("第7页检查 - 直接检测整页")
print("=" * 100)
for strategy in ["lines", "lines_strict"]:
    try:
        finder = page.find_tables(strategy=strategy, snap_tolerance=2, join_tolerance=2)
        for i, tbl in enumerate(finder.tables):
            data = tbl.extract()
            print(f"\n  {strategy} 表格{i}: {tbl.row_count}x{tbl.col_count}")
            print(f"    bbox: {tbl.bbox}")
            if data:
                print(f"    行数: {len(data)}")
                print(f"    最后一行: {data[-1]}")
    except Exception as e:
        print(f"  {strategy}: 错误 {e}")

fitz.open(pdf_path).close()
