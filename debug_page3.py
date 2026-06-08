"""
调试第3页列数问题
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
from backend.services.layout_service import detect_layout_batch
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI, prepare_pages
from backend.services.table_service import _find_valid_table, _validate_table
from pathlib import Path

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc_dir = str(Path(pdf_path).parent / Path(pdf_path).stem)

# 准备页面数据
pages_info = prepare_pages(pdf_path, doc_dir)
jpg_paths = [p["jpg_path"] for p in pages_info]
layout_results = detect_layout_batch(jpg_paths)

page = fitz.open(pdf_path)[2]  # 第3页
page_layout = layout_results[2]
tables = [elem for elem in page_layout if elem.get("element_type") == "Table"]

print("=" * 100)
print("第3页表格检测调试")
print("=" * 100)

for t_idx, t in enumerate(tables):
    print(f"\n表格{t_idx}:")
    bbox = t["bbox"]
    print(f"  Layout bbox: {bbox}")
    
    pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, DEFAULT_DPI)
    rect = fitz.Rect(pdf_bbox)
    print(f"  PDF rect: {rect}")
    
    # 扩大后的rect
    expansion = 3.0
    expanded_rect = fitz.Rect(
        max(0, rect.x0 - expansion),
        max(0, rect.y0 - expansion),
        min(page.rect.x1, rect.x1 + expansion),
        min(page.rect.y1, rect.y1 + expansion),
    )
    print(f"  扩大后rect: {expanded_rect}")
    
    # 测试不同策略和参数
    params_to_test = [
        ("lines", 2, 2),
        ("lines", 1, 1),
        ("lines", 3, 3),
        ("lines_strict", 2, 2),
        ("text", 2, 2),
    ]
    
    for strategy, snap, join in params_to_test:
        print(f"\n  策略={strategy}, snap={snap}, join={join}:")
        try:
            finder = page.find_tables(
                clip=expanded_rect,
                strategy=strategy,
                snap_tolerance=snap,
                join_tolerance=join,
            )
            for tbl_idx, tbl in enumerate(finder.tables):
                data = tbl.extract()
                valid = _validate_table(tbl, data)
                print(f"    表格{tbl_idx}: {tbl.row_count}x{tbl.col_count}, valid={valid}")
                if data:
                    print(f"    行0: {data[0]}")
                    print(f"    行1: {data[1]}")
                    print(f"    行2: {data[2]}")
                    
                    # 看看表格的col_x和row_y坐标
                    try:
                        print(f"    col_x: {tbl.col_x}")
                        print(f"    row_y: {tbl.row_y}")
                    except:
                        pass
        except Exception as e:
            print(f"    错误: {e}")
    
    # 直接检测整页看看
    print(f"\n  直接检测整页（lines策略）:")
    try:
        finder = page.find_tables(strategy="lines", snap_tolerance=2, join_tolerance=2)
        for tbl_idx, tbl in enumerate(finder.tables):
            data = tbl.extract()
            print(f"    表格{tbl_idx}: {tbl.row_count}x{tbl.col_count}, bbox={tbl.bbox}")
            if data:
                print(f"    行0: {data[0]}")
    except Exception as e:
        print(f"    错误: {e}")

# 看看页面的文本内容
print(f"\n\n页面文本内容（前20行）:")
text = page.get_text()
lines = text.split("\n")
for i, line in enumerate(lines[:30]):
    print(f"  {i:2d}: {line}")

fitz.open(pdf_path).close()
