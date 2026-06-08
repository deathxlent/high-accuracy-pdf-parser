"""
调试第5页表格检测问题
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
from backend.services.table_service import _find_valid_table, _validate_table
from backend.services.layout_service import detect_layout
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI
import tempfile
import os

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

page = doc[4]  # 第5页
print("第5页")
print("=" * 80)

# 先直接用PyMuPDF检测整页
print("直接检测整页:")
for strategy in ["lines", "lines_strict"]:
    finder = page.find_tables(strategy=strategy)
    print(f"  {strategy}: 检测到 {len(finder.tables)} 个表格")
    for i, t in enumerate(finder.tables):
        print(f"    表格{i}: {t.row_count}行x{t.col_count}列, bbox={t.bbox}")

# 获取layout检测结果
print("\nLayout检测结果:")
with tempfile.TemporaryDirectory() as tmpdir:
    pix = page.get_pixmap(dpi=150)
    jpg_path = os.path.join(tmpdir, "page_5.jpg")
    pix.save(jpg_path)
    
    layout = detect_layout(jpg_path)
    tables = [elem for elem in layout if elem["element_type"] == "Table"]
    for i, t in enumerate(tables):
        print(f"  Table {i}: bbox={t['bbox']}, conf={t['confidence']:.2f}")
        
        # 转换坐标
        pdf_bbox = jpg_bbox_to_pdf_bbox(t['bbox'], 150)
        print(f"    PDF bbox: {pdf_bbox}")
        
        # 检测这个区域
        rect = fitz.Rect(pdf_bbox)
        table, data, strategy = _find_valid_table(page, rect)
        if table:
            print(f"    检测结果: {table.row_count}行x{table.col_count}列, 策略={strategy}")
            print(f"    前2行: {data[:2]}")
            
            # 直接检测这个bbox看看
            finder = page.find_tables(strategy="lines", clip=rect)
            if finder.tables:
                print(f"    直接检测: {finder.tables[0].row_count}x{finder.tables[0].col_count}")
                # 看看col_bboxes和row_bboxes
                t2 = finder.tables[0]
                print(f"    列数: {len(t2.cols)}")
                print(f"    列坐标: {t2.cols}")

doc.close()
