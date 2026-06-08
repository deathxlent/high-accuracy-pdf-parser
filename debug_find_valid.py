"""
调试_find_valid_table问题
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
from backend.services.table_service import _find_valid_table, _validate_table
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

page = doc[4]  # 第5页

# layout检测的bbox
jpg_bbox = (139.85986328125, 1086.6778564453125, 1207.284912109375, 1601.0205078125)
pdf_bbox = jpg_bbox_to_pdf_bbox(jpg_bbox, 150)
rect = fitz.Rect(pdf_bbox)
print(f"检测区域: {rect}")

# 直接检测看看有多少个表格
print("\n直接检测（lines策略）:")
finder = page.find_tables(strategy="lines", clip=rect)
print(f"检测到 {len(finder.tables)} 个表格")
for i, t in enumerate(finder.tables):
    data = t.extract()
    valid = _validate_table(t, data)
    print(f"  表格{i}: {t.row_count}x{t.col_count}, bbox={t.bbox}, valid={valid}")
    if data:
        print(f"    行0: {data[0]}")

# 看看_find_valid_table返回什么
print("\n_find_valid_table返回:")
table, data, strategy = _find_valid_table(page, rect)
if table:
    print(f"  {table.row_count}x{table.col_count}, strategy={strategy}")
    print(f"  bbox={table.bbox}")
    if data:
        print(f"  行0: {data[0]}")

# 直接检测正确的bbox（小一点的区域）
print("\n直接检测正确的bbox（表格1的bbox）:")
correct_rect = fitz.Rect(68.6382874080113, 522.7659790039063, 576.9400329589844, 768.1260070800781)
finder2 = page.find_tables(strategy="lines", clip=correct_rect)
print(f"检测到 {len(finder2.tables)} 个表格")
for i, t in enumerate(finder2.tables):
    print(f"  表格{i}: {t.row_count}x{t.col_count}, bbox={t.bbox}")

doc.close()
