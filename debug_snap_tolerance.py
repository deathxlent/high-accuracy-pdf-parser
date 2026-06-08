"""
调试snap_tolerance和join_tolerance对表格检测的影响
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

page = doc[4]  # 第5页

rect = fitz.Rect(67.132734375, 521.60537109375, 579.4967578124999, 768.48984375)
print(f"检测区域: {rect}")

# 测试不同的snap_tolerance和join_tolerance
params = [
    (None, None),  # 默认值
    (2, 2),
    (3, 3),
    (5, 5),
    (10, 10),
]

for snap, join in params:
    kwargs = {"clip": rect, "strategy": "lines"}
    if snap is not None:
        kwargs["snap_tolerance"] = snap
    if join is not None:
        kwargs["join_tolerance"] = join
    
    finder = page.find_tables(**kwargs)
    if finder.tables:
        t = finder.tables[0]
        data = t.extract()
        print(f"\nsnap={snap}, join={join}: {t.row_count}x{t.col_count}")
        if data:
            print(f"  行0: {data[0]}")
            # 看看rows结构
            if t.rows:
                row0 = t.rows[0]
                none_count = sum(1 for c in row0.cells if c is None)
                print(f"  第一行cells: {len(row0.cells)}个, None数: {none_count}")

doc.close()
