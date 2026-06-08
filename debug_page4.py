"""
调试第4页的表格检测
"""
import fitz

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

page = doc[3]  # 第4页
print("第4页")
print("=" * 80)

# 先看看整页文本
text = page.get_text("text")
lines = text.split("\n")
print("整页文本前60行:")
for i, line in enumerate(lines[:60]):
    print(f"{i:3d}: {repr(line)}")

# 测试所有策略
print("\n" + "=" * 80)
print("所有检测策略:")
print("=" * 80)

strategies = ["lines_strict", "lines", "text"]
for strategy in strategies:
    finder = page.find_tables(strategy=strategy)
    print(f"\n策略 '{strategy}': 检测到 {len(finder.tables)} 个表格")
    for i, table in enumerate(finder.tables):
        cell_data = table.extract()
        print(f"  表格 {i}: {table.row_count}行 x {table.col_count}列, bbox={table.bbox}")
        # 打印前3行
        for ri, row in enumerate(cell_data[:3]):
            row_str = [str(c) if c is not None else "None" for c in row]
            print(f"    行{ri}: {row_str}")

# 看看第4页的两个正确表格大概在什么位置
# 正确答案第4页有两个8列的表格
# 我们手动指定区域试试
print("\n" + "=" * 80)
print("手动指定区域检测:")
print("=" * 80)

# 尝试上半部分（第一个表格）
rect1 = fitz.Rect(50, 50, 550, 400)
print(f"\n区域 {rect1}:")
for strategy in strategies:
    finder = page.find_tables(strategy=strategy, clip=rect1)
    print(f"  策略 '{strategy}': {len(finder.tables)} 个表格")
    if finder.tables:
        t = finder.tables[0]
        print(f"    {t.row_count}行 x {t.col_count}列")
        cd = t.extract()
        if cd:
            print(f"    行0: {cd[0]}")

# 尝试下半部分（第二个表格）
rect2 = fitz.Rect(50, 400, 550, 750)
print(f"\n区域 {rect2}:")
for strategy in strategies:
    finder = page.find_tables(strategy=strategy, clip=rect2)
    print(f"  策略 '{strategy}': {len(finder.tables)} 个表格")
    if finder.tables:
        t = finder.tables[0]
        print(f"    {t.row_count}行 x {t.col_count}列")
        cd = t.extract()
        if cd:
            print(f"    行0: {cd[0]}")

doc.close()
