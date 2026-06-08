"""
调试PyMuPDF所有表格检测策略
"""
import fitz

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

# 测试第1页的所有策略
page_num = 0
page = doc[page_num]
print(f"第 {page_num + 1} 页")
print("=" * 80)

strategies = ["lines_strict", "lines", "text", "grid"]
for strategy in strategies:
    try:
        finder = page.find_tables(strategy=strategy)
        print(f"\n策略 '{strategy}': 检测到 {len(finder.tables)} 个表格")
        for i, table in enumerate(finder.tables):
            print(f"  表格 {i}: {table.row_count}行 x {table.col_count}列")
            cell_data = table.extract()
            if cell_data:
                print(f"    前3行: {cell_data[:3]}")
    except Exception as e:
        print(f"\n策略 '{strategy}': 错误 - {e}")

# 直接查看第1页的文本内容
print("\n" + "=" * 80)
print("第1页直接提取的文本（表格区域附近）:")
print("=" * 80)
# 直接看整页文本
text = page.get_text("text")
lines = text.split("\n")
for i, line in enumerate(lines):
    if i < 50:  # 只看前50行
        print(f"{i:3d}: {repr(line)}")

# 检查第7页行数
print("\n" + "=" * 80)
print("第7页检查:")
print("=" * 80)
page7 = doc[6]
finder = page7.find_tables(strategy="lines_strict")
if finder.tables:
    table = finder.tables[0]
    cell_data = table.extract()
    print(f"PyMuPDF检测到 {table.row_count} 行 x {table.col_count} 列")
    print("\n所有行 (第一列 | 第二列):")
    for ri, row in enumerate(cell_data):
        col0 = row[0].strip() if row[0] else ""
        col1 = row[1].strip() if len(row) > 1 and row[1] else ""
        print(f"  行{ri:2d}: {col0:20s} | {col1}")

# 检查第7页最后几行的文本
print("\n第7页底部的文本:")
bottom_text = page7.get_text("text", clip=fitz.Rect(0, 700, 600, 850))
print(repr(bottom_text))

doc.close()
