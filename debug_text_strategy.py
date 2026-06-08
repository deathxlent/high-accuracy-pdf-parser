"""
调试PyMuPDF的text策略检测表格
"""
import fitz

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

# 测试第1、3、4页用text策略
for page_num in [0, 2, 3]:
    page = doc[page_num]
    print(f"\n{'='*80}")
    print(f"第 {page_num + 1} 页 - text策略检测")
    print("=" * 80)
    
    finder = page.find_tables(strategy="text")
    print(f"检测到 {len(finder.tables)} 个表格")
    
    for i, table in enumerate(finder.tables):
        print(f"\n表格 {i}: {table.row_count}行 x {table.col_count}列")
        print(f"bbox: {table.bbox}")
        print(f"table.header: {table.header is not None}")
        if table.header:
            print(f"header cells: {len(table.header.cells)}")
        
        # 打印前几行数据
        cell_data = table.extract()
        print("\n前5行数据:")
        for ri, row in enumerate(cell_data[:5]):
            print(f"  行{ri}: {row}")
        
        # 打印 rows 和 cells 结构
        print("\nrows结构 (前5行):")
        for ri in range(min(5, len(table.rows))):
            row = table.rows[ri]
            cells_info = []
            for ci, cell in enumerate(row.cells):
                if cell is None:
                    cells_info.append(f"col{ci}=None")
                else:
                    cells_info.append(f"col{ci}={cell[:2]}")
            print(f"  行{ri}: {cells_info}")

doc.close()

# 检查第7页是否缺少最后一行
print(f"\n{'='*80}")
print(f"第 7 页 - 检查行数")
print("=" * 80)
page = doc[6] if len(doc) > 6 else None
if page:
    finder = page.find_tables(strategy="lines_strict")
    if finder.tables:
        table = finder.tables[0]
        cell_data = table.extract()
        print(f"PyMuPDF检测到 {table.row_count} 行")
        print("\n所有行数据:")
        for ri, row in enumerate(cell_data):
            print(f"  行{ri}: {row[0] if row[0] else '|'} | {row[1] if len(row) > 1 else ''}")

doc.close()
