"""
调试PyMuPDF的lines策略检测所有页面的表格
"""
import fitz

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

for page_num in range(len(doc)):
    page = doc[page_num]
    print(f"\n{'='*80}")
    print(f"第 {page_num + 1} 页")
    print("=" * 80)
    
    # 测试lines策略
    finder = page.find_tables(strategy="lines")
    print(f"lines策略: 检测到 {len(finder.tables)} 个表格")
    
    for i, table in enumerate(finder.tables):
        cell_data = table.extract()
        print(f"\n  表格 {i}: {table.row_count}行 x {table.col_count}列")
        print(f"  bbox: {table.bbox}")
        print(f"  有表头: {table.header is not None}")
        
        # 打印前5行
        print("\n  前5行数据:")
        for ri, row in enumerate(cell_data[:5]):
            # 把None转为空字符串，方便查看
            row_str = [str(c) if c is not None else "None" for c in row]
            print(f"    行{ri}: {row_str}")
        
        # 打印rows结构前3行
        print("\n  rows结构 (前3行):")
        for ri in range(min(3, len(table.rows))):
            row_obj = table.rows[ri]
            cells_info = []
            for ci, cell in enumerate(row_obj.cells):
                if cell is None:
                    cells_info.append(f"c{ci}=None")
                else:
                    cells_info.append(f"c{ci}=bbox")
            print(f"    行{ri}: {cells_info}")

doc.close()
