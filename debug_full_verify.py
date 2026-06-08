"""
完整验证修复效果
"""
import sys
sys.path.insert(0, r'c:\ws\high accuracy pdf parser')

import fitz
from backend.services.table_service import extract_table_from_native, _find_valid_table, _validate_table
from backend.services.layout_service import detect_layout
import tempfile
import os

pdf_path = r"c:\ws\high accuracy pdf parser\tmp\a1234686457a47e59112a27a9d81a116.pdf"
doc = fitz.open(pdf_path)

# 用于跨页表格检测
prev_table_info = None

print("=" * 100)
print("完整验证修复效果")
print("=" * 100)

for page_num in range(len(doc)):
    page = doc[page_num]
    print(f"\n{'='*100}")
    print(f"第 {page_num + 1} 页")
    print("=" * 100)
    
    # 先获取layout检测的table bbox
    with tempfile.TemporaryDirectory() as tmpdir:
        pix = page.get_pixmap(dpi=150)
        jpg_path = os.path.join(tmpdir, f"page_{page_num+1}.jpg")
        pix.save(jpg_path)
        
        layout = detect_layout(jpg_path)
        tables = [elem for elem in layout if elem["element_type"] == "Table"]
        
        if not tables:
            print("Layout未检测到Table")
            prev_table_info = None
            continue
        
        print(f"Layout检测到 {len(tables)} 个Table")
        
        for ti, table_elem in enumerate(tables):
            bbox = table_elem["bbox"]
            print(f"\n  表格 {ti}: bbox={bbox}, conf={table_elem['confidence']:.2f}")
            
            # 检测是否是跨页接续表格
            force_no_header = False
            if prev_table_info is not None:
                try:
                    from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI
                    pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, DEFAULT_DPI)
                    rect = fitz.Rect(pdf_bbox)
                    preview_table, preview_data, _ = _find_valid_table(page, rect)
                    
                    if preview_table and preview_data:
                        from backend.services.parse_service import _is_continuation_table
                        is_cont = _is_continuation_table(
                            preview_table.col_count,
                            preview_data[0],
                            prev_table_info["col_count"],
                            prev_table_info["last_row"]
                        )
                        if is_cont:
                            print(f"    ✓ 检测到跨页接续表格，强制不识别表头")
                            force_no_header = True
                except Exception as e:
                    pass
            
            # 提取表格
            result = extract_table_from_native(page, bbox, True, 150, force_no_header)
            print(f"    策略: {result.get('strategy', 'unknown')}")
            print(f"    尺寸: {result.get('rows', 0)}行 x {result.get('cols', 0)}列")
            
            if result.get('html'):
                # 打印HTML的前15行
                html_lines = result['html'].split('\n')
                print(f"    HTML预览 (前15行):")
                for line in html_lines[:15]:
                    print(f"      {line}")
                if len(html_lines) > 15:
                    print(f"      ... 共 {len(html_lines)} 行")
                
                # 检查是否有<th>标签（表头）
                th_count = result['html'].count('<th>')
                print(f"    表头单元格数: {th_count}")
                
                # 记录当前表格信息
                try:
                    from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI
                    pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, DEFAULT_DPI)
                    rect = fitz.Rect(pdf_bbox)
                    info_table, info_data, _ = _find_valid_table(page, rect)
                    
                    if info_table and info_data:
                        prev_table_info = {
                            "col_count": info_table.col_count,
                            "last_row": info_data[-1],
                            "page_number": page_num + 1,
                        }
                        print(f"    ✓ 已记录表格信息，用于下一页跨页检测")
                except Exception as e:
                    print(f"    ✗ 记录表格信息失败: {e}")
            else:
                print("    ✗ 未提取到HTML内容")
                prev_table_info = None

doc.close()

# 单独检查第7页最后一行问题
print("\n" + "=" * 100)
print("第7页最后一行检查")
print("=" * 100)
doc = fitz.open(pdf_path)
page7 = doc[6]
finder = page7.find_tables(strategy="lines")
if finder.tables:
    table = finder.tables[0]
    cell_data = table.extract()
    print(f"PyMuPDF检测到 {table.row_count} 行")
    print(f"\n最后5行:")
    for ri in range(max(0, table.row_count - 5), table.row_count):
        row = cell_data[ri]
        col0 = str(row[0]).strip() if row[0] else ""
        col1 = str(row[1]).strip() if len(row) > 1 and row[1] else ""
        print(f"  行{ri:2d}: {col0:20s} | {col1}")
    
    print(f"\n正确答案第7页应该有14行（从铜大公路到六六公路）")
    print(f"PyMuPDF检测到 {table.row_count} 行，{'✓ 一致' if table.row_count == 14 else '✗ 不一致'}")
    
    # 检查页面底部是否有更多内容
    print(f"\n页面底部y坐标范围:")
    print(f"  表格bbox: {table.bbox}")
    print(f"  页面高度: {page7.rect.height}")
    print(f"  表格底部到页面底部距离: {page7.rect.height - table.bbox[3]:.2f}")

doc.close()

# 检查第4页问题
print("\n" + "=" * 100)
print("第4页问题分析")
print("=" * 100)
doc = fitz.open(pdf_path)
page4 = doc[3]
print("第4页正确答案有2个8列的表格，但内容是流式文本布局（每个单元格文本被分割成多行）")
print("PyMuPDF的lines和lines_strict策略都检测不到表格")
print("text策略把整个页面当一个表格且单词被错误分割")
print("\n结论：第4页的表格由于没有任何线条，且文本是流式布局，")
print("      PyMuPDF的table能力无法正确识别。需要更高级的表格结构分析算法。")
doc.close()
