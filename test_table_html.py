"""
测试表格 HTML 输出，特别是跨行跨列处理
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz
from backend.services.table_service import (
    extract_table_from_native,
    _table_to_html,
    _text_to_html_table,
    TABLE_STRATEGY,
)
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI

print("=" * 70)
print("Testing Table HTML Output (with rowspan/colspan)")
print("=" * 70)

# 找到一个测试 PDF
test_pdf = None
tmp_dirs = list(Path("tmp").glob("*"))
for d in tmp_dirs:
    pdfs = list(d.glob("*.pdf"))
    if pdfs:
        # 找单页 PDF
        single_pages = list(d.glob("page_*.pdf"))
        if single_pages:
            test_pdf = str(single_pages[0])
        else:
            test_pdf = str(pdfs[0])
        break

if not test_pdf or not os.path.exists(test_pdf):
    print("ERROR: No test PDF found")
    sys.exit(1)

print(f"\nUsing test PDF: {os.path.basename(test_pdf)}")

doc = fitz.open(test_pdf)
if len(doc) == 0:
    print("ERROR: PDF has no pages")
    sys.exit(1)

page = doc[0]
page_rect = page.rect
print(f"Page size: {page_rect.width:.1f}x{page_rect.height:.1f} pts")

# ── 测试1: 整页表格提取 ─────────────────────────────────────────
print("\n" + "-" * 70)
print("Test 1: Extract table from full page")
print("-" * 70)

# 模拟 JPG bbox（整页）
jpg_bbox = (0, 0, page_rect.width * 200/72, page_rect.height * 200/72)
print(f"Simulated JPG bbox: ({jpg_bbox[0]:.1f}, {jpg_bbox[1]:.1f}, {jpg_bbox[2]:.1f}, {jpg_bbox[3]:.1f})")

result = extract_table_from_native(page, jpg_bbox, bbox_is_jpg=True)

print(f"\nResult:")
print(f"  Rows: {result['rows']}")
print(f"  Cols: {result['cols']}")
print(f"  Has HTML: {bool(result.get('html'))}")
print(f"  Has Markdown: {bool(result.get('markdown'))}")

if result.get('html'):
    print(f"\n── HTML Output ──")
    print(result['html'])

    # 检查是否包含 rowspan/colspan
    has_rowspan = 'rowspan' in result['html']
    has_colspan = 'colspan' in result['html']
    print(f"\n── Span Detection ──")
    print(f"  Has rowspan: {has_rowspan}")
    print(f"  Has colspan: {has_colspan}")

if result.get('markdown'):
    print(f"\n── Markdown Output (fallback, no span support) ──")
    print(result['markdown'][:300] + "..." if len(result['markdown']) > 300 else result['markdown'])

# ── 测试2: 直接调用 _table_to_html 检查 span 逻辑 ───────────────
print("\n" + "-" * 70)
print("Test 2: Direct _table_to_html test (if table found)")
print("-" * 70)

try:
    finder = page.find_tables(
        clip=page.rect,
        strategy=TABLE_STRATEGY,
        snap_tolerance=5,
        join_tolerance=5,
    )
    if finder.tables:
        table = finder.tables[0]
        print(f"Found table: {table.row_count} rows x {table.col_count} cols")

        # 检查是否有跨行跨列单元格（通过 bbox 分析）
        cells_bbox = table.cells
        x_coords = []
        y_coords = []
        for bbox in cells_bbox:
            x0, y0, x1, y1 = bbox
            x_coords.extend([x0, x1])
            y_coords.extend([y0, y1])

        def _dedup_and_sort(coords, tol=1.0):
            coords = sorted(set(round(c, 2) for c in coords))
            result = [coords[0]]
            for c in coords[1:]:
                if c - result[-1] > tol:
                    result.append(c)
            return result

        indexes_x = _dedup_and_sort(x_coords)
        indexes_y = _dedup_and_sort(y_coords)

        print(f"Grid lines: {len(indexes_x)} vertical, {len(indexes_y)} horizontal")
        print(f"Expected cells if no spanning: {(len(indexes_x)-1)} x {(len(indexes_y)-1)} = {(len(indexes_x)-1)*(len(indexes_y)-1)}")
        print(f"Actual cells detected: {len(cells_bbox)}")

        if len(cells_bbox) < (len(indexes_x)-1) * (len(indexes_y)-1):
            print("\n✓ This table likely has spanned cells!")
        else:
            print("\n⚠ This table likely does not have spanned cells")

        html = _table_to_html(table)
        print(f"\nHTML generated successfully: {len(html)} chars")
        has_rowspan = 'rowspan' in html
        has_colspan = 'colspan' in html
        print(f"Has rowspan: {has_rowspan}")
        print(f"Has colspan: {has_colspan}")

        # 验证 HTML 结构正确性
        import re
        td_th_count = len(re.findall(r'<t[hd]', html))
        tr_count = len(re.findall(r'<tr>', html))
        print(f"HTML check: {tr_count} rows, {td_th_count} cells")
        print(f"HTML check: {'valid' if tr_count == table.row_count else '⚠ row count mismatch'}")
    else:
        print("No tables found by find_tables()")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# ── 测试3: 纯文本转 HTML fallback ─────────────────────────────
print("\n" + "-" * 70)
print("Test 3: Text to HTML fallback")
print("-" * 70)

test_text = """Col1|Col2|Col3
A|B|C
D|E|F"""

html = _text_to_html_table(test_text)
print(f"Input text:\n{test_text}")
print(f"\nOutput HTML:\n{html}")

# ── 测试4: 坐标转换验证 ─────────────────────────────────────────
print("\n" + "-" * 70)
print("Test 4: Coordinate conversion")
print("-" * 70)

test_jpg_bbox = (100, 100, 500, 300)
pdf_bbox = jpg_bbox_to_pdf_bbox(test_jpg_bbox)
scale = 72.0 / DEFAULT_DPI
expected = (test_jpg_bbox[0]*scale, test_jpg_bbox[1]*scale,
            test_jpg_bbox[2]*scale, test_jpg_bbox[3]*scale)

print(f"JPG bbox: {test_jpg_bbox}")
print(f"PDF bbox: {tuple(round(x, 2) for x in pdf_bbox)}")
print(f"Expected: {tuple(round(x, 2) for x in expected)}")
print(f"Correct: {abs(pdf_bbox[0] - expected[0]) < 0.01}")

doc.close()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
