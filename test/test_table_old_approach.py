"""
PyMuPDF 原生表格提取测试 (旧版正确方案)
======================================
从 old code (init commit) 中还原的正确 table 提取逻辑。

用法:
    python test/test_table_old_approach.py <pdf_path> <page_number> [--layout-bbox x1 y1 x2 y2]

流程:
    1. 打开 PDF 页面
    2. (可选) 用 layout bbox 约束查找区域
    3. 使用 PyMuPDF find_tables() 提取表格
    4. 输出 markdown / html / 行列数

说明:
    当前系统 table 提取全面报错，因为改用 TableTransformer 后模型/依赖不稳定。
    old code 直接用 PyMuPDF 自带的 find_tables()，轻量且可靠。
    此脚本还原 old 方案作为参考。
"""

import argparse
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("table_test")


def extract_table_native(pdf_path: str, page_number: int,
                         layout_bbox: tuple[float, float, float, float] | None = None) -> dict:
    """
    从 PDF 页面用 PyMuPDF 原生方式提取表格。

    这是 old code 的正确方案:
        page.find_tables(clip=rect, strategy=TABLE_STRATEGY, ...)

    Args:
        pdf_path: PDF 文件路径
        page_number: 页码 (1-based)
        layout_bbox: (x1, y1, x2, y2) — 来自 YOLO layout 的 bbox，单位是 JPG 像素坐标。
                     内部会自动转换到 PDF 点坐标。
                     传 None 则检测整页。

    Returns:
        {"markdown": ..., "html": ..., "rows": int, "cols": int}
    """
    import fitz

    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        logger.error(f"页码越界: {page_number}, 共 {len(doc)} 页")
        doc.close()
        return {"markdown": "", "html": "", "rows": 0, "cols": 0}

    page = doc[page_number - 1]

    # ── 坐标转换 ──────────────────────────────────────────────────────
    # YOLO bbox 来自 JPG (200 DPI), PDF 使用点坐标 (72 DPI)
    # X_pdf = X_jpg * 72 / 200 = X_jpg * 0.36
    # 这是解决 picture 截取错误同款 coordinate mismatch 的关键。
    if layout_bbox is not None:
        dpi = 200
        scale = 72.0 / dpi
        rect = fitz.Rect(
            layout_bbox[0] * scale,
            layout_bbox[1] * scale,
            layout_bbox[2] * scale,
            layout_bbox[3] * scale,
        )
        logger.info(f"Layout bbox (JPG px): {layout_bbox}")
        logger.info(f"转换后 PDF rect: ({rect.x0:.1f}, {rect.y0:.1f}, {rect.x1:.1f}, {rect.y1:.1f})")
    else:
        rect = page.rect  # 整页
        logger.info("未指定 layout bbox，检测整页")

    # ── 核心：PyMuPDF find_tables (old code 方案) ─────────────────────
    # old code 使用的就是这一行:
    #   finder = page.find_tables(clip=rect, strategy=TABLE_STRATEGY, ...)
    finder = page.find_tables(
        clip=rect,
        strategy="lines_strict",  # 与 old config 一致
        snap_tolerance=5,
        join_tolerance=5,
    )

    if not finder.tables:
        logger.warning("未找到任何表格")
        # fallback: 纯文本提取（old code 也有此回退）
        text = page.get_text("text", clip=rect)
        if text.strip():
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            md_lines = []
            for i, line in enumerate(lines):
                cells = [c.strip() for c in line.split("|") if c.strip()]
                if not cells:
                    cells = [line]
                md_lines.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
            markdown = "\n".join(md_lines)
            doc.close()
            return {"markdown": markdown, "html": _md_to_html(markdown),
                    "rows": len(lines), "cols": len(lines[0].split("|")) if lines else 0}
        doc.close()
        return {"markdown": "", "html": "", "rows": 0, "cols": 0}

    table = finder.tables[0]
    logger.info(f"找到 {len(finder.tables)} 个表格，使用第一个")
    logger.info(f"  行数: {table.row_count}, 列数: {table.col_count}")

    # ── 输出 ──────────────────────────────────────────────────────────
    try:
        markdown = table.to_markdown()
    except Exception as e:
        logger.warning(f"to_markdown 失败: {e}")
        markdown = ""

    try:
        html = _table_to_html(table)
    except Exception as e:
        logger.warning(f"HTML 转换失败: {e}")
        html = ""

    doc.close()
    return {
        "markdown": markdown,
        "html": html,
        "rows": table.row_count,
        "cols": table.col_count,
    }


def _table_to_html(table) -> str:
    """将 PyMuPDF Table 对象转为 HTML（带 rowspan/colspan）"""
    extract = table.extract()
    if not extract:
        return ""

    html_parts = ["<table>"]
    for row_idx, row in enumerate(extract):
        html_parts.append("<tr>")
        for col_idx, cell in enumerate(row):
            if cell is None:
                cell = ""
            tag = "th" if row_idx == 0 else "td"
            span_attr = ""
            # 尝试获取 span 信息
            try:
                for c in table.cells:
                    if c.row_id == row_idx and c.col_id == col_idx:
                        if c.rowspan > 1:
                            span_attr += f' rowspan="{c.rowspan}"'
                        if c.colspan > 1:
                            span_attr += f' colspan="{c.colspan}"'
                        break
            except Exception:
                pass
            html_parts.append(f"<{tag}{span_attr}>{cell}</{tag}>")
        html_parts.append("</tr>")
    html_parts.append("</table>")
    return "".join(html_parts)


def _md_to_html(md: str) -> str:
    lines = md.strip().split("\n")
    if len(lines) < 2:
        return ""
    html = ["<table>"]
    cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    html.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            html.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    html.append("</table>")
    return "\n".join(html)


def main():
    parser = argparse.ArgumentParser(
        description="PyMuPDF 原生表格提取测试 (旧版正确方案)"
    )
    parser.add_argument("pdf", type=str, help="PDF 文件路径")
    parser.add_argument("page", type=int, help="页码 (1-based)")
    parser.add_argument("--layout-bbox", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="来自 layout 检测的 bbox (JPG 像素坐标)，内部转换到 PDF 坐标")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        logger.error(f"文件不存在: {pdf_path}")
        sys.exit(1)

    result = extract_table_native(str(pdf_path), args.page, args.layout_bbox)

    print("=" * 60)
    print("  表格提取结果")
    print("=" * 60)
    print(f"行数: {result['rows']}, 列数: {result['cols']}")
    print()
    print("── Markdown ──")
    print(result["markdown"] or "(空)")
    print()
    print("── HTML ──")
    print(result["html"] or "(空)")
    print("=" * 60)


if __name__ == "__main__":
    main()
