"""
表格提取服务
===========

功能:
    从 PDF 页面中提取表格，输出带 rowspan/colspan 的 HTML 格式。

设计说明:
    1. 非扫描件: 使用 PyMuPDF 原生 find_tables()（稳定可靠，支持跨行跨列）
    2. 扫描件:  使用 OCR 识别表格区域后转为 HTML
    3. 输出格式: 以 HTML 为主（天然支持 colspan/rowspan），Markdown 作为 fallback

坐标处理:
    传入的 bbox 默认是 JPG 图片的像素坐标（200 DPI），
    内部会自动转换为 PDF 点坐标（72 DPI）后再进行提取。
"""

import logging
import fitz
from backend.services.ocr_service import ocr_region
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI

logger = logging.getLogger(__name__)

# PyMuPDF find_tables 配置参数
TABLE_STRATEGY = "lines_strict"  # 严格基于线条检测，适合规整表格
SNAP_TOLERANCE = 5               # 线条吸附容差（像素）
JOIN_TOLERANCE = 5               # 线条连接容差（像素）


def _parse_span_num(cell_start: float, cell_end: float, indexes: list[float],
                    tolerance: float = 1.0) -> int:
    """
    计算单元格跨越的行/列数（参考 old-code 的 _parse_span_num）。

    原理:
        1. indexes 是排序后的网格线坐标（x 坐标用于 colspan，y 坐标用于 rowspan）
        2. 找到 cell_start 最接近的起始索引，cell_end 最接近的结束索引
        3. 结束索引 - 起始索引 = 跨越的格子数

    Args:
        cell_start: 单元格的起始坐标（左或上）
        cell_end: 单元格的结束坐标（右或下）
        indexes: 排序后的网格线坐标列表
        tolerance: 容差（默认 1 像素）

    Returns:
        跨越的格子数（>= 1）
    """
    if not indexes:
        return 1

    # 找到起始坐标在 indexes 中的位置（允许一定容差）
    start_idx = 0
    for i, val in enumerate(indexes):
        if abs(val - cell_start) <= tolerance:
            start_idx = i
            break
        elif val > cell_start:
            start_idx = max(0, i - 1)
            break
    else:
        start_idx = len(indexes) - 1

    # 找到结束坐标在 indexes 中的位置（允许一定容差）
    end_idx = len(indexes) - 1
    for i in range(len(indexes) - 1, -1, -1):
        if abs(indexes[i] - cell_end) <= tolerance:
            end_idx = i
            break
        elif indexes[i] < cell_end:
            end_idx = min(len(indexes) - 1, i + 1)
            break
    else:
        end_idx = 0

    span = end_idx - start_idx
    return max(1, span)


def _table_to_html(table) -> str:
    """
    将 PyMuPDF Table 对象转换为 HTML 表格。

    关键实现（参考 old-code 的思路）:
        1. 从所有单元格的 bbox 中提取 x/y 坐标，建立网格线
        2. 通过 _parse_span_num 计算每个单元格的 colspan/rowspan
        3. 使用 covered 标记矩阵避免重复输出被跨越的单元格
        4. 第一行作为表头使用 <th>，其余使用 <td>
        5. HTML 特殊字符转义，避免注入

    Args:
        table: PyMuPDF Table 对象（来自 page.find_tables()）

    Returns:
        HTML 表格字符串，如果没有数据则返回空字符串
    """
    cell_data = table.extract()
    if not cell_data or not cell_data[0]:
        return ""

    row_count = len(cell_data)
    col_count = len(cell_data[0])

    # ── 步骤1: 收集所有单元格的 bbox，建立网格线 ────────────────────
    # table.cells 返回的是平铺的单元格列表（按行优先顺序）
    # 每个 cell 是 (x0, y0, x1, y1) 元组
    cells_bbox = table.cells

    # 提取所有 x 和 y 坐标，去重并排序，得到网格线
    x_coords = []
    y_coords = []
    for bbox in cells_bbox:
        x0, y0, x1, y1 = bbox
        x_coords.extend([x0, x1])
        y_coords.extend([y0, y1])

    # 去重并排序（合并相近的坐标，容差 1 像素）
    def _dedup_and_sort(coords: list[float], tol: float = 1.0) -> list[float]:
        if not coords:
            return []
        coords = sorted(set(round(c, 2) for c in coords))
        result = [coords[0]]
        for c in coords[1:]:
            if c - result[-1] > tol:
                result.append(c)
        return result

    indexes_x = _dedup_and_sort(x_coords)  # 垂直线（列分隔线）
    indexes_y = _dedup_and_sort(y_coords)  # 水平线（行分隔线）

    logger.debug(f"Grid lines: {len(indexes_x)} vertical, {len(indexes_y)} horizontal")

    # ── 步骤2: 构建覆盖标记矩阵 ────────────────────────────────────
    covered = [[False for _ in range(col_count)] for _ in range(row_count)]

    html_parts = ["<table border='1' cellpadding='4' cellspacing='0'>"]

    # ── 步骤3: 遍历单元格生成 HTML ─────────────────────────────────
    for row_idx in range(row_count):
        html_parts.append("  <tr>")
        for col_idx in range(col_count):
            # 如果当前位置已被前面的跨行跨列覆盖，则跳过
            if covered[row_idx][col_idx]:
                continue

            # 平铺索引：row_idx * col_count + col_idx
            cell_flat_idx = row_idx * col_count + col_idx
            if cell_flat_idx >= len(cells_bbox):
                logger.warning(f"Cell index out of range: {cell_flat_idx}/{len(cells_bbox)}")
                continue

            bbox = cells_bbox[cell_flat_idx]
            x0, y0, x1, y1 = bbox

            # ── 计算 colspan/rowspan（参考 old-code 的 _parse_2_html_and_extract_text） ──
            colspan = _parse_span_num(x0, x1, indexes_x)
            rowspan = _parse_span_num(y0, y1, indexes_y)

            # 边界检查
            if row_idx + rowspan > row_count:
                rowspan = row_count - row_idx
            if col_idx + colspan > col_count:
                colspan = col_count - col_idx
            rowspan = max(1, rowspan)
            colspan = max(1, colspan)

            # 标记被覆盖的位置
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    if r < row_count and c < col_count:
                        covered[r][c] = True

            # 获取单元格文本
            cell_value = cell_data[row_idx][col_idx] or ""

            # 标签选择
            tag = "th" if row_idx == 0 else "td"

            # 构建 span 属性
            span_attrs = ""
            if colspan > 1:
                span_attrs += f" colspan='{colspan}'"
            if rowspan > 1:
                span_attrs += f" rowspan='{rowspan}'"

            # HTML 特殊字符转义
            cell_value = cell_value.replace("&", "&amp;")
            cell_value = cell_value.replace("<", "&lt;").replace(">", "&gt;")

            html_parts.append(f"    <{tag}{span_attrs}>{cell_value}</{tag}>")
        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


def _text_to_html_table(text: str) -> str:
    """
    当 find_tables 未检测到表格时，尝试从纯文本构建 HTML 表格。

    这是 fallback 方案，参考 old-code 中未检测到表格时的处理逻辑:
        - 按行分割文本
        - 尝试用 | 或空白符分割单元格
        - 生成简单的 HTML 表格（无 span）

    Args:
        text: 从 PDF 区域提取的纯文本

    Returns:
        HTML 表格字符串，如果文本为空则返回空字符串
    """
    if not text or not text.strip():
        return ""

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""

    html_parts = ["<table border='1' cellpadding='4' cellspacing='0'>"]

    for i, line in enumerate(lines):
        # 优先尝试用 | 分割，其次用多空格分割
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) <= 1:
            cells = [c.strip() for c in line.split() if c.strip()]
        if not cells:
            cells = [line]

        tag = "th" if i == 0 else "td"
        html_parts.append("  <tr>")
        for cell in cells:
            cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"    <{tag}>{cell}</{tag}>")
        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


def _markdown_to_html_simple(md: str) -> str:
    """
    简单的 Markdown 表格转 HTML（无 span 支持，仅作 fallback）。

    注意: Markdown 表格语法本身不支持跨行跨列，
    所以这只是一个简化的转换，不保证复杂表格的正确性。

    Args:
        md: Markdown 格式的表格文本

    Returns:
        HTML 表格字符串
    """
    if not md or not md.strip():
        return ""

    lines = md.strip().split("\n")
    if len(lines) < 2:
        return ""

    html_parts = ["<table border='1' cellpadding='4' cellspacing='0'>"]

    # 第一行是表头
    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    html_parts.append("  <tr>")
    for cell in header_cells:
        cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_parts.append(f"    <th>{cell}</th>")
    html_parts.append("  </tr>")

    # 跳过第二行（分隔线），从第三行开始是数据
    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        html_parts.append("  <tr>")
        for cell in cells:
            cell = cell.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(f"    <td>{cell}</td>")
        html_parts.append("  </tr>")

    html_parts.append("</table>")
    return "\n".join(html_parts)


def extract_table_from_native(pdf_page: fitz.Page,
                              bbox: tuple[float, float, float, float] | None,
                              bbox_is_jpg: bool = True,
                              dpi: int = DEFAULT_DPI) -> dict:
    """
    从非扫描 PDF 页面提取表格（使用 PyMuPDF 原生方案）。

    流程:
        1. 坐标转换: 如果传入的是 JPG 像素坐标，先转为 PDF 点坐标
        2. 表格检测: 使用 page.find_tables() 在指定区域内检测表格
        3. HTML 生成: 将检测到的表格转换为带 rowspan/colspan 的 HTML
        4. Fallback: 如果未检测到表格，尝试从纯文本构建 HTML

    Args:
        pdf_page: PyMuPDF Page 对象
        bbox: 表格区域坐标 (x1, y1, x2, y2)，传 None 则检测整页
        bbox_is_jpg: bbox 是否为 JPG 像素坐标（默认 True，会自动转换）
        dpi: JPG 图片的 DPI，用于坐标转换（默认 200）

    Returns:
        字典包含:
            - html: HTML 格式的表格（主要输出）
            - markdown: Markdown 格式的表格（fallback）
            - rows: 行数
            - cols: 列数
    """
    # ── 步骤1: 坐标转换 ──────────────────────────────────────────────
    if bbox is not None and bbox_is_jpg:
        # JPG 像素坐标 → PDF 点坐标: pdf = jpg * (72 / dpi)
        pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, dpi)
        rect = fitz.Rect(pdf_bbox)
    elif bbox is not None:
        rect = fitz.Rect(bbox)
    else:
        rect = pdf_page.rect

    # 边界检查
    if rect.is_empty or not rect.is_valid:
        logger.warning(f"Invalid table bbox: {bbox}")
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    # ── 步骤2: 使用 PyMuPDF 检测表格 ────────────────────────────────
    try:
        finder = pdf_page.find_tables(
            clip=rect,
            strategy=TABLE_STRATEGY,
            snap_tolerance=SNAP_TOLERANCE,
            join_tolerance=JOIN_TOLERANCE,
        )
    except Exception as e:
        logger.error(f"PyMuPDF find_tables failed: {e}")
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    # ── 步骤3: 处理检测结果 ─────────────────────────────────────────
    if not finder.tables:
        # 未检测到表格，fallback 到纯文本解析
        logger.debug("No tables detected by find_tables, trying text fallback")
        text = pdf_page.get_text("text", clip=rect)
        html = _text_to_html_table(text)
        # 粗略估算行列数
        lines = [l for l in text.split("\n") if l.strip()]
        cols = 0
        if lines:
            cols = max(len(l.split("|")) for l in lines)
        return {
            "html": html,
            "markdown": "",
            "rows": len(lines),
            "cols": cols,
        }

    # 取第一个表格（一般一个区域只有一个表格）
    table = finder.tables[0]
    if len(finder.tables) > 1:
        logger.info(f"Found {len(finder.tables)} tables in region, using first one")

    # ── 步骤4: 生成 HTML（参考 old-code 的 _parse_2_html_and_extract_text） ──
    try:
        html = _table_to_html(table)
    except Exception as e:
        logger.warning(f"Table to HTML conversion failed: {e}")
        html = ""

    # Markdown 作为 fallback（注意: 不支持 rowspan/colspan）
    try:
        markdown = table.to_markdown()
    except Exception as e:
        logger.debug(f"Table to Markdown failed: {e}")
        markdown = ""

    return {
        "html": html,
        "markdown": markdown,
        "rows": table.row_count,
        "cols": table.col_count,
    }


def extract_table_from_scanned(image_path: str, bbox: tuple) -> dict:
    """
    从扫描件 PDF 提取表格（使用 OCR 方案）。

    流程:
        1. OCR 识别: 对表格区域进行 OCR，获取纯文本
        2. 文本解析: 尝试按行和分隔符解析为表格结构
        3. HTML 生成: 转换为简单 HTML 表格（扫描件无法准确识别 span）

    注意:
        扫描件的跨行跨列识别非常困难，此函数生成的 HTML
        不包含 rowspan/colspan，仅为简单的网格结构。
        如果需要精确的跨行跨列，需要使用 TableTransformer 等
        更复杂的模型（参考 old-code 的 extract_table_html.py）。

    Args:
        image_path: 页面 JPG 图片路径
        bbox: 表格区域在 JPG 图片中的坐标（像素）

    Returns:
        字典包含:
            - html: HTML 格式的表格
            - markdown: Markdown 格式的表格
            - rows: 行数
            - cols: 列数
    """
    text = ocr_region(image_path, bbox)
    if not text.strip():
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    # 将 OCR 文本解析为表格
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 构建 Markdown（简化版，无 span）
    md_lines = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split() if c.strip()]
        if not cells:
            cells = [line]
        md_lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

    markdown = "\n".join(md_lines)

    # 构建 HTML（扫描件无法准确识别跨行跨列，生成简单表格）
    html = _text_to_html_table(text)

    # 粗略估算行列数
    rows = len(lines)
    cols = 0
    if md_lines:
        first_line_cells = [c for c in md_lines[0].split("|") if c.strip()]
        cols = len(first_line_cells)

    return {
        "html": html,
        "markdown": markdown,
        "rows": rows,
        "cols": cols,
    }
