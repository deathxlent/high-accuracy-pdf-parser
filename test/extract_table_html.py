"""
从 old-code 提取的表格提取独立脚本 (TableTransformer 方案)
=======================================================
提取自 old-code/biz.py + old-code/models.py 中所有与 table 相关的代码，
合并为单个可执行脚本，输出 HTML。

用法:
    python test/extract_table_html.py <pdf_path> <page_number> \\
        [--layout-bbox x1 y1 x2 y2] \\
        [--ocr] [--lang zh,en,ja]

流程:
    1. 打开 PDF 页面，渲染为图片 (200 DPI)
    2. 使用 TableTransformer 检测表格行列结构
    3. 提取每个单元格的文本内容
    4. 输出带 colspan/rowspan 的 HTML

依赖:
    pip install torch transformers pillow pymupdf
    (如果需要 OCR 支持扫描件: pip install surya-ocr)
"""

import argparse
import sys
import logging
import os
from math import floor, ceil
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("table_extract")


# =====================================================================
# 数据模型 (来自 old-code/models.py)
# =====================================================================

class TableBbox:
    """表格单元格 bbox 集合"""
    header: List[int]
    bboxes: List[List[int]]

    def __init__(self, bboxes, header=None):
        self.bboxes = bboxes
        self.header = header

    def to_dict(self):
        return {
            "bboxes": self.bboxes,
            "header": self.header
        }


# =====================================================================
# 模型加载 (来自 old-code/biz.py)
# =====================================================================

_TABLE_TRANSFORMER_MODEL = None


def init_and_get_tt_model():
    """
    初始化 TableTransformer 结构识别模型。
    来自 old-code/biz.py 的 init_and_get_tt_model()
    """
    global _TABLE_TRANSFORMER_MODEL
    if _TABLE_TRANSFORMER_MODEL is None:
        from transformers import TableTransformerForObjectDetection
        logger.info("正在加载 TableTransformer 模型 "
                     "(microsoft/table-transformer-structure-recognition-v1.1-pub) ...")
        _TABLE_TRANSFORMER_MODEL = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-pub")
        logger.info("模型加载完成。")
    return _TABLE_TRANSFORMER_MODEL


# surya OCR 模型（懒加载，仅 --ocr 时使用）
_REC_MODEL, _REC_PROCESSOR = None, None


def get_and_init_ocr_models():
    """初始化 surya OCR 模型"""
    global _REC_MODEL, _REC_PROCESSOR
    if _REC_MODEL is None or _REC_PROCESSOR is None:
        from surya.model.recognition.model import load_model as load_rec_model
        from surya.model.recognition.processor import load_processor as load_rec_processor
        _REC_MODEL, _REC_PROCESSOR = load_rec_model(), load_rec_processor()
    return _REC_MODEL, _REC_PROCESSOR


# =====================================================================
# 图像处理辅助 (来自 old-code/biz.py)
# =====================================================================

def _add_margin(pil_img, top, right, bottom, left, color):
    """为图像添加边距 (用于 table cell 检测前的 padding)"""
    from PIL import Image
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


# =====================================================================
# 表格单元格结构检测 (来自 old-code/biz.py)
# =====================================================================

def _extract_table_cell_bboxes(table_bbox, page_image):
    """
    使用 TableTransformer 检测表格区域内的单元格结构。

    来自 old-code/biz.py 的 _extract_table_cell_bboxes()

    Args:
        table_bbox: [x1, y1, x2, y2] 表格在 page_image 中的坐标
        page_image: PIL Image 整页图片

    Returns:
        TableBbox 对象
    """
    from transformers import DetrFeatureExtractor
    import torch

    model = init_and_get_tt_model()
    feature_extractor = DetrFeatureExtractor()

    # 裁剪表格区域（加少量边距）
    cropped_img = page_image.crop(
        (table_bbox[0] - 5, table_bbox[1] - 5,
         table_bbox[2] + 5, table_bbox[3] + 5))
    padded_img = _add_margin(cropped_img, 10, 10, 10, 10, 'white')

    # 模型推理
    encoding = feature_extractor(padded_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = [padded_img.size[::-1]]
    results = feature_extractor.post_process_object_detection(
        outputs, threshold=0.6, target_sizes=target_sizes)[0]

    table_bbox_result = handle_table_bboxes(
        padded_img, results['boxes'], results['labels'])
    return table_bbox_result


# =====================================================================
# 单元格网格构建 (来自 old-code/biz.py)
# =====================================================================

def _find_nearest_value(lst, value):
    """找到列表中与目标最接近的值"""
    return min(lst, key=lambda x: abs(x - value))


def _find_nearest_index(lst, value):
    """找到列表中与目标最接近的值的索引"""
    nearest = min(lst, key=lambda x: abs(x - value))
    return lst.index(nearest)


def _handle_table_cell_border(border_set, max_border):
    """
    处理检测到的边框位置，合并距离过近的边框。
    来自 old-code/biz.py 的 _handle_table_cell_border()
    """
    result = []
    last_value = 0
    border_set = sorted(border_set)
    size = len(border_set)
    for i in range(size):
        if i == 0:
            result.append(10)
            last_value = border_set[i]
        if i == size - 1:
            result.append(max_border - 10)
            break
        if abs(border_set[i] - last_value) < 13:
            continue
        else:
            result.append(border_set[i])
            last_value = border_set[i]
    return result


def _norm_in_table_headers(in_table_headers, x_border, y_border):
    """
    将检测到的 span 坐标对齐到最近的网格线。
    来自 old-code/biz.py 的 _norm_in_table_headers()
    """
    result = []
    for in_table_header in in_table_headers:
        result.append([
            _find_nearest_value(x_border, in_table_header[0]),
            _find_nearest_value(y_border, in_table_header[1]),
            _find_nearest_value(x_border, in_table_header[2]),
            _find_nearest_value(y_border, in_table_header[3])
        ])
    return result


def _return_a_independent_cell(x, y, x1, y1, spans, has_spans):
    """
    返回独立单元格，如果被 span 覆盖则返回 None。
    来自 old-code/biz.py 的 _return_a_independent_cell()
    """
    if not has_spans:
        return [x, y, x1, y1]
    for span in spans:
        if x == span[0] and y == span[1]:
            return span
        elif x >= span[0] and y >= span[1] and x1 <= span[2] and y1 <= span[3]:
            return None
    return [x, y, x1, y1]


def handle_table_bboxes(image, boxes, labels):
    """
    将 TableTransformer 的检测结果转换为单元格网格。

    来自 old-code/biz.py 的 handle_table_bboxes()

    TableTransformer 标签含义:
        0: table         1: table column
        2: table row     3: table column header
        4: table projected cell header
        5: table spanning cell

    Args:
        image: PIL Image (已 padding 的表格图片)
        boxes: 检测框张量
        labels: 标签张量

    Returns:
        TableBbox 对象
    """
    width, height = image.size
    x_set = set()
    y_set = set()
    table_header = None
    spans = []

    for i in range(len(boxes)):
        label = int(labels[i])
        bbox = boxes[i].tolist()
        if label == 1:  # table column
            x_set.add(bbox[0])
            x_set.add(bbox[2])
        elif label == 2:  # table row
            y_set.add(bbox[1])
            y_set.add(bbox[3])
        if label == 3:  # table column header
            table_header = [10, 10, width - 10, bbox[3]]
        elif label == 4 or label == 5:  # projected cell header / spanning cell
            spans.append(bbox)

    x_border = _handle_table_cell_border(x_set, width)
    y_border = _handle_table_cell_border(y_set, height)

    if table_header is not None:
        table_header = [
            table_header[0], table_header[1], table_header[2],
            _find_nearest_value(y_border, table_header[3])
        ]

    spans = _norm_in_table_headers(spans, x_border, y_border)
    no_span = len(spans) > 0

    bboxes = []
    for x_idx in range(len(x_border) - 1):
        for y_idx in range(len(y_border) - 1):
            bbox = _return_a_independent_cell(
                x_border[x_idx], y_border[y_idx],
                x_border[x_idx + 1], y_border[y_idx + 1],
                spans, no_span)
            if bbox is not None:
                bboxes.append(bbox)

    return TableBbox(bboxes, table_header)


# =====================================================================
# 文本提取 (来自 old-code/biz.py)
# =====================================================================

def _ocr(image, bbox=None):
    """使用 surya OCR 识别图片中的文字"""
    from surya.ocr import run_recognition
    result = []
    try:
        rec_model, rec_processor = get_and_init_ocr_models()
        if bbox:
            ocr_result = run_recognition(
                [image], [['zh', 'en', 'ja']],
                rec_model, rec_processor, [[bbox]])
        else:
            ocr_result = run_recognition(
                [image], [['zh', 'en', 'ja']],
                rec_model, rec_processor)
        if ocr_result and ocr_result[0] and ocr_result[0].text_lines:
            for text_line in ocr_result[0].text_lines:
                result.append(text_line.text.strip())
    except Exception as e:
        logger.exception(e)
    return " ".join(result)


def _ocr_batch(image_path, bboxes):
    """批量 OCR 识别"""
    from PIL import Image
    from surya.ocr import run_recognition
    result = []
    try:
        rec_model, rec_processor = get_and_init_ocr_models()
        ocr_result = run_recognition(
            [Image.open(image_path)], [['zh', 'en', 'ja']],
            rec_model, rec_processor, [bboxes])
        if ocr_result and ocr_result[0] and ocr_result[0].text_lines:
            for text_line in ocr_result[0].text_lines:
                result.append(text_line.text)
    except Exception as e:
        logger.exception(e)
    return result


def _extract_text_from_bbox(page, bbox, is_scanned, page_image=None):
    """
    从指定 bbox 提取文本。

    来自 old-code/biz.py 的 _extract_text()

    Args:
        page: fitz.Page 对象（非扫描件用）
        bbox: [x1, y1, x2, y2]
        is_scanned: 是否扫描件
        page_image: PIL Image（扫描件用）
    """
    if is_scanned:
        return _ocr(page_image, bbox)
    else:
        # PyMuPDF 坐标直接提取
        return page.get_textbox([
            floor(bbox[0]), floor(bbox[1]),
            ceil(bbox[2]), ceil(bbox[3])
        ])


# =====================================================================
# HTML 生成 (来自 old-code/biz.py)
# =====================================================================

def _parse_span_num(value1, value2, indexes):
    """计算 colspan/rowspan"""
    p1 = _find_nearest_index(indexes, value1)
    p2 = _find_nearest_index(indexes, value2)
    return p2 - p1


def _parse_2_html_and_extract_text(cell_bboxes, table_bbox_in_page,
                                    is_scanned, page, page_image_path):
    """
    将单元格 bbox 网格转换为 HTML 表格，并提取每个单元格的文本。

    来自 old-code/biz.py 的 _parse_2_html_and_extract_text()

    Args:
        cell_bboxes: TableBbox 对象
        table_bbox_in_page: [x1, y1, x2, y2] 表格在整页中的坐标
        is_scanned: 是否扫描件
        page: fitz.Page 对象
        page_image_path: 页面图片路径

    Returns:
        HTML 字符串
    """
    from PIL import Image

    position_maps = []
    indexes_x_set = set()
    indexes_y_set = set()
    image = Image.open(page_image_path)

    # 裁剪表格区域（与 cell 检测时一致，留边距）
    cropped_img = image.crop((
        table_bbox_in_page[0] - 15,
        table_bbox_in_page[1] - 15,
        table_bbox_in_page[2] + 15,
        table_bbox_in_page[3] + 15
    ))

    column_num = -1
    column_y = -1111
    column_bboxes = []
    sorted_bboxes = sorted(cell_bboxes.bboxes, key=lambda x: (x[1], x[0]))

    for cell_bbox in sorted_bboxes:
        if cell_bbox[1] != column_y:
            column_num += 1
            if column_num > 0:
                position_maps.append(column_bboxes)
            column_bboxes = []
            column_y = cell_bbox[1]
        column_bboxes.append(cell_bbox)
        indexes_x_set.add(cell_bbox[0])
        indexes_x_set.add(cell_bbox[2])
        indexes_y_set.add(cell_bbox[1])
        indexes_y_set.add(cell_bbox[3])
    position_maps.append(column_bboxes)

    result_html_table = ["<table border='1'>"]
    indexes_x = sorted(list(indexes_x_set))
    ocr_txt = []

    if is_scanned:
        # 批量 OCR（扫描件：一次性提交所有 bbox）
        bboxes = []
        for items in position_maps:
            for bbox in items:
                bboxes.append(bbox)
        ocr_txt = _ocr_batch(page_image_path, bboxes)

    ocr_txt_index = 0
    ocr_result_length = len(ocr_txt)

    if len(indexes_x) > 0:
        indexes_y = sorted(list(indexes_y_set))

        for items in position_maps:
            result_html_table.append("<tr>")
            for bbox in items:
                colspan = _parse_span_num(bbox[0], bbox[2], indexes_x)
                rowspan = _parse_span_num(bbox[1], bbox[3], indexes_y)
                colspan_str = f" colspan={colspan}" if colspan > 1 else ""
                rowspan_str = f" rowspan={rowspan}" if rowspan > 1 else ""

                text = ""
                if is_scanned:
                    if ocr_result_length > 0 and ocr_txt_index < ocr_result_length:
                        text = ocr_txt[ocr_txt_index]
                        ocr_txt_index += 1
                    else:
                        text = _extract_text_from_bbox(
                            page, bbox, is_scanned, cropped_img)
                else:
                    # 非扫描件：用 PyMuPDF 从原始 PDF 坐标提取
                    pdf_bbox = [
                        bbox[0] - 10 + table_bbox_in_page[0],
                        bbox[1] - 10 + table_bbox_in_page[1],
                        bbox[2] - 10 + table_bbox_in_page[0],
                        bbox[3] - 10 + table_bbox_in_page[1]
                    ]
                    text = _extract_text_from_bbox(
                        page, pdf_bbox, is_scanned, cropped_img)

                result_html_table.append(
                    f"<td{colspan_str}{rowspan_str}>{text}</td>")
            result_html_table.append("</tr>")

    result_html_table.append("</table>")
    return "\n".join(result_html_table)


def table_to_html(table_bbox, table_bbox_in_page,
                  is_scanned, page, page_image_path):
    """
    一键将表格检测结果转为 HTML。

    这是 _parse_2_html_and_extract_text 的封装，相当于
    old-code 中 _post_process_table 的 HTML 生成部分。

    Args:
        table_bbox: TableBbox 对象（来自 _extract_table_cell_bboxes）
        table_bbox_in_page: [x1, y1, x2, y2] 表格在原图中的坐标
        is_scanned: 是否扫描件
        page: fitz.Page 对象
        page_image_path: 页面图片路径

    Returns:
        HTML 字符串
    """
    return _parse_2_html_and_extract_text(
        table_bbox, table_bbox_in_page,
        is_scanned, page, page_image_path)


# =====================================================================
# 一站式表格提取入口
# =====================================================================

def extract_table_from_page(pdf_path, page_number,
                            layout_bbox=None, use_ocr=False,
                            dpi=200):
    """
    从 PDF 页面提取表格并输出 HTML。

    整合 old-code 中 table 提取的完整流程:
        1. 渲染页面为图片
        2. 裁切表格区域
        3. TableTransformer 检测单元格结构
        4. 生成 HTML

    Args:
        pdf_path: PDF 文件路径
        page_number: 页码 (1-based)
        layout_bbox: (x1, y1, x2, y2) 来自 layout 检测的 bbox (JPG 像素坐标)，
                     传 None 则使用整页
        use_ocr: 是否使用 OCR（扫描件设为 True）
        dpi: 页面渲染 DPI（默认 200，与 old-code 一致）

    Returns:
        {"html": str, "rows": int, "cols": int}
    """
    import fitz
    from PIL import Image

    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        logger.error(f"页码越界: {page_number}, 共 {len(doc)} 页")
        doc.close()
        return {"html": "", "rows": 0, "cols": 0}

    page = doc[page_number - 1]

    # ── 1. 渲染页面为图片 ──────────────────────────────────────────
    # old-code 使用 scale=2 渲染 (即 72*2=144 DPI)，但设置的是 scale=2
    # 对应 Matrix(2, 0, 0, 2, 0, 0)，实际是 144 DPI
    # 这里用 dpi 参数控制
    scale = dpi / 72.0
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    temp_img_path = f"{pdf_path}_page_{page_number}_table_temp.png"
    pix.save(temp_img_path)

    # 确定表格区域（像素坐标）
    if layout_bbox is not None:
        table_area = list(layout_bbox)
        logger.info(f"Layout bbox (JPG px): {table_area}")
    else:
        # 无 bbox 时使用整页，但一般不会有好结果
        logger.warning("未提供 layout_bbox，使用整页作为表格区域（效果可能不佳）")
        img = Image.open(temp_img_path)
        table_area = [0, 0, img.width, img.height]

    # ── 2. 检测表格单元格结构 ──────────────────────────────────────
    page_image = Image.open(temp_img_path)
    cell_bboxes = _extract_table_cell_bboxes(table_area, page_image)

    if not cell_bboxes.bboxes:
        logger.warning("TableTransformer 未检测到单元格结构")
        doc.close()
        os.remove(temp_img_path)
        return {"html": "", "rows": 0, "cols": 0}

    logger.info(f"检测到 {len(cell_bboxes.bboxes)} 个单元格")

    # ── 3. 生成 HTML ──────────────────────────────────────────────
    # 注意: old-code 的 _parse_2_html_and_extract_text 中，
    # cell_bboxes 坐标是相对于裁剪后(已padding)的图片的。
    # handle_table_bboxes 在 padded_img 上运行，加上了 10px margin，
    # 而 crop 时也减了 5px。所以 cell bbox -> 原图坐标的偏移量是:
    #   offset_x = table_area[0] - 15 + 10 = table_area[0] - 5
    #   offset_y = table_area[1] - 15 + 10 = table_area[1] - 5
    # 但 _parse_2_html_and_extract_text 内部在提取文本时已经处理了偏移，
    # 所以直接传入原始的 table_area 即可。
    is_scanned = use_ocr
    html = _parse_2_html_and_extract_text(
        cell_bboxes, table_area, is_scanned, page, temp_img_path)

    # 统计行列
    rows = len(cell_bboxes.bboxes)
    cols = 1
    if rows > 0:
        # 大致估计列数（取第一行单元格数）
        first_row_y = cell_bboxes.bboxes[0][1]
        first_row_cells = [b for b in cell_bboxes.bboxes if b[1] == first_row_y]
        cols = len(first_row_cells)

    # 清理临时文件
    doc.close()
    try:
        os.remove(temp_img_path)
    except Exception:
        pass

    return {
        "html": html,
        "rows": rows,
        "cols": cols,
    }


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 PDF 提取表格并输出 HTML (old-code TableTransformer 方案)")
    parser.add_argument("pdf", type=str, help="PDF 文件路径")
    parser.add_argument("page", type=int, help="页码 (1-based)")
    parser.add_argument("--layout-bbox", type=float, nargs=4,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Layout 检测到的表格 bbox (像素坐标)，格式: x1 y1 x2 y2")
    parser.add_argument("--ocr", action="store_true",
                        help="启用 OCR（扫描件 PDF）")
    parser.add_argument("--dpi", type=int, default=200,
                        help="页面渲染 DPI (默认: 200)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出 HTML 文件路径 (默认输出到终端)")
    args = parser.parse_args()

    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        logger.error(f"文件不存在: {pdf_path}")
        sys.exit(1)

    result = extract_table_from_page(
        pdf_path, args.page,
        layout_bbox=args.layout_bbox,
        use_ocr=args.ocr,
        dpi=args.dpi,
    )

    print("=" * 60)
    print("  表格提取结果")
    print("=" * 60)
    print(f"行数: {result['rows']}, 列数: {result['cols']}")
    print()
    print("── HTML ──")
    print(result["html"] or "(空)")
    print("=" * 60)

    if args.output and result["html"]:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["html"])
        logger.info(f"HTML 已保存至: {args.output}")


if __name__ == "__main__":
    main()
