import logging
import numpy as np
from PIL import Image
from backend.services.ocr_service import ocr_region

logger = logging.getLogger(__name__)

_feature_extractor = None
_table_model = None


def _get_table_transformer():
    global _feature_extractor, _table_model
    if _feature_extractor is not None and _table_model is not None:
        return _feature_extractor, _table_model

    try:
        from transformers import TableTransformerForObjectDetection, DetrFeatureExtractor

        logger.info("Initializing TableTransformer model...")

        _feature_extractor = DetrFeatureExtractor.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-pub"
        )
        _table_model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-structure-recognition-v1.1-pub"
        )

        logger.info("TableTransformer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load TableTransformer: {e}")
        raise

    return _feature_extractor, _table_model


def _class_to_label(class_id: int) -> str:
    labels = [
        "table", "table column", "table row", "table column header",
        "table projected row header", "table spanning cell"
    ]
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return f"unknown-{class_id}"


def _iou(box1: tuple, box2: tuple) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _process_table_structure(outputs, img_width: int, img_height: int, threshold: float = 0.7):
    target_sizes = [(img_height, img_width)]
    results = _feature_extractor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]

    rows = []
    columns = []
    column_headers = []
    spanning_cells = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = _class_to_label(label.item())

        if label_name == "table row":
            rows.append({"bbox": box, "score": score.item()})
        elif label_name == "table column":
            columns.append({"bbox": box, "score": score.item()})
        elif label_name == "table column header":
            column_headers.append({"bbox": box, "score": score.item()})
        elif label_name == "table spanning cell":
            spanning_cells.append({"bbox": box, "score": score.item()})

    rows.sort(key=lambda r: r["bbox"][1])
    columns.sort(key=lambda c: c["bbox"][0])

    return rows, columns, column_headers, spanning_cells


def _build_table_cells(rows: list, columns: list, spanning_cells: list):
    if not rows or not columns:
        return [], []

    cell_grid = [[None for _ in range(len(columns))] for _ in range(len(rows))]
    row_spans = [[1 for _ in range(len(columns))] for _ in range(len(rows))]
    col_spans = [[1 for _ in range(len(columns))] for _ in range(len(rows))]

    for sc in spanning_cells:
        sc_bbox = sc["bbox"]

        start_row, end_row = 0, len(rows) - 1
        for i, row in enumerate(rows):
            if row["bbox"][1] <= sc_bbox[1] <= row["bbox"][3]:
                start_row = i
                break
        for i in range(len(rows) - 1, -1, -1):
            if rows[i]["bbox"][1] <= sc_bbox[3] <= rows[i]["bbox"][3]:
                end_row = i
                break

        start_col, end_col = 0, len(columns) - 1
        for j, col in enumerate(columns):
            if col["bbox"][0] <= sc_bbox[0] <= col["bbox"][2]:
                start_col = j
                break
        for j in range(len(columns) - 1, -1, -1):
            if columns[j]["bbox"][0] <= sc_bbox[2] <= columns[j]["bbox"][2]:
                end_col = j
                break

        for i in range(start_row, min(end_row + 1, len(rows))):
            for j in range(start_col, min(end_col + 1, len(columns))):
                cell_grid[i][j] = "spanned"
                row_spans[i][j] = end_row - start_row + 1
                col_spans[i][j] = end_col - start_col + 1

    cell_bboxes = []
    for i, row in enumerate(rows):
        row_bboxes = []
        for j, col in enumerate(columns):
            if cell_grid[i][j] == "spanned":
                row_bboxes.append(None)
                continue

            x0 = col["bbox"][0]
            y0 = row["bbox"][1]
            x1 = col["bbox"][2]
            y1 = row["bbox"][3]

            if i > 0 and j > 0 and cell_grid[i - 1][j] == "spanned":
                above_bbox = spanning_cells[0]["bbox"]
                if _iou((x0, y0, x1, y1), above_bbox) > 0.1:
                    for sc_idx, sc in enumerate(spanning_cells):
                        if _iou((x0, y0, x1, y1), sc["bbox"]) > 0.1:
                            cell_grid[i][j] = "spanned"
                            row_bboxes.append(None)
                            break
                    if cell_grid[i][j] == "spanned":
                        continue

            cell_bbox = (x0, y0, x1, y1)
            row_bboxes.append(cell_bbox)
        cell_bboxes.append(row_bboxes)

    return cell_bboxes, spanning_cells


def extract_table_from_native(image_path: str, bbox: tuple, pdf_page=None, pdf_bbox=None) -> dict:
    try:
        feature_extractor, model = _get_table_transformer()
    except Exception as e:
        logger.error(f"TableTransformer not available: {e}, using fallback OCR")
        return extract_table_from_scanned(image_path, bbox)

    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        if bbox is not None:
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(width, x1)
            y1 = min(height, y1)
            if x1 <= x0 or y1 <= y0:
                return {"html": "", "markdown": "", "rows": 0, "cols": 0}
            img = img.crop((x0, y0, x1, y1))

        inputs = feature_extractor(images=img, return_tensors="pt")
        outputs = model(**inputs)

        rows, columns, column_headers, spanning_cells = _process_table_structure(
            outputs, img.width, img.height
        )

        if not rows or not columns:
            return extract_table_from_scanned(image_path, bbox)

        cell_bboxes, _ = _build_table_cells(rows, columns, spanning_cells)

        header_rows = set()
        for ch in column_headers:
            for i, row in enumerate(rows):
                if row["bbox"][1] <= ch["bbox"][1] <= row["bbox"][3]:
                    header_rows.add(i)
                    break

        cell_texts = []
        for i, row_bboxes in enumerate(cell_bboxes):
            row_texts = []
            for j, cell_bbox in enumerate(row_bboxes):
                if cell_bbox is None:
                    row_texts.append(None)
                    continue

                cx0, cy0, cx1, cy1 = cell_bbox
                if bbox is not None:
                    cx0 += x0
                    cy0 += y0
                    cx1 += x0
                    cy1 += y0

                text = ""
                if pdf_page is not None and pdf_bbox is not None:
                    scale_x = (pdf_bbox[2] - pdf_bbox[0]) / (bbox[2] - bbox[0])
                    scale_y = (pdf_bbox[3] - pdf_bbox[1]) / (bbox[3] - bbox[1])
                    px0 = pdf_bbox[0] + (cx0 - bbox[0]) * scale_x
                    py0 = pdf_bbox[1] + (cy0 - bbox[1]) * scale_y
                    px1 = pdf_bbox[0] + (cx1 - bbox[0]) * scale_x
                    py1 = pdf_bbox[1] + (cy1 - bbox[1]) * scale_y

                    try:
                        import fitz
                        rect = fitz.Rect(px0, py0, px1, py1)
                        text = pdf_page.get_text("text", clip=rect).strip()
                    except Exception:
                        text = ""

                if not text:
                    text = ocr_region(image_path, (cx0, cy0, cx1, cy1))

                row_texts.append(text)
            cell_texts.append(row_texts)

        md_lines = []
        html_parts = ["<table>"]

        for i, row_texts in enumerate(cell_texts):
            is_header = i in header_rows
            tag = "th" if is_header else "td"

            cells = []
            md_cells = []

            for j, text in enumerate(row_texts):
                if text is None:
                    continue

                text = text.strip() or ""
                md_cells.append(text)

                span_attr = ""
                if i < len(cell_texts) - 1 and j < len(row_texts) - 1:
                    if cell_texts[i + 1][j] is None:
                        span_attr += ' rowspan="2"'
                    if row_texts[j + 1] is None:
                        span_attr += ' colspan="2"'

                cells.append(f"<{tag}{span_attr}>{text}</{tag}>")

            if cells:
                html_parts.append("<tr>" + "".join(cells) + "</tr>")
            if md_cells:
                md_lines.append("| " + " | ".join(md_cells) + " |")
                if is_header and i == len(header_rows) - 1:
                    md_lines.append("| " + " | ".join(["---"] * len(md_cells)) + " |")

        html_parts.append("</table>")
        html_text = "".join(html_parts)
        markdown = "\n".join(md_lines)

        return {
            "html": html_text,
            "markdown": markdown,
            "rows": len(cell_texts),
            "cols": len(columns),
        }

    except Exception as e:
        logger.error(f"TableTransformer extraction failed: {e}")
        return extract_table_from_scanned(image_path, bbox)


def _markdown_table_to_html(md: str) -> str:
    lines = md.strip().split("\n")
    if len(lines) < 2:
        return ""

    html_parts = ["<table>"]

    header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
    html_parts.append("<tr>")
    for cell in header_cells:
        html_parts.append(f"<th>{cell}</th>")
    html_parts.append("</tr>")

    for line in lines[2:]:
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if cells:
            html_parts.append("<tr>")
            for cell in cells:
                html_parts.append(f"<td>{cell}</td>")
            html_parts.append("</tr>")

    html_parts.append("</table>")
    return "".join(html_parts)


def extract_table_from_scanned(image_path: str, bbox: tuple) -> dict:
    text = ocr_region(image_path, bbox)
    if not text.strip():
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    lines = text.split("\n")
    md_lines = []
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.split() if c.strip()]
        if cells:
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")

    markdown = "\n".join(md_lines)
    return {
        "html": _markdown_table_to_html(markdown),
        "markdown": markdown,
        "rows": len(md_lines) // 2 + 1,
        "cols": len(md_lines[0].split("|")) - 2 if md_lines else 0,
    }
