import logging
import fitz
from backend.services.ocr_service import ocr_region
from backend.services.pdf_service import jpg_bbox_to_pdf_bbox, DEFAULT_DPI

logger = logging.getLogger(__name__)

TABLE_STRATEGY = "lines_strict"
SNAP_TOLERANCE = 5
JOIN_TOLERANCE = 5


def _table_to_html(table) -> str:
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


def extract_table_from_native(pdf_page: fitz.Page, bbox: tuple[float, float, float, float] | None,
                              bbox_is_jpg: bool = True, dpi: int = DEFAULT_DPI) -> dict:
    if bbox is not None and bbox_is_jpg:
        pdf_bbox = jpg_bbox_to_pdf_bbox(bbox, dpi)
        rect = fitz.Rect(pdf_bbox)
    elif bbox is not None:
        rect = fitz.Rect(bbox)
    else:
        rect = pdf_page.rect

    if rect.is_empty or not rect.is_valid:
        return {"markdown": "", "html": "", "rows": 0, "cols": 0}

    try:
        finder = pdf_page.find_tables(
            clip=rect,
            strategy=TABLE_STRATEGY,
            snap_tolerance=SNAP_TOLERANCE,
            join_tolerance=JOIN_TOLERANCE,
        )
    except Exception as e:
        logger.error(f"find_tables failed: {e}")
        return {"markdown": "", "html": "", "rows": 0, "cols": 0}

    if not finder.tables:
        text = pdf_page.get_text("text", clip=rect)
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
            return {
                "markdown": markdown,
                "html": _md_to_html(markdown),
                "rows": len(lines),
                "cols": len(lines[0].split("|")) if lines else 0,
            }
        return {"markdown": "", "html": "", "rows": 0, "cols": 0}

    table = finder.tables[0]
    if len(finder.tables) > 1:
        logger.info(f"Found {len(finder.tables)} tables, using first one")

    try:
        markdown = table.to_markdown()
    except Exception as e:
        logger.warning(f"to_markdown failed: {e}")
        markdown = ""

    try:
        html = _table_to_html(table)
    except Exception as e:
        logger.warning(f"HTML conversion failed: {e}")
        html = ""

    return {
        "markdown": markdown,
        "html": html,
        "rows": table.row_count,
        "cols": table.col_count,
    }


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
