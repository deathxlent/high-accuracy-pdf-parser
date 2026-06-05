import logging
import fitz
from pathlib import Path
from backend.config import TABLE_STRATEGY
from backend.services.ocr_service import ocr_region

logger = logging.getLogger(__name__)


def extract_table_from_native(page: fitz.Page, bbox: tuple[float, float, float, float]) -> dict:
    rect = fitz.Rect(bbox)
    if rect.is_empty or not rect.is_valid:
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    try:
        finder = page.find_tables(
            clip=rect,
            strategy=TABLE_STRATEGY,
            snap_tolerance=5,
            join_tolerance=5,
        )

        if not finder.tables:
            return _extract_table_from_text(page, rect)

        table = finder.tables[0]

        try:
            md_text = table.to_markdown()
        except Exception:
            md_text = ""

        try:
            html_text = _table_to_html_with_spans(table)
        except Exception:
            html_text = ""

        row_count = table.row_count if hasattr(table, 'row_count') else 0
        col_count = table.col_count if hasattr(table, 'col_count') else 0

        return {
            "html": html_text,
            "markdown": md_text,
            "rows": row_count,
            "cols": col_count,
        }

    except Exception as e:
        logger.error(f"Native table extraction failed: {e}")
        return _extract_table_from_text(page, rect)


def _table_to_html_with_spans(table) -> str:
    try:
        extract = table.extract()
    except Exception:
        return ""

    if not extract:
        return ""

    header = None
    if hasattr(table, 'header') and table.header:
        try:
            header = table.header
        except Exception:
            header = None

    html_parts = ["<table>"]

    for row_idx, row in enumerate(extract):
        html_parts.append("<tr>")
        for col_idx, cell in enumerate(row):
            if cell is None:
                cell = ""

            is_header_row = header is not None and row_idx == 0
            tag = "th" if is_header_row else "td"

            span_attr = ""
            try:
                if hasattr(table, 'cells') and table.cells:
                    for c in table.cells:
                        if hasattr(c, 'row_id') and hasattr(c, 'col_id'):
                            if c.row_id == row_idx and c.col_id == col_idx:
                                rowspan = getattr(c, 'rowspan', 1)
                                colspan = getattr(c, 'colspan', 1)
                                if rowspan and rowspan > 1:
                                    span_attr += f' rowspan="{rowspan}"'
                                if colspan and colspan > 1:
                                    span_attr += f' colspan="{colspan}"'
                                break
            except Exception:
                pass

            html_parts.append(f"<{tag}{span_attr}>{cell}</{tag}>")
        html_parts.append("</tr>")

    html_parts.append("</table>")
    return "".join(html_parts)


def _extract_table_from_text(page: fitz.Page, rect: fitz.Rect) -> dict:
    text = page.get_text("text", clip=rect)
    if not text.strip():
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return {"html": "", "markdown": "", "rows": 0, "cols": 0}

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
        "html": _markdown_table_to_html(markdown),
        "markdown": markdown,
        "rows": len(lines),
        "cols": len(lines[0].split("|")) if lines else 0,
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


def extract_table_from_scanned(image_path: str, bbox: tuple[float, float, float, float]) -> dict:
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
