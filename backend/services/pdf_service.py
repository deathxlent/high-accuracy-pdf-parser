import fitz
import os
from pathlib import Path
from backend.config import TMP_DIR, SCAN_TEXT_THRESHOLD


def validate_pdf(file_path: str) -> dict:
    result = {"valid": False, "encrypted": False, "page_count": 0, "error": None}
    try:
        doc = fitz.open(file_path)
        if doc.is_encrypted:
            try:
                doc.authenticate("")
            except Exception:
                pass
            if doc.is_encrypted:
                result["encrypted"] = True
                result["error"] = "PDF is encrypted and cannot be opened"
                doc.close()
                return result
        result["page_count"] = len(doc)
        result["valid"] = True
        doc.close()
    except fitz.FileDataError as e:
        result["error"] = f"Invalid PDF file: {e}"
    except Exception as e:
        result["error"] = f"Error opening PDF: {e}"
    return result


def is_page_scanned(page: fitz.Page) -> bool:
    text = page.get_text("text").strip()
    if len(text) < SCAN_TEXT_THRESHOLD:
        return True
    return False


def convert_page_to_jpg(page: fitz.Page, output_path: str, dpi: int = 200) -> str:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_path)
    return output_path


def save_single_page_pdf(doc: fitz.Document, page_idx: int, output_path: str) -> str:
    single_doc = fitz.open()
    single_doc.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
    single_doc.save(output_path)
    single_doc.close()
    return output_path


def extract_text_in_region(page: fitz.Page, bbox: tuple[float, float, float, float]) -> str:
    rect = fitz.Rect(bbox)
    if rect.is_empty or not rect.is_valid:
        return ""
    text_dict = page.get_text("dict", clip=rect)
    lines = []
    for block in text_dict.get("blocks", []):
        if block["type"] == 0:
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                if line_text.strip():
                    lines.append(line_text.strip())
    return "\n".join(lines)


def prepare_pages(file_path: str, doc_dir: str) -> list[dict]:
    doc = fitz.open(file_path)
    pages_info = []
    doc_dir_path = Path(doc_dir)
    doc_dir_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(doc)):
        page = doc[i]
        rect = page.rect
        width, height = rect.width, rect.height

        jpg_path = str(doc_dir_path / f"page_{i + 1}.jpg")
        single_pdf_path = str(doc_dir_path / f"page_{i + 1}.pdf")

        convert_page_to_jpg(page, jpg_path)
        save_single_page_pdf(doc, i, single_pdf_path)

        scanned = is_page_scanned(page)

        pages_info.append({
            "page_number": i + 1,
            "width": width,
            "height": height,
            "is_scanned": scanned,
            "jpg_path": jpg_path,
            "single_pdf_path": single_pdf_path,
        })

    doc.close()
    return pages_info


def clip_region_as_image(page: fitz.Page, bbox: tuple[float, float, float, float],
                         output_path: str, dpi: int = 200) -> str:
    rect = fitz.Rect(bbox)
    if rect.is_empty or not rect.is_valid:
        return ""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    pix.save(output_path)
    return output_path
