import logging
import asyncio
import fitz
from pathlib import Path
from backend import database as db
from backend.services.pdf_service import validate_pdf, prepare_pages, extract_text_in_region
from backend.services.layout_service import detect_layout
from backend.services.order_service import assign_reading_order
from backend.services.ocr_service import ocr_region, ocr_formula
from backend.services.table_service import extract_table_from_native, extract_table_from_scanned
from backend.services.picture_service import extract_picture

logger = logging.getLogger(__name__)

TEXT_TYPES = {"Caption", "Footnote", "List-item", "Page-footer", "Page-header",
              "Section-header", "Text", "Title"}


async def process_upload(file_path: str, original_filename: str) -> dict:
    validation = validate_pdf(file_path)
    if not validation["valid"]:
        return {"error": validation["error"], "encrypted": validation["encrypted"]}

    if validation["encrypted"]:
        return {"error": "PDF is encrypted", "encrypted": True}

    import os
    file_size = os.path.getsize(file_path)
    filename = Path(file_path).name

    doc_id = await db.create_document(filename, original_filename, file_path, file_size)
    await db.update_document(doc_id, page_count=validation["page_count"], status="validated")

    return {"document_id": doc_id, "page_count": validation["page_count"]}


async def process_document(doc_id: int):
    doc_info = await db.get_document(doc_id)
    if not doc_info:
        logger.error(f"Document {doc_id} not found")
        return

    await db.update_document(doc_id, status="processing")

    try:
        file_path = doc_info["file_path"]
        doc_dir = str(Path(file_path).parent / Path(file_path).stem)
        pages_info = await asyncio.to_thread(prepare_pages, file_path, doc_dir)

        for page_info in pages_info:
            page_id = await db.create_page(
                doc_id,
                page_info["page_number"],
                page_info["width"],
                page_info["height"],
                page_info["jpg_path"],
                page_info["single_pdf_path"],
            )
            await db.update_page(page_id, is_scanned=1 if page_info["is_scanned"] else 0)

        await db.update_document(doc_id, status="pages_ready")

        pages = await db.get_pages(doc_id)
        for page in pages:
            try:
                await _parse_page(doc_id, page, doc_dir)
            except Exception as e:
                logger.error(f"Failed to parse page {page['page_number']}: {e}")
                await db.update_page(page["id"], status="failed", error_message=str(e))

        await db.update_document(doc_id, status="completed")

    except Exception as e:
        logger.error(f"Failed to process document {doc_id}: {e}")
        await db.update_document(doc_id, status="failed", error_message=str(e))


async def _parse_page(doc_id: int, page_info: dict, doc_dir: str):
    page_id = page_info["id"]
    jpg_path = page_info["jpg_path"]
    single_pdf_path = page_info["single_pdf_path"]
    is_scanned = page_info["is_scanned"]

    await db.update_page(page_id, status="parsing_layout")

    elements = await asyncio.to_thread(detect_layout, jpg_path)
    logger.info(f"Page {page_info['page_number']}: detected {len(elements)} elements")

    elements = await asyncio.to_thread(assign_reading_order, elements, jpg_path)
    logger.info(f"Page {page_info['page_number']}: reading order assigned")

    await db.update_page(page_id, status="parsing_content")

    pdf_doc = fitz.open(single_pdf_path)
    pdf_page = pdf_doc[0]

    output_dir = str(Path(doc_dir) / "output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for elem in elements:
        elem_type = elem["element_type"]
        bbox = elem["bbox"]
        confidence = elem["confidence"]
        reading_order = elem["reading_order"]
        content = ""
        content_format = "markdown"

        try:
            if elem_type in TEXT_TYPES:
                if is_scanned:
                    content = await asyncio.to_thread(ocr_region, jpg_path, bbox)
                    content_format = "markdown"
                else:
                    content = await asyncio.to_thread(extract_text_in_region, pdf_page, bbox)
                    content_format = "markdown"

            elif elem_type == "Formula":
                content = await asyncio.to_thread(ocr_formula, jpg_path, bbox)
                content_format = "latex"

            elif elem_type == "Picture":
                result = await asyncio.to_thread(
                    extract_picture, pdf_page, bbox, output_dir, page_id * 1000 + reading_order
                )
                content = result.get("image_path", "")
                content_format = "image_path"

            elif elem_type == "Table":
                if is_scanned:
                    result = await asyncio.to_thread(extract_table_from_scanned, jpg_path, bbox)
                else:
                    result = await asyncio.to_thread(extract_table_from_native, pdf_page, bbox)
                content = result.get("markdown", "") or result.get("html", "")
                content_format = "markdown" if result.get("markdown") else "html"

            await db.create_element(
                page_id, elem_type, bbox, confidence, reading_order,
                content=content, content_format=content_format
            )

        except Exception as e:
            logger.error(f"Failed to parse element {elem_type} at {bbox}: {e}")
            await db.create_element(
                page_id, elem_type, bbox, confidence, reading_order,
                content=f"[ERROR: {str(e)}]", content_format="error"
            )

    pdf_doc.close()
    await db.update_page(page_id, status="completed")
    logger.info(f"Page {page_info['page_number']}: parsing completed")


async def get_parse_results(doc_id: int) -> dict:
    doc = await db.get_document(doc_id)
    if not doc:
        return {"error": "Document not found"}

    pages = await db.get_pages(doc_id)
    result_pages = []

    for page in pages:
        elements = await db.get_elements(page["id"])
        page_data = {
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "is_scanned": bool(page["is_scanned"]),
            "status": page["status"],
            "elements": [],
        }

        for elem in elements:
            page_data["elements"].append({
                "type": elem["element_type"],
                "bbox": [elem["bbox_x0"], elem["bbox_y0"], elem["bbox_x1"], elem["bbox_y1"]],
                "confidence": elem["confidence"],
                "reading_order": elem["reading_order"],
                "content": elem["content"],
                "content_format": elem["content_format"],
            })

        result_pages.append(page_data)

    markdown = _build_markdown(result_pages)

    return {
        "document": {
            "id": doc["id"],
            "original_filename": doc["original_filename"],
            "page_count": doc["page_count"],
            "status": doc["status"],
        },
        "pages": result_pages,
        "markdown": markdown,
    }


def _build_markdown(pages: list[dict]) -> str:
    parts = []

    for page in pages:
        parts.append(f"\n---\n**Page {page['page_number']}**\n")

        elements = sorted(page["elements"], key=lambda e: e["reading_order"])

        for elem in elements:
            etype = elem["type"]
            content = elem.get("content", "") or ""

            if etype == "Title":
                parts.append(f"# {content}\n")
            elif etype == "Section-header":
                parts.append(f"## {content}\n")
            elif etype in TEXT_TYPES:
                if content.strip():
                    parts.append(f"{content}\n")
            elif etype == "Formula":
                parts.append(f"\n{content}\n")
            elif etype == "Table":
                parts.append(f"\n{content}\n")
            elif etype == "Picture":
                if content:
                    parts.append(f"\n![Picture]({content})\n")
            elif etype == "Caption":
                parts.append(f"*{content}*\n")

    return "\n".join(parts)
