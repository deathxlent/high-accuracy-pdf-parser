import os
import uuid
import asyncio
import aiosqlite
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from backend.config import TMP_DIR, DB_PATH
from backend import database as db
from backend.services.parse_service import process_upload, process_document, get_parse_results

router = APIRouter(prefix="/api")

_processing_tasks: dict[int, asyncio.Task] = {}


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    ext = Path(file.filename).suffix
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = str(TMP_DIR / unique_name)

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    result = await process_upload(save_path, file.filename)

    if "error" in result:
        try:
            os.unlink(save_path)
        except OSError:
            pass
        if result.get("encrypted"):
            raise HTTPException(status_code=422, detail=result["error"])
        raise HTTPException(status_code=422, detail=result["error"])

    return {"document_id": result["document_id"], "page_count": result["page_count"]}


@router.post("/parse/{doc_id}")
async def parse_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc["status"] == "processing":
        return {"message": "Already processing", "document_id": doc_id}

    if doc["status"] == "completed":
        return {"message": "Already completed", "document_id": doc_id}

    task = asyncio.create_task(process_document(doc_id))
    _processing_tasks[doc_id] = task

    return {"message": "Parsing started", "document_id": doc_id}


@router.get("/status/{doc_id}")
async def get_status(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = await db.get_pages(doc_id)
    page_statuses = [
        {
            "id": p["id"],
            "page_number": p["page_number"],
            "width": p["width"],
            "height": p["height"],
            "jpg_width": p["jpg_width"],
            "jpg_height": p["jpg_height"],
            "status": p["status"],
            "is_scanned": bool(p["is_scanned"]),
            "jpg_path": p["jpg_path"],
            "single_pdf_path": p["single_pdf_path"],
        }
        for p in pages
    ]

    return {
        "document_id": doc_id,
        "status": doc["status"],
        "page_count": doc["page_count"],
        "pages": page_statuses,
    }


@router.get("/results/{doc_id}")
async def results(doc_id: int):
    result = await get_parse_results(doc_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/documents")
async def list_docs():
    docs = await db.list_documents()
    return {"documents": docs}


@router.delete("/documents/{doc_id}")
async def delete_doc(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    import shutil
    try:
        doc_dir = Path(doc["file_path"]).parent / Path(doc["file_path"]).stem
        if doc_dir.exists():
            shutil.rmtree(str(doc_dir), ignore_errors=True)
        if Path(doc["file_path"]).exists():
            os.unlink(doc["file_path"])
    except OSError:
        pass

    await db.delete_document(doc_id)
    return {"message": "Deleted", "document_id": doc_id}


@router.post("/reparse/{doc_id}")
async def reparse_document(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if doc["status"] == "processing":
        return {"message": "Already processing", "document_id": doc_id}

    await db.execute_query("DELETE FROM page_elements WHERE page_id IN (SELECT id FROM pdf_pages WHERE document_id = ?)", (doc_id,))
    await db.execute_query("DELETE FROM pdf_pages WHERE document_id = ?", (doc_id,))

    await db.update_document(doc_id, status="uploaded", error_message=None)

    task = asyncio.create_task(process_document(doc_id))
    _processing_tasks[doc_id] = task

    return {"message": "Reparsing started", "document_id": doc_id}


@router.put("/elements/{element_id}")
async def update_element(element_id: int, data: dict):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM page_elements WHERE id = ?", (element_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Element not found")

        updates = {}
        if "content" in data:
            updates["content"] = data["content"]
        if "reading_order" in data:
            updates["reading_order"] = data["reading_order"]
        if "element_type" in data:
            updates["element_type"] = data["element_type"]

        if updates:
            sets = ", ".join(f"{k} = ?" for k in updates)
            vals = list(updates.values()) + [element_id]
            await db.execute(f"UPDATE page_elements SET {sets} WHERE id = ?", vals)
            await db.commit()

        cursor = await db.execute("SELECT * FROM page_elements WHERE id = ?", (element_id,))
        row = await cursor.fetchone()
        return dict(row)


@router.put("/pages/{page_id}/elements/reorder")
async def reorder_elements(page_id: int, data: dict):
    element_order = data.get("element_order", [])
    if not element_order:
        raise HTTPException(status_code=400, detail="element_order is required")

    async with aiosqlite.connect(str(DB_PATH)) as db:
        for idx, elem_id in enumerate(element_order):
            await db.execute(
                "UPDATE page_elements SET reading_order = ? WHERE id = ? AND page_id = ?",
                (idx, elem_id, page_id)
            )
        await db.commit()

    return {"message": "Elements reordered", "page_id": page_id}


@router.get("/pages/{page_id}/elements")
async def get_page_elements(page_id: int):
    elements = await db.get_elements(page_id)
    return {"elements": elements}


@router.get("/documents/{doc_id}/thumbnail")
async def get_document_thumbnail(doc_id: int):
    doc = await db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pages = await db.get_pages(doc_id)
    if pages and pages[0]["jpg_path"]:
        jpg_path = pages[0]["jpg_path"]
        if os.path.exists(jpg_path):
            return FileResponse(jpg_path)

    return {"error": "No thumbnail available"}, 404


@router.get("/pages/{page_id}/pdf")
async def get_page_pdf(page_id: int):
    async with aiosqlite.connect(str(DB_PATH)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM pdf_pages WHERE id = ?", (page_id,))
        row = await cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Page not found")

        if row["single_pdf_path"] and os.path.exists(row["single_pdf_path"]):
            return FileResponse(row["single_pdf_path"])

    return {"error": "No single PDF available"}, 404



