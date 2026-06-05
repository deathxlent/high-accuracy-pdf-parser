import os
import uuid
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.config import TMP_DIR
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
            "page_number": p["page_number"],
            "status": p["status"],
            "is_scanned": bool(p["is_scanned"]),
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
