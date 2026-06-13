import asyncio
import sys
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

import sqlite3
from backend import database as db
from backend.services.parse_service import process_document
from backend.config import DB_PATH

async def reparse_document_native(doc_id: int):
    await db.init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    doc = cur.execute("SELECT * FROM pdf_documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        print(f"Document {doc_id} not found!")
        return

    print(f"=== Document {doc_id} ===")
    print(f"Filename: {doc[2]}")
    print(f"Original: {doc[1]}")
    print(f"File path: {doc[3]}")
    print(f"Status: {doc[7]}")

    pages = cur.execute("""
        SELECT id, page_number, is_scanned, jpg_path, status 
        FROM pdf_pages WHERE document_id = ? ORDER BY page_number
    """, (doc_id,)).fetchall()
    print(f"\nPages ({len(pages)}):")
    for p in pages:
        print(f"  Page {p[1]} (id={p[0]}): scanned={p[2]}, status={p[3]}")

    print(f"\nDeleting old elements for document {doc_id}...")
    cur.execute("DELETE FROM page_elements WHERE page_id IN (SELECT id FROM pdf_pages WHERE document_id = ?)", (doc_id,))
    conn.commit()

    print(f"Resetting page statuses to 'pending'...")
    cur.execute("UPDATE pdf_pages SET status = 'pending', error_message = NULL WHERE document_id = ?", (doc_id,))
    conn.commit()

    print(f"Resetting document status to 'pages_ready'...")
    cur.execute("UPDATE pdf_documents SET status = 'pages_ready', error_message = NULL WHERE id = ?", (doc_id,))
    conn.commit()

    conn.close()

    print(f"\nStarting re-parse of document {doc_id} (native PDF, no OCR)...")
    print(f"NOTE: This will run YOLO layout detection (CPU) and Surya reading order (CPU).")
    print(f"      For non-scanned PDFs, text is extracted directly via PyMuPDF, no OCR needed.\n")

    await process_document(doc_id)

    print(f"\nReparse complete!")

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    pages = cur.execute("""
        SELECT id, page_number, is_scanned, status 
        FROM pdf_pages WHERE document_id = ? ORDER BY page_number
    """, (doc_id,)).fetchall()
    print(f"\nFinal status:")
    for p in pages:
        elems = cur.execute("SELECT COUNT(*) FROM page_elements WHERE page_id = ?", (p[0],)).fetchone()[0]
        elem_types = cur.execute("""
            SELECT element_type, COUNT(*) 
            FROM page_elements WHERE page_id = ? 
            GROUP BY element_type ORDER BY element_type
        """, (p[0],)).fetchall()
        print(f"  Page {p[1]} (id={p[0]}): status={p[3]}, elems={elems}")
        print(f"    Types: {elem_types}")

        print(f"    Reading order:")
        elem_order = cur.execute("""
            SELECT reading_order, element_type, substr(COALESCE(content, ''), 1, 80)
            FROM page_elements WHERE page_id = ? 
            ORDER BY reading_order
        """, (p[0],)).fetchall()
        for ro, etype, content in elem_order:
            content_display = content.replace('\n', '\\n')
            print(f"      #{ro:2d} [{etype:15s}]: {content_display}")
        print()

    conn.close()

if __name__ == "__main__":
    doc_id = 10
    if len(sys.argv) > 1:
        doc_id = int(sys.argv[1])
    asyncio.run(reparse_document_native(doc_id))
