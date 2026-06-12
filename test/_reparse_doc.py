import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3
from backend.config import DB_PATH
from backend import database as db
from backend.services.parse_service import process_document

async def reparse_document(doc_id: int):
    await db.init_db()

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    doc = cur.execute("SELECT * FROM pdf_documents WHERE id = ?", (doc_id,)).fetchone()
    if not doc:
        print(f"Document {doc_id} not found!")
        return

    print(f"Document {doc_id}: {doc[2]}")
    print(f"File path: {doc[3]}")
    print(f"Status: {doc[7]}")

    pages = cur.execute("SELECT * FROM pdf_pages WHERE document_id = ? ORDER BY page_number", (doc_id,)).fetchall()
    print(f"\nPages ({len(pages)}):")
    for p in pages:
        print(f"  Page {p[2]}: id={p[0]}, scanned={p[8]}, status={p[11]}, jpg={p[9]}")

    print(f"\nDeleting old elements for document {doc_id}...")
    cur.execute("DELETE FROM page_elements WHERE page_id IN (SELECT id FROM pdf_pages WHERE document_id = ?)", (doc_id,))
    conn.commit()

    print(f"Resetting page statuses...")
    cur.execute("UPDATE pdf_pages SET status = 'pending', error_message = NULL WHERE document_id = ?", (doc_id,))
    conn.commit()

    print(f"Resetting document status...")
    cur.execute("UPDATE pdf_documents SET status = 'pages_ready', error_message = NULL WHERE id = ?", (doc_id,))
    conn.commit()

    conn.close()

    print(f"\nStarting re-parse of document {doc_id}...")
    await process_document(doc_id)

    print(f"\nReparse complete!")

if __name__ == "__main__":
    doc_id = 11
    if len(sys.argv) > 1:
        doc_id = int(sys.argv[1])
    asyncio.run(reparse_document(doc_id))
