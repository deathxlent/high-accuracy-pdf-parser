import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

doc = cur.execute("SELECT * FROM pdf_documents WHERE id = 10").fetchone()
if doc:
    print(f"Document 10:")
    print(f"  Filename: {doc[2]}")
    print(f"  Original: {doc[1]}")
    print(f"  File path: {doc[3]}")
    print(f"  Status: {doc[7]}")
    print(f"  Error: {doc[8]}")
    
    pages = cur.execute("""
        SELECT id, page_number, is_scanned, jpg_path, status, error_message
        FROM pdf_pages WHERE document_id = 10 ORDER BY page_number
    """).fetchall()
    print(f"\n  Pages ({len(pages)}):")
    for p in pages:
        pid, pnum, is_scanned, jpg_path, status, err = p
        elems = cur.execute("SELECT COUNT(*) FROM page_elements WHERE page_id = ?", (pid,)).fetchone()[0]
        elem_types = cur.execute("""
            SELECT element_type, COUNT(*) 
            FROM page_elements 
            WHERE page_id = ? 
            GROUP BY element_type 
            ORDER BY element_type
        """, (pid,)).fetchall()
        print(f"    Page {pnum} (id={pid}): scanned={is_scanned}, status={status}, elems={elems}")
        print(f"      Types: {elem_types}")

conn.close()
