import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

for doc_id in [9, 11]:
    print(f"\n{'='*60}")
    print(f"DOCUMENT {doc_id}")
    print(f"{'='*60}")
    
    doc = cur.execute("SELECT * FROM pdf_documents WHERE id = ?", (doc_id,)).fetchone()
    if doc:
        print(f"Filename: {doc[2]}")
        print(f"Original: {doc[1]}")
        print(f"File path: {doc[3]}")
        print(f"Status: {doc[7]}")
        print(f"Error: {doc[8]}")

    pages = cur.execute("SELECT id, page_number, is_scanned, jpg_path, status, error_message FROM pdf_pages WHERE document_id = ? ORDER BY page_number", (doc_id,)).fetchall()
    print(f"\n--- Pages ({len(pages)}) ---")
    
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
        
        print(f"\n  Page {pnum} (id={pid}): scanned={is_scanned}, status={status}")
        print(f"    Elements: {elems} total - {[(t, c) for t, c in elem_types]}")
        if err:
            print(f"    Error: {err[:200]}")
        
        for etype, _ in elem_types:
            elems_detail = cur.execute("""
                SELECT id, reading_order, content_format, substr(COALESCE(content, ''), 1, 150)
                FROM page_elements 
                WHERE page_id = ? AND element_type = ?
                ORDER BY reading_order
            """, (pid, etype)).fetchall()
            
            print(f"    --- {etype} ({len(elems_detail)}) ---")
            for eid, ro, cfmt, content_snippet in elems_detail:
                content_display = content_snippet.replace('\n', '\\n')
                print(f"      #{eid} order={ro} fmt={cfmt}: {content_display}")

conn.close()
