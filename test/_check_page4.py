import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

print("=== ID=9 Page 4 (Native) ===")
cur.execute("""
    SELECT e.id, e.element_type, e.reading_order, e.content_format, substr(COALESCE(e.content, ''), 1, 200)
    FROM page_elements e
    JOIN pdf_pages p ON e.page_id = p.id
    WHERE p.document_id = 9 AND p.page_number = 4
    ORDER BY e.reading_order
""")
for r in cur.fetchall():
    content = r[4].replace('\n', '\\n')
    print(f"  #{r[0]} [{r[1]}] order={r[2]} fmt={r[3]}")
    print(f"    Content: {content[:300]}")
    print()

print("\n=== ID=11 Page 4 (Scanned VL) ===")
cur.execute("""
    SELECT e.id, e.element_type, e.reading_order, e.content_format, substr(COALESCE(e.content, ''), 1, 200)
    FROM page_elements e
    JOIN pdf_pages p ON e.page_id = p.id
    WHERE p.document_id = 11 AND p.page_number = 4
    ORDER BY e.reading_order
""")
for r in cur.fetchall():
    content = r[4].replace('\n', '\\n')
    print(f"  #{r[0]} [{r[1]}] order={r[2]} fmt={r[3]}")
    print(f"    Content: {content[:300]}")
    print()

conn.close()
