import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

print("Cleaning up duplicate pages for doc_id=11...")

cur.execute("SELECT id, page_number FROM pdf_pages WHERE document_id = 11 ORDER BY id")
rows = cur.fetchall()
print(f"Found {len(rows)} pages for doc 11: {rows}")

page_numbers_seen = {}
to_delete = []
to_keep = []

for pid, pnum in rows:
    if pnum in page_numbers_seen:
        old_id = page_numbers_seen[pnum]
        to_delete.append(old_id)
        page_numbers_seen[pnum] = pid
    else:
        page_numbers_seen[pnum] = pid

to_keep = list(page_numbers_seen.values())
print(f"Keeping pages: {sorted(page_numbers_seen.items())}")
print(f"Deleting duplicate pages (old): {to_delete}")

for pid in to_delete:
    cur.execute("DELETE FROM page_elements WHERE page_id = ?", (pid,))
    cur.execute("DELETE FROM pdf_pages WHERE id = ?", (pid,))

conn.commit()

print("\nVerifying cleanup:")
cur.execute("SELECT id, page_number, status FROM pdf_pages WHERE document_id = 11 ORDER BY page_number")
for row in cur.fetchall():
    elems = cur.execute("SELECT COUNT(*) FROM page_elements WHERE page_id = ?", (row[0],)).fetchone()[0]
    print(f"  Page {row[1]} (id={row[0]}): status={row[2]}, elements={elems}")

conn.close()
print("\nCleanup complete!")
