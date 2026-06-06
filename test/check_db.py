import sys, sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import DB_PATH

conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()

print("=== PAGES ===")
for row in cur.execute("SELECT * FROM pdf_pages"):
    print(row)

print("\n=== ELEMENTS ===")
for row in cur.execute("SELECT * FROM page_elements ORDER BY reading_order"):
    eid, pid, etype, bx0, by0, bx1, by1, conf, ro, content, cformat, created = row
    print(f"\n--- Elem #{eid} | {etype} | conf={conf:.3f} | order={ro} | fmt={cformat} ---")
    print(f"  bbox: ({bx0:.1f}, {by0:.1f}) - ({bx1:.1f}, {by1:.1f})")
    if content:
        c = content[:500] + "..." if len(content) > 500 else content
        print(f"  content: {c}")
    else:
        print(f"  content: [empty]")

conn.close()
