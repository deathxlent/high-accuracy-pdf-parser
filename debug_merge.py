import requests, json

r = requests.get('http://localhost:8001/api/results/9')
d = r.json()

from backend.api.routes import _merge_cross_page_tables, _extract_table_rows
pages = d['pages']

group_tables = {}
for page in pages:
    for elem in page["elements"]:
        cpg = elem.get("cross_page_group")
        if cpg is not None and elem["element_type"] == "Table":
            if cpg not in group_tables:
                group_tables[cpg] = []
            group_tables[cpg].append(elem)

for gid, tables in group_tables.items():
    print(f"Group {gid}: {len(tables)} tables")
    for t in tables:
        content = t.get("content", "") or ""
        rows = _extract_table_rows(content)
        print(f"  Table id={t['id']} rows={len(rows)} content_len={len(content)}")
        if len(content) > 0:
            print(f"  First 200 chars: {content[:200]}")

merged = _merge_cross_page_tables(pages)
for p in merged:
    for e in p['elements']:
        if e['element_type'] == 'Table' and e.get('cross_page_group') == 1:
            content = e.get("content", "") or ""
            rows = _extract_table_rows(content)
            print(f"After merge - Page {p['page_number']} Table id={e['id']} rows={len(rows)} content_len={len(content)}")
