import requests

r = requests.get('http://localhost:8001/api/results/11')
d = r.json()

for p in d['pages']:
    for e in p['elements']:
        if e['element_type'] == 'Table':
            content = e.get('content', '') or ''
            cpg = e.get('cross_page_group')
            first_tr = content[content.find('<tr'):content.find('</tr>')] if '<tr' in content else 'N/A'
            print(f"Page {p['page_number']} id={e['id']} cpg={cpg}")
            print(f"  First TR: {first_tr[:200]}")
