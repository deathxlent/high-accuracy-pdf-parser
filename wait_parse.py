import time, requests, sys
doc_id = int(sys.argv[1])
for i in range(120):
    r = requests.get(f'http://localhost:8001/api/progress/{doc_id}')
    d = r.json()
    status = d.get('status', '')
    progress = d.get('progress', {})
    stage = progress.get('stage', '')
    pct = progress.get('percent', 0)
    msg = progress.get('message', '')
    print(f'[{i}] status={status} stage={stage} pct={pct} msg={msg}')
    if status == 'completed' and stage != 'parsing_content':
        break
    if status == 'failed':
        break
    time.sleep(5)
