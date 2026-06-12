import urllib.request
import json

# Test server health
url = 'http://127.0.0.1:8080/health'
try:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('Health check:', resp.status)
        print(resp.read().decode('utf-8'))
except Exception as e:
    print('Health check error:', e)

# Get model info
url2 = 'http://127.0.0.1:8080/v1/models'
try:
    req = urllib.request.Request(url2)
    with urllib.request.urlopen(req, timeout=10) as resp:
        print('\nModels:', resp.status)
        data = json.loads(resp.read().decode('utf-8'))
        print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception as e:
    print('Models error:', e)
