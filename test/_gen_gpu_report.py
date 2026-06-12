import json, base64, os

with open(r'C:\ws\high accuracy pdf parser\test\output_ocr_gpu\page_1_res.json', encoding='utf-8') as f:
    data = json.load(f)

with open(r'C:\ws\high accuracy pdf parser\test\output_ocr_gpu\page_1_ocr_res_img.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

texts = data['rec_texts']
scores = data['rec_scores']
boxes = data['rec_boxes']
angles = data['textline_orientation_angles']

n = len(texts)
avg_conf = sum(scores)/n
max_conf = max(scores)
min_conf = min(scores)
high_conf = sum(1 for s in scores if s > 0.9)
n_ge98 = sum(1 for s in scores if s >= 0.98)
n_mid = sum(1 for s in scores if 0.9 <= s < 0.98)
n_low = sum(1 for s in scores if s < 0.9)
pct_ge98 = n_ge98 / n * 100
pct_mid = n_mid / n * 100
pct_low = n_low / n * 100

rows = []
for i in range(n):
    conf = scores[i]
    cls = 'conf-high' if conf >= 0.98 else ('conf-mid' if conf >= 0.9 else 'conf-low')
    rows.append(
        f'<tr class="{cls}">'
        f'<td>{i}</td>'
        f'<td class="text-col">{texts[i]}</td>'
        f'<td><span class="badge">{conf:.4f}</span></td>'
        f'<td>[{boxes[i][0]},{boxes[i][1]},{boxes[i][2]},{boxes[i][3]}]</td>'
        f'<td>{angles[i]}</td>'
        f'</tr>'
    )
rows_html = '\n'.join(rows)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaddleOCR GPU (PP-OCRv5) Test Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f0f2f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #1a73e8, #0d47a1); color: white; padding: 40px 0; text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header p {{ opacity: 0.85; font-size: 14px; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 36px; font-weight: 700; color: #1a73e8; }}
.card .num.green {{ color: #34a853; }}
.card .num.orange {{ color: #fbbc04; }}
.card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
.card .sub {{ font-size: 11px; color: #999; margin-top: 2px; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; margin-bottom: 16px; color: #1a73e8; border-left: 4px solid #1a73e8; padding-left: 12px; }}
.dist-bar {{ display: flex; height: 32px; border-radius: 6px; overflow: hidden; margin: 12px 0; }}
.dist-bar .seg {{ display: flex; align-items: center; justify-content: center; font-size: 12px; color: white; font-weight: 600; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ background: #f8f9fa; padding: 10px 8px; text-align: left; font-weight: 600; color: #555; border-bottom: 2px solid #e0e0e0; position: sticky; top: 0; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; }}
tr:hover {{ background: #f5f8ff; }}
.text-col {{ max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 12px; font-weight: 600; }}
.conf-high .badge {{ background: #e6f4ea; color: #1e7e34; }}
.conf-mid .badge {{ background: #fef7e0; color: #f9a825; }}
.conf-low .badge {{ background: #fce8e6; color: #d93025; }}
.table-wrap {{ max-height: 600px; overflow-y: auto; border: 1px solid #e0e0e0; border-radius: 8px; }}
.img-section {{ text-align: center; }}
.img-section img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,0.1); }}
.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.info-grid .item {{ padding: 6px 0; border-bottom: 1px solid #f0f0f0; }}
.info-grid .item .k {{ color: #888; font-size: 12px; }}
.info-grid .item .v {{ font-size: 14px; }}
</style>
</head>
<body>
<div class="header">
    <h1>PaddleOCR GPU (PP-OCRv5) 测试报告</h1>
    <p>GPU: NVIDIA Quadro P1000 &middot; CUDA 11.8 &middot; PaddlePaddle 3.2.1 &middot; PaddleOCR 3.6.0</p>
</div>
<div class="container">

    <div class="cards">
        <div class="card">
            <div class="num green">{n}</div>
            <div class="label">文本块总数</div>
        </div>
        <div class="card">
            <div class="num">{avg_conf:.4f}</div>
            <div class="label">平均置信度</div>
        </div>
        <div class="card">
            <div class="num green">{max_conf:.4f}</div>
            <div class="label">最高置信度</div>
        </div>
        <div class="card">
            <div class="num orange">{min_conf:.4f}</div>
            <div class="label">最低置信度</div>
        </div>
        <div class="card">
            <div class="num green">{high_conf}</div>
            <div class="label">置信度 &gt; 0.9</div>
            <div class="sub">/ {n} 个块</div>
        </div>
        <div class="card">
            <div class="num">{n_low}</div>
            <div class="label">置信度 &lt; 0.9</div>
        </div>
    </div>

    <div class="section">
        <h2>置信度分布</h2>
        <div class="dist-bar">
            <div class="seg" style="width:{pct_ge98:.1f}%;background:#34a853;">{n_ge98}</div>
            <div class="seg" style="width:{pct_mid:.1f}%;background:#fbbc04;">{n_mid}</div>
            <div class="seg" style="width:{pct_low:.1f}%;background:#ea4335;">{n_low}</div>
        </div>
        <div style="display:flex;gap:16px;font-size:12px;color:#666;">
            <span>&ge; 0.98: {n_ge98}</span>
            <span>0.9-0.98: {n_mid}</span>
            <span>&lt; 0.9: {n_low}</span>
        </div>
    </div>

    <div class="section">
        <h2>配置参数</h2>
        <div class="info-grid">
            <div class="item"><div class="k">文档预处理</div><div class="v">禁用</div></div>
            <div class="item"><div class="k">文字方向检测</div><div class="v">启用</div></div>
            <div class="item"><div class="k">检测阈值</div><div class="v">0.3</div></div>
            <div class="item"><div class="k">框阈值</div><div class="v">0.6</div></div>
            <div class="item"><div class="k">Unclip 比例</div><div class="v">1.5</div></div>
            <div class="item"><div class="k">文字类型</div><div class="v">通用</div></div>
        </div>
    </div>

    <div class="section">
        <h2>文本块明细 ({n} 项)</h2>
        <div class="table-wrap">
        <table>
            <thead><tr><th>#</th><th>文本</th><th>置信度</th><th>边界框</th><th>角度</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
    </div>

    <div class="section img-section">
        <h2>OCR 可视化</h2>
        <img src="data:image/jpeg;base64,{img_b64}" alt="OCR Result Visualization">
    </div>

</div>
</body>
</html>'''

path = r'C:\ws\high accuracy pdf parser\test\output_ocr_gpu\report.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'OK: {path} ({os.path.getsize(path)} bytes)')
