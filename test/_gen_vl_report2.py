import re

# Read the actual llama-cli output with utf-8
with open(r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\llama_cli_output.txt', encoding='utf-8', errors='replace') as f:
    raw = f.read()

# Extract OCR text: everything between "> OCR:" and the performance line
ocr_match = re.search(r'>\s*OCR:\s*\n(.+?)(?:\s*\[ Prompt:)', raw, re.DOTALL)
if ocr_match:
    ocr_text = ocr_match.group(1).strip()
else:
    # Fallback: everything after "> OCR:" and before the last ">"
    parts = raw.split('> OCR:')
    if len(parts) > 1:
        ocr_text = parts[1].strip()
        # Remove trailing prompt line
        ocr_text = re.sub(r'\s*\[ Prompt:.*', '', ocr_text).strip()
        ocr_text = re.sub(r'\n> \nExiting\.\.\..*', '', ocr_text).strip()
    else:
        ocr_text = raw

# Extract performance
perf_match = re.search(r'\[ Prompt:\s*([\d.]+)\s*t/s \| Generation:\s*([\d.]+)\s*t/s', raw)
prompt_speed = perf_match.group(1) if perf_match else 'N/A'
gen_speed = perf_match.group(2) if perf_match else 'N/A'

# Escape HTML
ocr_text_escaped = ocr_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaddleOCR-VL-1.6 GGUF Test Report</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif; background: #f0f2f5; color: #333; }}
.header {{ background: linear-gradient(135deg, #7c4dff, #4a148c); color: white; padding: 40px 0; text-align: center; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header p {{ opacity: 0.85; font-size: 14px; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 32px; }}
.card {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; }}
.card .num {{ font-size: 32px; font-weight: 700; color: #7c4dff; }}
.card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
.card .sub {{ font-size: 11px; color: #999; margin-top: 2px; }}
.section {{ background: white; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.section h2 {{ font-size: 18px; margin-bottom: 16px; color: #7c4dff; border-left: 4px solid #7c4dff; padding-left: 12px; }}
pre {{ background: #1e1e1e; color: #d4d4d4; padding: 20px; border-radius: 8px; font-size: 13px; line-height: 1.7; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; font-family: 'Cascadia Code', 'Fira Code', 'Source Han Mono', monospace; }}
.info-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
.info-grid .item {{ padding: 6px 0; border-bottom: 1px solid #f0f0f0; }}
.info-grid .item .k {{ color: #888; font-size: 12px; }}
.info-grid .item .v {{ font-size: 14px; }}
.compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.compare-box {{ padding: 16px; border-radius: 8px; }}
.compare-box.gpu {{ background: #e8f0fe; border: 1px solid #c5d9f7; }}
.compare-box.vl {{ background: #f3e8ff; border: 1px solid #dcc7ff; }}
.compare-box h3 {{ font-size: 14px; margin-bottom: 8px; }}
.compare-box ul {{ padding-left: 20px; font-size: 13px; line-height: 1.8; }}
</style>
</head>
<body>
<div class="header">
    <h1>PaddleOCR-VL-1.6 GGUF 测试报告</h1>
    <p>llama.cpp &middot; GGUF (Q4_K_M 300MB + mmproj 882MB) &middot; GPU offload &middot; 文档解析</p>
</div>
<div class="container">

    <div class="cards">
        <div class="card">
            <div class="num">{prompt_speed}</div>
            <div class="label">提示处理速度</div>
            <div class="sub">tokens/s</div>
        </div>
        <div class="card">
            <div class="num">{gen_speed}</div>
            <div class="label">生成速度</div>
            <div class="sub">tokens/s</div>
        </div>
        <div class="card">
            <div class="num">0.5B</div>
            <div class="label">模型参数量</div>
            <div class="sub">LLM 主干</div>
        </div>
        <div class="card">
            <div class="num">1.18 GB</div>
            <div class="label">总模型大小</div>
            <div class="sub">GGUF + mmproj</div>
        </div>
    </div>

    <div class="section">
        <h2>模型配置</h2>
        <div class="info-grid">
            <div class="item"><div class="k">LLM 主干</div><div class="v">PaddleOCR-VL-1.6.Q4_K_M.gguf (300 MB)</div></div>
            <div class="item"><div class="k">视觉编码器</div><div class="v">PaddleOCR-VL-1.6-GGUF-mmproj.gguf (882 MB)</div></div>
            <div class="item"><div class="k">运行工具</div><div class="v">llama-cli (b9571)</div></div>
            <div class="item"><div class="k">提示词</div><div class="v">OCR:</div></div>
            <div class="item"><div class="k">温度</div><div class="v">0 (确定性输出)</div></div>
            <div class="item"><div class="k">GPU</div><div class="v">NVIDIA Quadro P1000 (4GB VRAM)</div></div>
        </div>
    </div>

    <div class="section">
        <h2>解析输出全文</h2>
        <pre>{ocr_text_escaped}</pre>
    </div>

    <div class="section">
        <h2>PP-OCRv5 vs PaddleOCR-VL 对比</h2>
        <div class="compare">
            <div class="compare-box gpu">
                <h3>PP-OCRv5 (传统 OCR)</h3>
                <ul>
                    <li>检测 + 识别两阶段流水线</li>
                    <li>输出 92 个独立文本块</li>
                    <li>每块带有置信度分数</li>
                    <li>平均置信度 0.9948</li>
                    <li>适合纯文本提取任务</li>
                </ul>
            </div>
            <div class="compare-box vl">
                <h3>PaddleOCR-VL (视觉语言模型)</h3>
                <ul>
                    <li>端到端文档理解</li>
                    <li>输出结构化段落/表格</li>
                    <li>理解文档上下文语义</li>
                    <li>速度 {prompt_speed}/{gen_speed} t/s</li>
                    <li>适合复杂文档解析</li>
                </ul>
            </div>
        </div>
    </div>

</div>
</body>
</html>'''

path = r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\report.html'
with open(path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'OK: {path} ({len(html)} bytes)')
print(f'OCR text length: {len(ocr_text)} chars')
print(f'Prompt speed: {prompt_speed} t/s')
print(f'Gen speed: {gen_speed} t/s')
