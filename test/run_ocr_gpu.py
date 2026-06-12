"""
PaddleOCR GPU Test Script
=========================
Based on: https://www.paddleocr.ai/main/quick_start.html
Uses PaddleOCR 3.x API (PaddleOCR.predict) with GPU backend

Usage:
    python test/run_ocr_gpu.py

Environment:
    - PaddlePaddle GPU 3.2.1 (CUDA 11.8)
    - PaddleOCR 3.6.0
    - Models: PP-OCRv5 (server detection + recognition)
"""

import os
import sys
import time
import json
from pathlib import Path

# ── Configure model cache to avoid Chinese-character path issues ──
CACHE_DIR = Path("C:/paddlex_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(CACHE_DIR)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ── Paths ──
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
IMAGE_PATH = PROJECT_ROOT / "tmp" / "42e59745cdb54b6fb2c635d7c11dbd43" / "page_1.jpg"
OUTPUT_DIR = PROJECT_ROOT / "test" / "output_ocr_gpu"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── GPU Environment Check ──
def check_environment():
    print("=" * 60)
    print("  PaddleOCR GPU Environment Check")
    print("=" * 60)

    try:
        import paddle
        print(f"[OK] PaddlePaddle version: {paddle.__version__}")
        cuda_avail = paddle.device.is_compiled_with_cuda()
        print(f"[OK] CUDA compiled: {cuda_avail}")
        if cuda_avail:
            count = paddle.device.cuda.device_count()
            name = paddle.device.cuda.get_device_name(0) if count > 0 else "N/A"
            cuda_ver = paddle.version.cuda()
            print(f"[OK] GPU count: {count}")
            print(f"[OK] GPU name: {name}")
            print(f"[OK] CUDA version: {cuda_ver}")
    except ImportError:
        print("[ERROR] PaddlePaddle not installed!")
        sys.exit(1)

    try:
        from paddleocr import PaddleOCR
        import paddleocr
        print(f"[OK] PaddleOCR version: {paddleocr.__version__}")
        print(f"[OK] PaddleOCR class methods: predict, ocr available")
    except ImportError:
        print("[ERROR] PaddleOCR not installed!")
        sys.exit(1)

    if not IMAGE_PATH.exists():
        print(f"[ERROR] Test image not found: {IMAGE_PATH}")
        sys.exit(1)
    img_size = IMAGE_PATH.stat().st_size
    print(f"[OK] Test image: {IMAGE_PATH.name} ({img_size / 1024:.0f} KB)")

    return True


# ── Main OCR Processing ──
def run_ocr():
    from paddleocr import PaddleOCR

    print("\n" + "=" * 60)
    print("  PaddleOCR GPU Inference (PP-OCRv5)")
    print("=" * 60)

    # Step 1: Initialize (downloads models on first run)
    print("\n[1/3] Initializing PaddleOCR pipeline...")
    t0 = time.time()
    ocr = PaddleOCR(
        lang="ch",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    t_init = time.time() - t0
    print(f"       Init time: {t_init:.1f}s")

    # Step 2: Run prediction
    print(f"\n[2/3] Running OCR on: {IMAGE_PATH.name}")
    t1 = time.time()
    results = ocr.predict(str(IMAGE_PATH))
    t_ocr = time.time() - t1
    print(f"       OCR time: {t_ocr:.1f}s")

    # Step 3: Save and analyze results
    print(f"\n[3/3] Saving results...")
    for i, res in enumerate(results):
        # Save JSON
        json_path = str(OUTPUT_DIR / f"page_1_res.json")
        res.save_to_json(save_path=str(OUTPUT_DIR))
        print(f"       JSON saved: {json_path}")

        # Save visualization image
        img_path = str(OUTPUT_DIR / f"page_1_ocr_res_img.jpg")
        res.save_to_img(save_path=str(OUTPUT_DIR))
        print(f"       Image saved: {img_path}")

        # Extract text data
        raw = res.json if hasattr(res, "json") else {}
        if isinstance(raw, dict):
            rec_texts = raw.get("rec_texts", [])
            rec_scores = raw.get("rec_scores", [])
        else:
            rec_texts = raw.get("res", {}).get("rec_texts", [])
            rec_scores = raw.get("res", {}).get("rec_scores", [])

        total_time = time.time() - t0

        # Collect results for report
        result_data = {
            "image": IMAGE_PATH.name,
            "image_size": f"{IMAGE_PATH.stat().st_size / 1024:.0f} KB",
            "pipeline": "PP-OCRv5 (PaddleOCR 3.x)",
            "device": "GPU (Quadro P1000)",
            "paddle_version": __import__("paddle").__version__,
            "paddleocr_version": __import__("paddleocr").__version__,
            "init_time_s": round(t_init, 2),
            "ocr_time_s": round(t_ocr, 2),
            "total_time_s": round(total_time, 2),
            "num_blocks": len(rec_texts),
            "avg_confidence": round(sum(rec_scores) / len(rec_scores), 4) if rec_scores else 0,
            "max_confidence": round(max(rec_scores), 4) if rec_scores else 0,
            "min_confidence": round(min(rec_scores), 4) if rec_scores else 0,
            "confidence_distribution": {
                "above_0.9": sum(1 for s in rec_scores if s > 0.9),
                "0.8_to_0.9": sum(1 for s in rec_scores if 0.8 < s <= 0.9),
                "0.5_to_0.8": sum(1 for s in rec_scores if 0.5 < s <= 0.8),
                "below_0.5": sum(1 for s in rec_scores if s <= 0.5),
            },
            "texts": [
                {"text": t, "confidence": round(s, 4)}
                for t, s in zip(rec_texts, rec_scores)
                if t and t.strip()
            ],
        }
        return result_data

    return None


# ── HTML Report Generation ──
def generate_html_report(data):
    if not data:
        print("[ERROR] No data to generate report")
        return

    # Build text table rows
    text_rows = ""
    for i, item in enumerate(data["texts"]):
        conf_class = "high" if item["confidence"] > 0.95 else "mid" if item["confidence"] > 0.8 else "low"
        text_rows += f"""
        <tr>
            <td>{i + 1}</td>
            <td class="{conf_class}">{item['confidence']:.4f}</td>
            <td>{item['text']}</td>
        </tr>"""

    conf_dist = data["confidence_distribution"]
    total = sum(conf_dist.values())

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>PaddleOCR GPU Test Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif; background: #f0f2f5; color: #333; padding: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1a1a2e; border-bottom: 3px solid #4CAF50; padding-bottom: 12px; margin-bottom: 25px; font-size: 28px; }}
        h2 {{ color: #16213e; margin: 25px 0 15px; font-size: 20px; }}
        h3 {{ color: #333; margin: 15px 0 8px; font-size: 16px; }}

        /* Summary cards */
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 15px; margin-bottom: 25px; }}
        .card {{ background: white; border-radius: 10px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .card .label {{ font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
        .card .value {{ font-size: 26px; font-weight: 700; margin-top: 5px; }}
        .card .value.green {{ color: #4CAF50; }}
        .card .value.blue {{ color: #2196F3; }}
        .card .value.orange {{ color: #FF9800; }}
        .card .value.purple {{ color: #9C27B0; }}

        /* Environment table */
        .env-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 25px; }}
        .env-table td {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; }}
        .env-table td:first-child {{ font-weight: 600; color: #555; width: 200px; background: #fafafa; }}
        .env-table tr:last-child td {{ border-bottom: none; }}

        /* Results image */
        .result-img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 25px; }}

        /* Metrics section */
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 25px; }}
        .metric {{ background: white; border-radius: 8px; padding: 14px 16px; text-align: center; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }}
        .metric .val {{ font-size: 22px; font-weight: 700; }}
        .metric .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}

        /* Distribution bar */
        .dist-bar {{ display: flex; height: 30px; border-radius: 6px; overflow: hidden; margin: 10px 0 20px; }}
        .dist-bar .seg {{ display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; color: white; transition: width 0.3s; }}
        .seg-1 {{ background: #4CAF50; }}
        .seg-2 {{ background: #FFC107; color: #333; }}
        .seg-3 {{ background: #FF9800; }}
        .seg-4 {{ background: #f44336; }}

        /* Text results table */
        .text-table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
        .text-table th {{ background: #4CAF50; color: white; padding: 10px 14px; text-align: left; font-weight: 600; }}
        .text-table td {{ padding: 8px 14px; border-bottom: 1px solid #f0f0f0; font-size: 14px; }}
        .text-table tr:hover {{ background: #f5fdf5; }}
        .text-table .high {{ color: #4CAF50; font-weight: 600; }}
        .text-table .mid {{ color: #FF9800; font-weight: 600; }}
        .text-table .low {{ color: #f44336; font-weight: 600; }}
        .text-table .idx {{ color: #aaa; font-size: 12px; }}

        .footer {{ margin-top: 30px; text-align: center; color: #aaa; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">

    <h1>📄 PaddleOCR GPU Test Report</h1>

    <!-- Summary -->
    <div class="summary">
        <div class="card">
            <div class="label">Total Time</div>
            <div class="value green">{data['total_time_s']}s</div>
        </div>
        <div class="card">
            <div class="label">OCR Inference</div>
            <div class="value blue">{data['ocr_time_s']}s</div>
        </div>
        <div class="card">
            <div class="label">Text Blocks</div>
            <div class="value orange">{data['num_blocks']}</div>
        </div>
        <div class="card">
            <div class="label">Avg Confidence</div>
            <div class="value purple">{data['avg_confidence']:.4f}</div>
        </div>
    </div>

    <!-- Environment -->
    <h2>🔧 Environment</h2>
    <table class="env-table">
        <tr><td>Test Image</td><td>{data['image']} ({data['image_size']})</td></tr>
        <tr><td>Pipeline</td><td>{data['pipeline']}</td></tr>
        <tr><td>Device</td><td>{data['device']}</td></tr>
        <tr><td>PaddlePaddle</td><td>{data['paddle_version']}</td></tr>
        <tr><td>PaddleOCR</td><td>{data['paddleocr_version']}</td></tr>
        <tr><td>Init Time</td><td>{data['init_time_s']}s (model loading)</td></tr>
        <tr><td>OCR Inference</td><td>{data['ocr_time_s']}s</td></tr>
    </table>

    <!-- Visualization -->
    <h2>🖼️ OCR Visualization</h2>
    <img class="result-img" src="output_ocr_gpu/page_1_ocr_res_img.jpg" alt="OCR Visualization" />

    <!-- Quality Metrics -->
    <h2>📊 Quality Metrics</h2>
    <div class="metrics">
        <div class="metric">
            <div class="val" style="color:#4CAF50">{data['avg_confidence']:.4f}</div>
            <div class="lbl">Average Confidence</div>
        </div>
        <div class="metric">
            <div class="val" style="color:#2196F3">{data['max_confidence']:.4f}</div>
            <div class="lbl">Max Confidence</div>
        </div>
        <div class="metric">
            <div class="val" style="color:#FF9800">{data['min_confidence']:.4f}</div>
            <div class="lbl">Min Confidence</div>
        </div>
        <div class="metric">
            <div class="val" style="color:#9C27B0">{data['num_blocks']}</div>
            <div class="lbl">Text Blocks Detected</div>
        </div>
    </div>

    <h3>Confidence Distribution</h3>
    <div class="dist-bar">
        <div class="seg seg-1" style="width: {conf_dist['above_0.9']/total*100:.1f}%">{conf_dist['above_0.9']} (>0.9)</div>
        <div class="seg seg-2" style="width: {conf_dist['0.8_to_0.9']/total*100:.1f}%">{conf_dist['0.8_to_0.9']} (0.8-0.9)</div>
        <div class="seg seg-3" style="width: {conf_dist['0.5_to_0.8']/total*100:.1f}%">{conf_dist['0.5_to_0.8']} (0.5-0.8)</div>
        <div class="seg seg-4" style="width: {conf_dist['below_0.5']/total*100:.1f}%">{conf_dist['below_0.5']} (<0.5)</div>
    </div>

    <!-- Full Text Output -->
    <h2>📝 Full OCR Text Output</h2>
    <p style="color:#888;margin-bottom:10px;font-size:13px;">
        Sorted by detection order (top-to-bottom). {len(data['texts'])} text blocks found.
    </p>
    <table class="text-table">
        <thead>
            <tr><th style="width:50px">#</th><th style="width:120px">Confidence</th><th>Text</th></tr>
        </thead>
        <tbody>
            {text_rows}
        </tbody>
    </table>

    <div class="footer">
        Generated by Sisyphus · PaddleOCR GPU Test · {time.strftime("%Y-%m-%d %H:%M:%S")}
    </div>

</div>
</body>
</html>"""

    report_path = OUTPUT_DIR / "report_gpu_ocr.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{'=' * 60}")
    print(f"  Report generated: {report_path}")
    print(f"  Open in browser to view full report")
    print(f"{'=' * 60}")
    return report_path


# ── Main ──
if __name__ == "__main__":
    check_environment()
    data = run_ocr()
    if data:
        report = generate_html_report(data)
    print("\nDone!")
