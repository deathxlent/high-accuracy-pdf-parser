import os
import sys
import json
import time
from pathlib import Path

try:
    import torch
except OSError:
    pass

BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_cpu"

IMAGES = sorted([f for f in os.listdir(BASE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])

PIPELINE_VERSION = "v1.6"


def check_environment():
    print("[CHECK] Verifying environment for PaddleOCR-VL-1.6 (CPU) ...")

    try:
        import paddle
        print(f"[OK] PaddlePaddle version: {paddle.__version__}")
    except ImportError:
        print("[ERROR] PaddlePaddle not installed.")
        print("        Run: python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/")
        sys.exit(1)

    try:
        import paddleocr
        print(f"[OK] PaddleOCR version: {paddleocr.__version__}")
    except ImportError:
        print("[ERROR] PaddleOCR not installed.")
        print('        Run: python -m pip install -U "paddleocr[doc-parser]>=3.6.0"')
        sys.exit(1)

    try:
        from paddleocr import PaddleOCRVL
        print("[OK] PaddleOCRVL available")
    except ImportError:
        print("[ERROR] PaddleOCRVL not available. Ensure paddleocr[doc-parser] is installed.")
        print('        Run: python -m pip install -U "paddleocr[doc-parser]>=3.6.0"')
        sys.exit(1)


def process_images():
    from paddleocr import PaddleOCRVL

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - Document Parsing Pipeline (CPU)")
    print("=" * 60)

    print(f"[INFO] Pipeline version: {PIPELINE_VERSION}")
    print(f"[INFO] Device: CPU")
    print(f"[INFO] Images to process: {len(IMAGES)}")

    print(f"\n[LOAD] Initializing PaddleOCRVL pipeline (first run downloads models) ...")
    t0 = time.time()
    pipeline = PaddleOCRVL(
        pipeline_version=PIPELINE_VERSION,
        device="cpu",
    )
    t_load = time.time() - t0
    print(f"[OK] Pipeline initialized in {t_load:.1f}s")

    all_results = []

    for img_name in IMAGES:
        img_path = str(BASE_DIR / img_name)
        img_out_dir = OUTPUT_DIR / Path(img_name).stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PROC] Processing: {img_name}")
        t0 = time.time()

        try:
            output = pipeline.predict(img_path)

            result = {
                "image": img_name,
                "pipeline_version": PIPELINE_VERSION,
                "device": "cpu",
                "elements": [],
                "markdown": "",
                "error": None,
            }

            for res in output:
                try:
                    res.save_to_json(save_path=str(img_out_dir))
                except Exception:
                    pass

                try:
                    res.save_to_markdown(save_path=str(img_out_dir))
                except Exception:
                    pass

                try:
                    if hasattr(res, 'json'):
                        result["elements"] = res.json
                except Exception:
                    pass

                try:
                    md_path = img_out_dir / f"{Path(img_name).stem}.md"
                    if md_path.exists():
                        result["markdown"] = md_path.read_text(encoding="utf-8")
                except Exception:
                    pass

            t_proc = time.time() - t0
            result["time_seconds"] = round(t_proc, 2)
            print(f"       Done in {t_proc:.1f}s")

            all_results.append(result)

        except Exception as e:
            t_proc = time.time() - t0
            print(f"       [ERROR] {e}")
            all_results.append({
                "image": img_name,
                "pipeline_version": PIPELINE_VERSION,
                "device": "cpu",
                "elements": [],
                "markdown": "",
                "error": str(e),
                "time_seconds": round(t_proc, 2),
            })

    return all_results


def generate_report(results):
    html_parts = ["""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>PaddleOCR-VL-1.6 Results (CPU)</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #333; border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.img-box img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.img-box p { text-align: center; color: #666; font-size: 14px; margin-top: 5px; }
.md-content { background: #fafafa; padding: 15px; border: 1px solid #eee; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-family: 'Consolas', monospace; font-size: 13px; line-height: 1.6; max-height: 600px; overflow-y: auto; }
.md-content table { border-collapse: collapse; width: 100%; }
.md-content th, .md-content td { border: 1px solid #ddd; padding: 6px 10px; }
.md-content th { background: #f0f0f0; }
.error { color: red; background: #fee; padding: 10px; border-radius: 4px; }
.summary { background: #e3f2fd; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
.meta { color: #888; font-size: 13px; margin: 5px 0; }
</style></head><body>
<h1>PaddleOCR-VL-1.6 - Document Parsing Results (CPU)</h1>
"""]

    success = sum(1 for r in results if not r.get("error"))
    total_time = sum(r.get("time_seconds", 0) for r in results)
    html_parts.append(f'<div class="summary">')
    html_parts.append(f'<strong>Summary:</strong> Processed {len(results)} images, {success} successful | Total time: {total_time:.1f}s | Device: CPU | Pipeline: {PIPELINE_VERSION}')
    html_parts.append(f'</div>')

    for r in results:
        html_parts.append(f'<div class="card">')
        html_parts.append(f'<h2>{r["image"]}</h2>')
        html_parts.append(f'<div class="meta">Time: {r.get("time_seconds", 0)}s | Device: {r.get("device", "cpu")}</div>')

        if r.get("error"):
            html_parts.append(f'<div class="error">Error: {r["error"]}</div>')
        else:
            orig_img = BASE_DIR / r["image"]
            md_path = OUTPUT_DIR / Path(r["image"]).stem / f"{Path(r['image']).stem}.md"

            if orig_img.exists():
                html_parts.append(f'<div class="img-box">')
                html_parts.append(f'<img src="{orig_img}" />')
                html_parts.append(f'<p>Original: {r["image"]}</p>')
                html_parts.append(f'</div>')

            md_escaped = r.get("markdown", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if md_escaped:
                html_parts.append(f'<h3>Markdown Output</h3>')
                html_parts.append(f'<div class="md-content">{md_escaped}</div>')
            elif md_path.exists():
                html_parts.append(f'<h3>Markdown Output</h3>')
                html_parts.append(f'<p><a href="{md_path}">Download Markdown</a></p>')

        html_parts.append(f'</div>')

    html_parts.append("</body></html>")

    report_path = OUTPUT_DIR / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"\n{'=' * 60}")
    print(f"  Report generated: {report_path}")
    print(f"  Open in browser to view results")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    print("Step 1: Check environment")
    check_environment()
    print("\nStep 2: Process images")
    results = process_images()
    print("\nStep 3: Generate report")
    generate_report(results)
# python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
# python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
# python run_paddleocr_vl_cpu.py
