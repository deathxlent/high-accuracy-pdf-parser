import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

try:
    import torch
except OSError:
    pass

import paddle

_original_prod = paddle.Tensor.prod
def _patched_prod(self, *args, **kwargs):
    if self.place.is_gpu_place():
        return _original_prod(self.cpu(), *args, **kwargs)
    return _original_prod(self, *args, **kwargs)
paddle.Tensor.prod = _patched_prod

def _patch_update_model_kwargs_for_generation():
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )
    from paddlex.inference.models.common.transformers.transformers.model_outputs import (
        ModelOutput,
    )

    _orig_update = PaddleOCRVLForConditionalGeneration.update_model_kwargs_for_generation

    def _patched_update(self, outputs, model_kwargs, is_encoder_decoder=False):
        if isinstance(outputs, ModelOutput) and "past_key_values" in outputs and outputs.past_key_values is not None:
            model_kwargs["past_key_values"] = outputs.past_key_values
        elif isinstance(outputs, tuple) and len(outputs) > 1 and not isinstance(outputs[1], paddle.Tensor):
            model_kwargs["past_key_values"] = outputs[1]
        if not is_encoder_decoder and model_kwargs.get("attention_mask", None) is not None:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = paddle.concat(
                [
                    attention_mask,
                    paddle.ones(
                        [attention_mask.shape[0], 1], dtype=attention_mask.dtype
                    ),
                ],
                axis=-1,
            )
        return model_kwargs

    PaddleOCRVLForConditionalGeneration.update_model_kwargs_for_generation = _patched_update

_patch_update_model_kwargs_for_generation()

BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_gpu"
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
IMAGES = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])

PIPELINE_VERSION = "v1.6"


def _get_cuda_info():
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            gpu_count = paddle.device.cuda.device_count()
            gpu_name = paddle.device.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            compute_capability = None
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    compute_capability = result.stdout.strip().split("\n")[0].strip()
            except Exception:
                pass
            return True, gpu_count, gpu_name, compute_capability
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "N/A"
            compute_capability = None
            try:
                cap = torch.cuda.get_device_capability(0)
                compute_capability = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass
            return True, gpu_count, gpu_name, compute_capability
    except Exception:
        pass
    return False, 0, "N/A", None


def _get_paddle_cuda_version():
    try:
        import paddle
        cuda_ver = paddle.version.cuda
        if cuda_ver and cuda_ver != "False":
            return cuda_ver
    except Exception:
        pass
    return None


def _is_sm_compatible(compute_capability):
    if not compute_capability:
        return True, None
    try:
        major, minor = map(int, compute_capability.split("."))
        sm_ver = major * 10 + minor
        
        cuda_ver = _get_paddle_cuda_version()
        if cuda_ver:
            cuda_major = int(cuda_ver.split(".")[0])
            if cuda_major >= 12:
                min_sm = 70
                cuda_label = "CUDA 12.x"
            else:
                min_sm = 35
                cuda_label = "CUDA 11.x"
        else:
            min_sm = 70
            cuda_label = "CUDA 12.x"
        
        if sm_ver < min_sm:
            return False, f"SM {compute_capability} (requires SM >= {min_sm//10}.{min_sm%10} with {cuda_label})"
        return True, None
    except Exception:
        return True, None


def check_environment():
    print("[CHECK] Verifying environment for PaddleOCR-VL-1.6 (GPU) ...")

    has_cuda, gpu_count, gpu_name, compute_capability = _get_cuda_info()
    if not has_cuda:
        print("[ERROR] No CUDA GPU detected.")
        print("        Please ensure NVIDIA GPU + drivers are installed, then run:")
        print("        python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
        sys.exit(1)

    print(f"[OK] GPU detected: {gpu_name} (x{gpu_count})")
    if compute_capability:
        print(f"[OK] Compute Capability: SM {compute_capability}")
        compatible, reason = _is_sm_compatible(compute_capability)
        if not compatible:
            print(f"[WARN] GPU may be incompatible: {reason}")
            print(f"       Will attempt auto-fallback to CPU mode if GPU fails.")

    try:
        import paddle
        is_gpu = paddle.device.is_compiled_with_cuda()
        print(f"[OK] PaddlePaddle version: {paddle.__version__} (GPU build: {is_gpu})")
        if not is_gpu:
            print("[WARN] PaddlePaddle GPU version not detected. For best performance, install GPU build:")
            print("       python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
    except ImportError:
        print("[ERROR] PaddlePaddle not installed.")
        print("        Run: python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/")
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

    has_cuda, gpu_count, gpu_name, compute_capability = _get_cuda_info()

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - Document Parsing Pipeline (GPU)")
    print("=" * 60)

    print(f"[INFO] Pipeline version: {PIPELINE_VERSION}")
    print(f"[INFO] Device: GPU ({gpu_name})")
    print(f"[INFO] GPU count: {gpu_count}")
    if compute_capability:
        print(f"[INFO] Compute Capability: SM {compute_capability}")
        compatible, reason = _is_sm_compatible(compute_capability)
        if not compatible:
            print(f"[WARN] GPU incompatibility detected: {reason}")
            print(f"       Please install CUDA 11.x compatible paddlepaddle-gpu:")
            print(f"       pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
    print(f"[INFO] Model cache dir: {MODELS_DIR}")
    print(f"[INFO] Images to process: {len(IMAGES)}")

    if len(IMAGES) == 0:
        print("\n[WARN] No test images found.")
        print(f"       Place images named 'page_*.jpg/png' in: {BASE_DIR}")
        print("       Continuing to initialize pipeline (model download test)...")

    print(f"\n[LOAD] Initializing PaddleOCRVL pipeline with GPU ...")
    print(f"       This may take several minutes on first run...")
    t0 = time.time()
    try:
        pipeline = PaddleOCRVL(
            pipeline_version=PIPELINE_VERSION,
            device="gpu",
        )
        t_load = time.time() - t0
        print(f"[OK] Pipeline initialized (GPU) in {t_load:.1f}s")
    except Exception as e:
        err_msg = str(e)
        print(f"[ERROR] GPU initialization failed: {err_msg}")
        if "SM" in err_msg or "not compiled" in err_msg or "compute" in err_msg.lower():
            print("[ERROR] This appears to be a GPU compute capability incompatibility.")
            print("[ERROR] Your GPU has SM {compute_capability}, but current paddlepaddle-gpu requires SM >= 7.0.".format(
                compute_capability=compute_capability or "unknown"))
            print("[ERROR] Solution: Install CUDA 11.x compatible paddlepaddle-gpu:")
            print("[ERROR]   pip uninstall -y paddlepaddle-gpu paddlepaddle")
            print("[ERROR]   pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/")
        raise

    if len(IMAGES) == 0:
        print("\n[INFO] Pipeline initialized successfully. No images to process.")
        print(f"       Put 'page_*.jpg/png' images in: {BASE_DIR}")
        return []

    all_results = []

    for img_name in IMAGES:
        img_path = IMAGE_DIR +"/"+ img_name
        img_out_dir = OUTPUT_DIR / Path(img_name).stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PROC] Processing: {img_name}")
        t0 = time.time()

        try:
            output = pipeline.predict(img_path)

            result = {
                "image": img_name,
                "pipeline_version": PIPELINE_VERSION,
                "device": f"gpu:{gpu_name}",
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
                "device": f"gpu:{gpu_name}",
                "elements": [],
                "markdown": "",
                "error": str(e),
                "time_seconds": round(t_proc, 2),
            })

    return all_results


def generate_report(results):
    if not results:
        print("\n[INFO] No results to generate report for (no images processed).")
        return

    html_parts = ["""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>PaddleOCR-VL-1.6 Results (GPU)</title>
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
h2 { color: #555; margin-top: 30px; }
.card { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.img-box img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
.img-box p { text-align: center; color: #666; font-size: 14px; margin-top: 5px; }
.md-content { background: #fafafa; padding: 15px; border: 1px solid #eee; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; font-family: 'Consolas', monospace; font-size: 13px; line-height: 1.6; max-height: 600px; overflow-y: auto; }
.md-content table { border-collapse: collapse; width: 100%; }
.md-content th, .md-content td { border: 1px solid #ddd; padding: 6px 10px; }
.md-content th { background: #f0f0f0; }
.error { color: red; background: #fee; padding: 10px; border-radius: 4px; }
.summary { background: #e8f5e9; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
.meta { color: #888; font-size: 13px; margin: 5px 0; }
</style></head><body>
<h1>PaddleOCR-VL-1.6 - Document Parsing Results (GPU)</h1>
"""]

    success = sum(1 for r in results if not r.get("error"))
    total_time = sum(r.get("time_seconds", 0) for r in results)
    gpu_info = results[0].get("device", "gpu") if results else "gpu"
    html_parts.append(f'<div class="summary">')
    html_parts.append(f'<strong>Summary:</strong> Processed {len(results)} images, {success} successful | Total time: {total_time:.1f}s | Device: {gpu_info} | Pipeline: {PIPELINE_VERSION}')
    html_parts.append(f'</div>')

    for r in results:
        html_parts.append(f'<div class="card">')
        html_parts.append(f'<h2>{r["image"]}</h2>')
        html_parts.append(f'<div class="meta">Time: {r.get("time_seconds", 0)}s | Device: {r.get("device", "gpu")}</div>')

        if r.get("error"):
            html_parts.append(f'<div class="error">Error: {r["error"]}</div>')
        else:
            orig_img = Path(IMAGE_DIR) / r["image"]
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
# python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
# python -m pip install -U "paddleocr[doc-parser]>=3.6.0"
# python run_paddleocr_vl_gpu.py