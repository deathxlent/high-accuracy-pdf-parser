"""逐block分析：输出每个block的像素数、generate时间和子模块时间"""
import os
import sys
import time
import functools
import numpy as np
from pathlib import Path

MODELS_DIR = Path(r"C:\ws\high accuracy pdf parser\models")
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
PIPELINE_VERSION = "v1.6"

_timers = {}
_block_info = []

def main():
    from paddleocr import PaddleOCRVL

    print()
    print("=" * 80)
    print("  PaddleOCR-VL-1.6 - Per-Block Analysis")
    print("=" * 80)

    pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION, device="gpu")

    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl as _vl
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._siglip as _siglip
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._ernie as _ernie
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._projector as _proj

    _orig_generate = _vl.PaddleOCRVLForConditionalGeneration.generate
    _orig_forward = _vl.PaddleOCRVLForConditionalGeneration.forward

    def generate_with_info(self, *args, **kwargs):
        pixel_count = None
        grid_thw = kwargs.get("image_grid_thw", None)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if grid_thw is not None:
            try:
                if hasattr(grid_thw, 'shape') and len(grid_thw.shape) >= 1:
                    pixel_count = int(np.prod(grid_thw.shape[0] * [np.prod(g) for g in grid_thw.numpy()]))
                else:
                    pixel_count = sum(np.prod(thw) for thw in grid_thw)
            except Exception:
                pixel_count = "N/A"

        n_input_tokens = input_ids.shape[1] if input_ids is not None and hasattr(input_ids, 'shape') else "N/A"

        ve_times = []
        llm_times = []

        _orig_ve = _siglip.SiglipVisionModel.forward
        def ve_timed(self_ve, *a, **kw):
            t0 = time.perf_counter()
            r = _orig_ve(self_ve, *a, **kw)
            ve_times.append(time.perf_counter() - t0)
            return r
        _siglip.SiglipVisionModel.forward = ve_timed

        _orig_llm = _ernie.Ernie4_5Model.forward
        def llm_timed(self_llm, *a, **kw):
            t0 = time.perf_counter()
            r = _orig_llm(self_llm, *a, **kw)
            llm_times.append(time.perf_counter() - t0)
            return r
        _ernie.Ernie4_5Model.forward = llm_timed

        t0 = time.perf_counter()
        try:
            result = _orig_generate(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            n_out = len(result[0]) if isinstance(result, (list, tuple)) else "N/A"
            info = {
                "time": dt,
                "pixels": pixel_count,
                "input_tokens": n_input_tokens,
                "output_tokens": n_out,
                "ve_total": sum(ve_times),
                "ve_count": len(ve_times),
                "llm_total": sum(llm_times),
                "llm_count": len(llm_times),
                "llm_avg_decode": np.mean(llm_times[1:]) if len(llm_times) > 1 else 0,
            }
            _block_info.append(info)
            px_str = str(pixel_count)
            in_str = str(n_input_tokens)
            out_str = str(n_out)
            print(f"  [{len(_block_info):2d}] gen={dt:6.1f}s  pixels={px_str:>8s}  "
                  f"in_tok={in_str:>4s}  out_tok={out_str:>4s}  "
                  f"ve={sum(ve_times):5.1f}s  llm={sum(llm_times):5.1f}s  "
                  f"llm_decode_avg={np.mean(llm_times[1:]) if len(llm_times) > 1 else 0:.3f}s")

        _siglip.SiglipVisionModel.forward = _orig_ve
        _ernie.Ernie4_5Model.forward = _orig_llm
        return result

    _vl.PaddleOCRVLForConditionalGeneration.generate = generate_with_info

    img_path = str(Path(IMAGE_DIR) / "page_1.jpg")
    print(f"\n[PROC] Processing: page_1.jpg")

    t0 = time.perf_counter()
    output = list(pipeline.predict(img_path))
    t1 = time.perf_counter()
    print(f"\n       Total: {t1 - t0:.1f}s")

    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    total_gen = sum(b["time"] for b in _block_info)
    total_ve = sum(b["ve_total"] for b in _block_info)
    total_llm = sum(b["llm_total"] for b in _block_info)
    print(f"  Total generate: {total_gen:.1f}s  |  Visual Encoder: {total_ve:.1f}s  |  LLM: {total_llm:.1f}s")

    normal = [b for b in _block_info if b["time"] < 30]
    slow = [b for b in _block_info if b["time"] >= 30]
    if normal:
        print(f"\n  Normal blocks ({len(normal)}): avg={np.mean([b['time'] for b in normal]):.1f}s  "
              f"min={min(b['time'] for b in normal):.1f}s  max={max(b['time'] for b in normal):.1f}s")
    if slow:
        print(f"  Slow blocks ({len(slow)}): avg={np.mean([b['time'] for b in slow]):.1f}s  "
              f"pixels={[b['pixels'] for b in slow]}")
    print()

if __name__ == "__main__":
    main()
