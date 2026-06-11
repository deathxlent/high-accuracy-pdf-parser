"""
完整性能测试脚本（所有 15 blocks）：
- 安装性能打点
- 运行完整处理
- 打印详细性能报告
"""
import os
import sys
import time
import functools
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODELS_DIR = Path(r"C:\ws\high accuracy pdf parser\models")
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR)
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
PIPELINE_VERSION = os.environ.get("PIPELINE_VERSION", "v1.6")

IMG_NAME = "page_1.jpg"

_timers = {}

def timed_call(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                dt = time.perf_counter() - t0
                _timers.setdefault(name, []).append(dt)
            return result
        return wrapper
    return decorator

_PATCHED = False

def install_profiler():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl as _vl
    import paddlex.inference.models.common.transformers.generation.utils as _gen
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._siglip as _siglip
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._projector as _proj
    import paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._ernie as _ernie

    _vl.PaddleOCRVLForConditionalGeneration.forward = timed_call(
        "0_PaddleOCRVL.forward"
    )(_vl.PaddleOCRVLForConditionalGeneration.forward)

    _siglip.SiglipVisionModel.forward = timed_call(
        "1_visual_encoder"
    )(_siglip.SiglipVisionModel.forward)

    _proj.Projector.forward = timed_call(
        "2_mlp_AR_projector"
    )(_proj.Projector.forward)

    _ernie.Ernie4_5Model.forward = timed_call(
        "3_ernie4_5_LLM"
    )(_ernie.Ernie4_5Model.forward)

    _orig_generate = _vl.PaddleOCRVLForConditionalGeneration.generate
    @functools.wraps(_orig_generate)
    def generate_wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = _orig_generate(self, *args, **kwargs)
        finally:
            dt = time.perf_counter() - t0
            n_tokens = len(result[0]) if isinstance(result, (list, tuple)) else "N/A"
            print(f"  [PROF] generate() done in {dt:.1f}s, tokens={n_tokens}")
            _timers.setdefault("generate", []).append(dt)
        return result
    _vl.PaddleOCRVLForConditionalGeneration.generate = generate_wrapper

    print("[PROF] Profiling patches installed (ALL blocks).")

def print_report():
    print()
    print("=" * 80)
    print("  FULL PIPELINE PERFORMANCE REPORT (ALL BLOCKS)")
    print("=" * 80)

    order = sorted(
        _timers.keys(),
        key=lambda k: (0 if k == "generate" else 1, k),
    )
    total_gen = sum(_timers.get("generate", []))
    for name in order:
        ts = _timers[name]
        if not ts:
            continue
        total = sum(ts)
        print(
            f"  {name:<36s}: total={total:7.3f}s  count={len(ts):>4d}  "
            f"avg={total/len(ts):.6f}s  min={min(ts):.6f}s  max={max(ts):.6f}s"
        )

    fwd_times = _timers.get("0_PaddleOCRVL.forward", [])
    ve_times = _timers.get("1_visual_encoder", [])
    proj_times = _timers.get("2_mlp_AR_projector", [])
    llm_times = _timers.get("3_ernie4_5_LLM", [])
    gen_times = _timers.get("generate", [])

    print()
    print(f"--- Summary ---")
    if gen_times:
        print(f"  Total generate() time : {sum(gen_times):.2f}s ({len(gen_times)} blocks)")
        print(f"  Average per block     : {np.mean(gen_times):.2f}s")
        print(f"  Min block             : {min(gen_times):.2f}s")
        print(f"  Max block             : {max(gen_times):.2f}s")

    if len(fwd_times) >= 1:
        print()
        print(f"--- Prefill vs Decode Breakdown ---")
        print(f"  Total forward calls: {len(fwd_times)}")
        # first call = prefill
        prefill_total = fwd_times[0]
        prefill_ve = ve_times[0] if ve_times else None
        prefill_proj = proj_times[0] if proj_times else None
        prefill_llm = llm_times[0] if llm_times else None
        prefill_other = prefill_total
        if prefill_ve:
            prefill_other -= prefill_ve
        if prefill_proj:
            prefill_other -= prefill_proj
        if prefill_llm:
            prefill_other -= prefill_llm
        print(f"  First forward (PREFILL) block 0:")
        print(f"    total forward: {prefill_total:7.4f}s")
        if prefill_ve:
            print(f"    visual_encoder: {prefill_ve:7.4f}s")
        if prefill_proj:
            print(f"    projector:      {prefill_proj:7.4f}s")
        if prefill_llm:
            print(f"    LLM:            {prefill_llm:7.4f}s")
        print(f"    other(embed+rope+scatter): {max(0,prefill_other):7.4f}s")

        if len(fwd_times) > 1:
            decode_fwd = fwd_times[1:]
            decode_llm = llm_times[1:] if len(llm_times) > 1 else []
            print(f"  Decode forwards (avg over {len(decode_fwd)} calls):")
            print(f"    total forward avg: {np.mean(decode_fwd):.6f}s")
            if decode_llm:
                print(f"    LLM avg:           {np.mean(decode_llm):.6f}s")

    print()

def main():
    install_profiler()

    t0_load = time.perf_counter()
    from paddleocr import PaddleOCRVL

    print()
    print("=" * 60)
    print(f"  PaddleOCR-VL-1.6 - Full Performance Test (ALL blocks)")
    print("=" * 60)
    print()

    pipeline = PaddleOCRVL(
        pipeline_version=PIPELINE_VERSION,
        device="gpu",
    )
    t1_load = time.perf_counter()
    print(f"[OK] Pipeline initialized in {t1_load - t0_load:.1f}s")
    print()

    img_path = str(Path(IMAGE_DIR) / IMG_NAME)
    print(f"[PROC] Processing: {IMG_NAME}")

    t0_proc = time.perf_counter()
    output = list(pipeline.predict(img_path))
    t1_proc = time.perf_counter()
    print(f"       Done in {t1_proc - t0_proc:.1f}s")

    print_report()

if __name__ == "__main__":
    main()
