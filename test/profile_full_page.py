"""Full-page profiler: KV cache patch + VE/LLM call counting (no unpatching, no block limit)."""
import os
import sys
import time
import functools
from pathlib import Path
from collections import defaultdict

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

# ---- KV cache patch ----
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
                [attention_mask, paddle.ones([attention_mask.shape[0], 1], dtype=attention_mask.dtype)],
                axis=-1,
            )
        return model_kwargs
    PaddleOCRVLForConditionalGeneration.update_model_kwargs_for_generation = _patched_update

_patch_update_model_kwargs_for_generation()

# ---- Per-block tracking (no unpatching, no timing overhead) ----
_ve_call_count = [0]
_llm_call_count = [0]
_ve_time = [0.0]
_llm_time = [0.0]
_block_stats = []

def install_counters():
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._siglip import SiglipVisionModel
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._ernie import Ernie4_5Model

    orig_ve_forward = SiglipVisionModel.forward
    @functools.wraps(orig_ve_forward)
    def ve_counted(self, *args, **kwargs):
        _ve_call_count[0] += 1
        t0 = time.perf_counter()
        result = orig_ve_forward(self, *args, **kwargs)
        _ve_time[0] += time.perf_counter() - t0
        return result
    SiglipVisionModel.forward = ve_counted

    orig_llm_forward = Ernie4_5Model.forward
    @functools.wraps(orig_llm_forward)
    def llm_counted(self, *args, **kwargs):
        _llm_call_count[0] += 1
        t0 = time.perf_counter()
        result = orig_llm_forward(self, *args, **kwargs)
        _llm_time[0] += time.perf_counter() - t0
        return result
    Ernie4_5Model.forward = llm_counted

    orig_generate = PaddleOCRVLForConditionalGeneration.generate
    @functools.wraps(orig_generate)
    def tracked_generate(self, *args, **kwargs):
        block_idx = len(_block_stats)
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        n_in = input_ids.shape[1] if input_ids is not None and hasattr(input_ids, 'shape') else "?"
        px_shape = "?"
        pv = kwargs.get("pixel_values", None)
        if pv is not None and hasattr(pv, 'shape'):
            px_shape = list(pv.shape)

        _ve_call_count[0] = 0
        _llm_call_count[0] = 0
        ve_before = _ve_time[0]
        llm_before = _llm_time[0]

        t0 = time.perf_counter()
        result = orig_generate(self, *args, **kwargs)
        dt = time.perf_counter() - t0
        ve_dt = _ve_time[0] - ve_before
        llm_dt = _llm_time[0] - llm_before

        n_out = result[0].shape[1] if isinstance(result, (list, tuple)) else "?"
        n_ve = _ve_call_count[0]
        n_llm = _llm_call_count[0]

        _block_stats.append({
            "time": dt,
            "ve_calls": n_ve,
            "ve_time": ve_dt,
            "llm_calls": n_llm,
            "llm_time": llm_dt,
            "in_tok": n_in,
            "out_tok": n_out,
            "px_shape": px_shape,
        })
        llm_decode_avg = (llm_dt / (n_llm - 1)) if n_llm > 1 else 0
        print(f"  [{block_idx+1:2d}] gen={dt:6.1f}s  ve={ve_dt:5.1f}s({n_ve}x)  llm={llm_dt:5.1f}s({n_llm}x)  "
              f"out_tok={n_out:3d}  llm_decode_avg={llm_decode_avg:.3f}s  px={px_shape}")
        return result
    PaddleOCRVLForConditionalGeneration.generate = tracked_generate

    print("[PROF] KV cache patch + counters installed.")

install_counters()

BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_gpu"
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
IMAGES = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])
PIPELINE_VERSION = "v1.6"

def print_report():
    print("\n" + "=" * 80)
    print("  FULL-PAGE BREAKDOWN REPORT")
    print("=" * 80)
    total_gen = sum(b["time"] for b in _block_stats)
    total_ve = sum(b["ve_time"] for b in _block_stats)
    total_llm = sum(b["llm_time"] for b in _block_stats)
    total_ve_calls = sum(b["ve_calls"] for b in _block_stats)
    total_llm_calls = sum(b["llm_calls"] for b in _block_stats)
    total_tokens = sum(b["out_tok"] for b in _block_stats)

    print(f"  Blocks: {len(_block_stats)}")
    print(f"  Total generate: {total_gen:.1f}s")
    print(f"  Visual Encoder: {total_ve:.1f}s ({total_ve_calls} calls)")
    print(f"  LLM:            {total_llm:.1f}s ({total_llm_calls} calls)")
    print(f"  Total tokens:   {total_tokens}")
    print(f"  Other overhead: {total_gen - total_ve - total_llm:.1f}s")

    slow_blocks = [b for b in _block_stats if b["time"] > 30]
    fast_blocks = [b for b in _block_stats if b["time"] <= 30]
    if fast_blocks:
        f_gen = sum(b["time"] for b in fast_blocks)
        f_ve = sum(b["ve_time"] for b in fast_blocks)
        f_llm = sum(b["llm_time"] for b in fast_blocks)
        f_tok = sum(b["out_tok"] for b in fast_blocks)
        print(f"\n  Fast blocks ({len(fast_blocks)}): gen={f_gen:.1f}s  ve={f_ve:.1f}s  llm={f_llm:.1f}s  tokens={f_tok}")
    if slow_blocks:
        s_gen = sum(b["time"] for b in slow_blocks)
        s_ve = sum(b["ve_time"] for b in slow_blocks)
        s_llm = sum(b["llm_time"] for b in slow_blocks)
        s_tok = sum(b["out_tok"] for b in slow_blocks)
        print(f"\n  SLOW blocks ({len(slow_blocks)}): gen={s_gen:.1f}s ({s_gen/total_gen*100:.0f}%) "
              f"ve={s_ve:.1f}s llm={s_llm:.1f}s tokens={s_tok}")

    # Check KV cache effectiveness
    blocks_with_multi_ve = [b for b in _block_stats if b["ve_calls"] > 1]
    if blocks_with_multi_ve:
        print(f"\n  ⚠ KV cache MISS for {len(blocks_with_multi_ve)} blocks (VE called >1x):")
        for b in blocks_with_multi_ve:
            idx = _block_stats.index(b) + 1
            print(f"    Block {idx}: VE called {b['ve_calls']}x ({b['ve_time']:.1f}s) out_tok={b['out_tok']}")
    else:
        print(f"\n  ✅ KV cache HIT for ALL blocks (VE called exactly once per block)")

    print("\n" + "=" * 80)

def main():
    from paddleocr import PaddleOCRVL

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - Full Page Profile (KV patch + counters)")
    print("=" * 60)

    print(f"\n[LOAD] Initializing pipeline...")
    t0 = time.time()
    pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION, device="gpu")
    t_load = time.time() - t0
    print(f"[OK] Pipeline initialized in {t_load:.1f}s")

    for img_name in IMAGES[:1]:
        img_path = IMAGE_DIR + "/" + img_name
        print(f"\n[PROC] Processing: {img_name}")
        t0 = time.time()
        try:
            output = pipeline.predict(img_path)
        except Exception as e:
            print(f"  [WARN] Error: {e}")
        t_proc = time.time() - t0
        print(f"\n  Total predict: {t_proc:.1f}s")
        print(f"  Generate subtotal: {sum(b['time'] for b in _block_stats):.1f}s")

    print_report()

if __name__ == "__main__":
    main()
