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

_timers = defaultdict(list)
_block_count = [0]
_MAX_BLOCKS = 3


def _time_method(cls_ref, method_name, timer_name):
    orig = getattr(cls_ref, method_name)
    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        result = orig(self, *args, **kwargs)
        dt = time.time() - t0
        _timers[timer_name].append(dt)
        return result
    setattr(cls_ref, method_name, wrapper)


def install_profiling():
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._siglip import SiglipVisionModel
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._projector import Projector
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._ernie import Ernie4_5Model

    _time_method(SiglipVisionModel, "forward", "1_visual_encoder")
    _time_method(Projector, "forward", "2_mlp_AR_projector")
    _time_method(Ernie4_5Model, "forward", "3_ernie4_5_LLM")

    orig_forward = PaddleOCRVLForConditionalGeneration.forward
    @functools.wraps(orig_forward)
    def timed_forward(self, *args, **kwargs):
        t0 = time.time()
        result = orig_forward(self, *args, **kwargs)
        dt = time.time() - t0
        _timers["0_PaddleOCRVL.forward"].append(dt)
        return result
    PaddleOCRVLForConditionalGeneration.forward = timed_forward

    orig_generate = PaddleOCRVLForConditionalGeneration.generate
    @functools.wraps(orig_generate)
    def timed_generate(self, inputs, **kwargs):
        block_idx = _block_count[0]
        _block_count[0] += 1

        px_info = ""
        for k, v in inputs.items():
            if isinstance(v, paddle.Tensor):
                px_info += f" {k}={list(v.shape)}"

        if block_idx >= _MAX_BLOCKS:
            print(f"\n  [PROF] === Block {block_idx}: SKIPPED (max_blocks={_MAX_BLOCKS}) ===")
            return paddle.zeros([1, 1], dtype=paddle.int64)

        print(f"\n  [PROF] === Block {block_idx}: generate(){px_info} ===")
        t0 = time.time()
        result = orig_generate(self, inputs, **kwargs)
        dt = time.time() - t0
        _timers["generate"].append(dt)
        n_tok = result.shape[-1] if isinstance(result, paddle.Tensor) else "N/A"
        print(f"  [PROF] === Block {block_idx}: generate() done in {dt:.1f}s, tokens={n_tok} ===")
        return result
    PaddleOCRVLForConditionalGeneration.generate = timed_generate

    print("[PROF] Profiling patches installed (max_blocks=3).")


BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_gpu"
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
IMAGES = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])
PIPELINE_VERSION = "v1.6"


def print_report():
    print("\n" + "=" * 80)
    print("  SUB-MODULE BREAKDOWN REPORT (AFTER FIXES)")
    print("=" * 80)

    for name, times in sorted(_timers.items()):
        if not times:
            continue
        total = sum(times)
        avg = total / len(times)
        print(f"  {name:35s}: total={total:8.3f}s  count={len(times):4d}  avg={avg:.6f}s  min={min(times):.6f}s  max={max(times):.6f}s")

    print("\n--- Prefill vs Decode Breakdown ---")
    fwd_times = _timers.get("0_PaddleOCRVL.forward", [])
    vis_times = _timers.get("1_visual_encoder", [])
    proj_times = _timers.get("2_mlp_AR_projector", [])
    llm_times = _timers.get("3_ernie4_5_LLM", [])

    n_fwd = len(fwd_times)
    print(f"  Total forward calls: {n_fwd}")
    if n_fwd > 0:
        print(f"  First forward (PREFILL):")
        print(f"    total forward:  {fwd_times[0]:.4f}s")
        if vis_times:
            print(f"    visual_encoder: {vis_times[0]:.4f}s")
        if proj_times:
            print(f"    projector:      {proj_times[0]:.4f}s")
        if llm_times:
            print(f"    LLM:            {llm_times[0]:.4f}s")
        overhead = fwd_times[0]
        if vis_times:
            overhead -= vis_times[0]
        if proj_times:
            overhead -= proj_times[0]
        if llm_times:
            overhead -= llm_times[0]
        print(f"    other(embed+rope+scatter): {overhead:.4f}s")

    if n_fwd > 1:
        decode_fwd = fwd_times[1:]
        decode_llm = llm_times[1:] if len(llm_times) > 1 else []
        print(f"\n  Decode forwards (avg over {len(decode_fwd)} calls):")
        print(f"    total forward avg: {sum(decode_fwd)/len(decode_fwd):.6f}s")
        if decode_llm:
            print(f"    LLM avg:           {sum(decode_llm)/len(decode_llm):.6f}s")

    print("\n" + "=" * 80)


def main():
    install_profiling()

    from paddleocr import PaddleOCRVL
    from paddlex.inference.utils.misc import is_bfloat16_available, is_float16_available

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - Post-Fix Validation (3 blocks)")
    print("=" * 60)

    print(f"\n[ENV] is_bfloat16_available('gpu'): {is_bfloat16_available('gpu')}")
    print(f"[ENV] is_float16_available('gpu'): {is_float16_available('gpu')}")
    print(f"[ENV] PaddlePaddle: {paddle.__version__}")

    print(f"\n[LOAD] Initializing pipeline...")
    t0 = time.time()
    pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION, device="gpu")
    t_load = time.time() - t0
    print(f"[OK] Pipeline initialized in {t_load:.1f}s")

    try:
        predictor = pipeline.paddlex_pipeline.vl_rec_model
        print(f"\n[PRED] dtype setting: {predictor.dtype}")
    except Exception as e:
        print(f"\n[PRED] Could not read predictor.dtype: {e}")

    model = pipeline.paddlex_pipeline.vl_rec_model.infer
    print(f"\n[MODEL] dtype check (first 5 params):")
    all_ok = True
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: dtype={param.dtype}, place={param.place}")
        if param.dtype != paddle.float16:
            all_ok = False

    print(f"\n[MODEL] All params are fp16: {all_ok}")

    print(f"\n[MODEL] Checking visual encoder attention SDPA support:")
    try:
        vision_model = model.vision_model if hasattr(model, 'vision_model') else model.vision_tower
        first_layer = vision_model.encoder.layers[0] if hasattr(vision_model, 'encoder') else None
        if first_layer is not None and hasattr(first_layer, 'self_attn'):
            print(f"  vision encoder layer.self_attn._supports_sdpa = {first_layer.self_attn._supports_sdpa}")
    except Exception as e:
        print(f"  [WARN] Could not check SDPA: {e}")

    for img_name in IMAGES[:1]:
        img_path = IMAGE_DIR + "/" + img_name
        img_out_dir = OUTPUT_DIR / Path(img_name).stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PROC] Processing: {img_name} (max {_MAX_BLOCKS} blocks)")
        _timers.clear()
        _block_count[0] = 0

        t0 = time.time()
        try:
            output = pipeline.predict(img_path)
        except Exception as e:
            import traceback
            print(f"  [WARN] Error after {_MAX_BLOCKS} blocks: {e}")
            traceback.print_exc()
        t_proc = time.time() - t0
        print(f"       Done in {t_proc:.1f}s")

    print_report()


if __name__ == "__main__":
    main()
