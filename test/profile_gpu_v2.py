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
_counters = defaultdict(int)
_block_count = [0]
_step_data = []
_forward_call_data = []


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
        PaddleOCRVLCausalLMOutputWithPast,
    )
    from paddlex.inference.models.common.transformers.generation.utils import GenerationMixin
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._siglip import SiglipVisionModel
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._projector import Projector
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._ernie import Ernie4_5Model

    _time_method(SiglipVisionModel, "forward", "visual_encoder.forward")
    _time_method(Projector, "forward", "mlp_AR.forward")
    _time_method(Ernie4_5Model, "forward", "ernie4_5_LLM.forward")

    orig_forward = PaddleOCRVLForConditionalGeneration.forward
    @functools.wraps(orig_forward)
    def timed_forward(self, *args, **kwargs):
        call_idx = len(_timers["PaddleOCRVL.forward"])
        block_idx = _block_count[0]

        has_px = kwargs.get('pixel_values') is not None
        has_pkv = kwargs.get('past_key_values') is not None
        if has_px and not has_pkv:
            phase = "PREFILL"
        elif has_pkv:
            phase = "DECODE"
        else:
            phase = "FORWARD"

        t0 = time.time()
        result = orig_forward(self, *args, **kwargs)
        dt = time.time() - t0

        _timers["PaddleOCRVL.forward"].append(dt)
        _forward_call_data.append((block_idx, call_idx, phase, dt))
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

        print(f"\n  [PROF] === Block {block_idx}: generate(){px_info} ===")
        t0 = time.time()
        result = orig_generate(self, inputs, **kwargs)
        dt = time.time() - t0
        _timers["generate"].append(dt)
        n_tok = result.shape[-1] if isinstance(result, paddle.Tensor) else "N/A"
        print(f"  [PROF] === Block {block_idx}: generate() done in {dt:.1f}s, tokens={n_tok} ===")
        return result
    PaddleOCRVLForConditionalGeneration.generate = timed_generate

    orig_greedy = GenerationMixin.greedy_search
    def profiled_greedy(self, input_ids, logits_processors, max_length, pad_token_id, eos_token_id,
                         stopping_criteria=None, streamer=None, fast_ptq_sampling=False,
                         trunc_input=True, synced_gpus=False, **model_kwargs):
        from paddlex.inference.models.common.transformers.generation.utils import (
            LogitsProcessorList, StoppingCriteriaList, get_unfinished_flag
        )
        from paddlex.inference.models.common.transformers.generation.stopping_criteria import validate_stopping_criteria

        block_idx = _block_count[0] - 1
        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        generate_end = False
        step_count = 0

        while True:
            step_start = time.time()
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                from paddlex.inference.models.common.transformers.transformers.model_outputs import ModelOutput
                if isinstance(outputs, ModelOutput):
                    logits = outputs.logits
                else:
                    logits = outputs

            next_token_logits = logits[:, -1, :]
            next_token_logits = self.adjust_logits_during_generation(next_token_logits)
            probs = logits_processors(input_ids, next_token_logits)
            next_tokens = paddle.argmax(probs, axis=-1).unsqueeze(-1)
            next_scores = paddle.index_sample(probs, next_tokens)

            if eos_token_id is not None:
                next_tokens = paddle.where(
                    unfinished_flag, next_tokens,
                    paddle.full_like(next_tokens, pad_token_id),
                )

            scores = self.update_scores_for_generation(scores, next_scores, cur_len - origin_len, unfinished_flag)
            cur_len += 1
            input_ids = paddle.concat([input_ids, next_tokens], axis=1)

            if stopping_criteria(input_ids, scores):
                generate_end = True
            if eos_token_id is not None:
                unfinished_flag = get_unfinished_flag(input_ids, unfinished_flag, eos_token_id)
                if not paddle.any(unfinished_flag):
                    generate_end = True

            step_time = time.time() - step_start
            step_count += 1

            kv_len = ""
            if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
                pkv = model_kwargs["past_key_values"]
                kv_len = pkv[0][0].shape[1]

            _step_data.append((block_idx, step_count, step_time, cur_len, kv_len))

            if step_count <= 5 or step_count % 100 == 0 or generate_end:
                print(f"  [PROF]     step {step_count}: {step_time:.4f}s  cur_len={cur_len}  kv_len={kv_len}")

            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if fast_ptq_sampling:
                break

        block_steps = [s for s in _step_data if s[0] == block_idx]
        total_step_time = sum(s[2] for s in block_steps)
        print(f"  [PROF]   greedy_search done: {step_count} steps, total_step_time={total_step_time:.2f}s")
        if streamer is not None:
            streamer.end()
        return input_ids[:, origin_len:] if trunc_input else input_ids, scores

    GenerationMixin.greedy_search = profiled_greedy

    orig_prod = paddle.Tensor.prod
    def counting_prod(self, *args, **kwargs):
        if self.place.is_gpu_place():
            _counters["prod_gpu_cpu"] += 1
        return orig_prod(self, *args, **kwargs)
    paddle.Tensor.prod = counting_prod

    print("[PROF] Profiling patches installed.")


BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_gpu"
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
IMAGES = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])
PIPELINE_VERSION = "v1.6"


def print_report():
    print("\n" + "=" * 80)
    print("  DETAILED PROFILING REPORT")
    print("=" * 80)

    print("\n--- Block-Level Summary ---")
    block_ids = sorted(set(b for b, _, _, _, _ in _step_data))
    for bid in block_ids:
        steps = [(s, t, cl, kv) for b, s, t, cl, kv in _step_data if b == bid]
        total_t = sum(t for _, t, _, _ in steps)
        n_steps = len(steps)
        first_step = steps[0][1] if steps else 0
        avg_decode = sum(t for _, t, _, _ in steps[1:]) / max(n_steps - 1, 1) if n_steps > 1 else 0
        print(f"  Block {bid}: {n_steps} steps, step_total={total_t:.2f}s, "
              f"first_step={first_step:.4f}s, avg_decode={avg_decode:.4f}s")

    print("\n--- Sub-module Timing (per call) ---")
    for name, times in sorted(_timers.items()):
        if not times:
            continue
        total = sum(times)
        avg = total / len(times)
        print(f"  {name:40s}: total={total:8.3f}s  count={len(times):4d}  avg={avg:.6f}s  min={min(times):.6f}s  max={max(times):.6f}s")

    print("\n--- Forward Call Breakdown (first 30) ---")
    for i, (blk, call, phase, dt) in enumerate(_forward_call_data[:30]):
        print(f"  [{i:3d}] Block{blk} Call{call:3d} {phase:8s} {dt:.4f}s")
    if len(_forward_call_data) > 30:
        print(f"  ... ({len(_forward_call_data)} total forward calls)")

    print("\n--- Decode Step Timing (sample) ---")
    decode_steps = [(b, s, t, cl, kv) for b, s, t, cl, kv in _step_data if s > 1]
    if decode_steps:
        avg_decode = sum(t for _, _, t, _, _ in decode_steps) / len(decode_steps)
        print(f"  Average decode step: {avg_decode:.4f}s over {len(decode_steps)} steps")
        buckets = defaultdict(int)
        for _, _, t, _, _ in decode_steps:
            if t < 0.05: buckets["<50ms"] += 1
            elif t < 0.1: buckets["50-100ms"] += 1
            elif t < 0.2: buckets["100-200ms"] += 1
            elif t < 0.5: buckets["200-500ms"] += 1
            elif t < 1.0: buckets["500ms-1s"] += 1
            else: buckets[">1s"] += 1
        print(f"  Distribution: {dict(buckets)}")

    print(f"\n--- Counters ---")
    for name, count in sorted(_counters.items()):
        print(f"  {name}: {count}")

    print("\n" + "=" * 80)


def main():
    install_profiling()

    from paddleocr import PaddleOCRVL

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - Detailed GPU Profiling v2")
    print("=" * 60)

    from paddlex.inference.utils.misc import is_bfloat16_available, is_float16_available
    print(f"\n[ENV] is_bfloat16_available('gpu'): {is_bfloat16_available('gpu')}")
    print(f"[ENV] is_float16_available('gpu'): {is_float16_available('gpu')}")
    print(f"[ENV] PaddlePaddle: {paddle.__version__}, CUDA: {paddle.version.cuda}")
    print(f"[ENV] GPU: {paddle.device.cuda.get_device_name(0)}")

    print(f"\n[LOAD] Initializing pipeline...")
    t0 = time.time()
    pipeline = PaddleOCRVL(pipeline_version=PIPELINE_VERSION, device="gpu")
    t_load = time.time() - t0
    print(f"[OK] Pipeline initialized in {t_load:.1f}s")

    model = pipeline.paddlex_pipeline.vl_rec_model.infer
    print(f"\n[MODEL] dtype check (first 5 params):")
    for name, param in list(model.named_parameters())[:5]:
        print(f"  {name}: dtype={param.dtype}, place={param.place}")

    for img_name in IMAGES[:1]:
        img_path = IMAGE_DIR + "/" + img_name
        img_out_dir = OUTPUT_DIR / Path(img_name).stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PROC] Processing: {img_name}")
        _timers.clear()
        _counters.clear()
        _step_data.clear()
        _forward_call_data.clear()
        _block_count[0] = 0

        t0 = time.time()
        output = pipeline.predict(img_path)
        t_proc = time.time() - t0
        print(f"       Done in {t_proc:.1f}s")

        for res in output:
            try:
                res.save_to_json(save_path=str(img_out_dir))
            except Exception:
                pass
            try:
                res.save_to_markdown(save_path=str(img_out_dir))
            except Exception:
                pass

    print_report()


if __name__ == "__main__":
    main()
