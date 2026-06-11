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


_perf_timers = defaultdict(list)
_perf_counters = defaultdict(int)
_step_times = []
_kv_cache_sizes = []
_forward_phase_times = []
_generate_step_details = []
_cpu_transfer_count = [0]
_prod_call_count = [0]


def _record_timer(name, duration):
    _perf_timers[name].append(duration)


def _record_counter(name, count=1):
    _perf_counters[name] += count


def print_perf_report():
    print("\n" + "=" * 80)
    print("  PERFORMANCE PROFILING REPORT")
    print("=" * 80)

    print("\n--- Timer Summary ---")
    for name, times in sorted(_perf_timers.items()):
        total = sum(times)
        avg = total / len(times) if times else 0
        count = len(times)
        print(f"  {name}: total={total:.3f}s, count={count}, avg={avg:.4f}s, min={min(times):.4f}s, max={max(times):.4f}s")

    print("\n--- Counter Summary ---")
    for name, count in sorted(_perf_counters.items()):
        print(f"  {name}: {count}")

    if _step_times:
        print(f"\n--- Generation Steps ---")
        print(f"  Total steps: {len(_step_times)}")
        print(f"  Total time: {sum(_step_times):.3f}s")
        avg_step = sum(_step_times) / len(_step_times) if _step_times else 0
        print(f"  Avg time/step: {avg_step:.4f}s")
        if _step_times:
            print(f"  First 5 steps: {[f'{t:.4f}s' for t in _step_times[:5]]}")
            print(f"  Last 5 steps: {[f'{t:.4f}s' for t in _step_times[-5:]]}")
            buckets = defaultdict(int)
            for t in _step_times:
                if t < 0.01:
                    buckets["<10ms"] += 1
                elif t < 0.05:
                    buckets["10-50ms"] += 1
                elif t < 0.1:
                    buckets["50-100ms"] += 1
                elif t < 0.5:
                    buckets["100-500ms"] += 1
                elif t < 1.0:
                    buckets["500ms-1s"] += 1
                else:
                    buckets[">1s"] += 1
            print(f"  Step time distribution: {dict(buckets)}")

    if _forward_phase_times:
        print(f"\n--- Forward Phase Breakdown (first call = prefill, rest = decode) ---")
        for i, (phase, dur) in enumerate(_forward_phase_times[:10]):
            print(f"  Forward #{i}: {phase} = {dur:.4f}s")
        if len(_forward_phase_times) > 10:
            print(f"  ... ({len(_forward_phase_times)} total forwards)")

    if _generate_step_details:
        print(f"\n--- Generate Step Details (sample) ---")
        for i, detail in enumerate(_generate_step_details[:5]):
            print(f"  Step {i}: {detail}")
        if len(_generate_step_details) > 5:
            last_detail = _generate_step_details[-1]
            print(f"  Step {len(_generate_step_details)-1}: {last_detail}")

    if _kv_cache_sizes:
        print(f"\n--- KV Cache Sizes ---")
        for i, (step, size_info) in enumerate(_kv_cache_sizes[:5]):
            print(f"  Step {step}: {size_info}")
        if len(_kv_cache_sizes) > 5:
            print(f"  Step {_kv_cache_sizes[-1][0]}: {_kv_cache_sizes[-1][1]}")

    print(f"\n--- CPU Transfer Tracking ---")
    print(f"  paddle.Tensor.prod GPU->CPU calls: {_prod_call_count[0]}")
    print(f"  .cpu() calls detected in forward: {_cpu_transfer_count[0]}")

    print("\n" + "=" * 80)


def install_profiling_patches():
    from paddlex.inference.models.doc_vlm.predictor import DocVLMLocalPredictor
    from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl._paddleocr_vl import (
        PaddleOCRVLForConditionalGeneration,
    )
    from paddlex.inference.models.common.transformers.generation.utils import GenerationMixin

    orig_process = DocVLMLocalPredictor.process

    def profiled_process(self, data, **kwargs):
        t0 = time.time()
        data = self.processor.preprocess(data, min_pixels=kwargs.get('min_pixels'), max_pixels=kwargs.get('max_pixels'))
        t_preprocess = time.time() - t0
        _record_timer("DocVLMPredictor.preprocess", t_preprocess)

        t0 = time.time()
        data = self._switch_inputs_to_device(data)
        t_switch = time.time() - t0
        _record_timer("DocVLMPredictor._switch_inputs_to_device", t_switch)

        if data and isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, paddle.Tensor):
                    print(f"  [PROF] Input '{k}': shape={v.shape}, dtype={v.dtype}, place={v.place}")

        generate_kwargs = {}
        if kwargs.get('max_new_tokens') is not None:
            generate_kwargs["max_new_tokens"] = kwargs['max_new_tokens']
        else:
            from paddlex.inference.models.doc_vlm.constants import PADDLEOCR_VL_MAX_NEW_TOKENS
            generate_kwargs["max_new_tokens"] = PADDLEOCR_VL_MAX_NEW_TOKENS
        if kwargs.get('use_cache') is not None:
            generate_kwargs["use_cache"] = kwargs['use_cache']

        t0 = time.time()
        from paddlex.utils.device import TemporaryDeviceChanger
        with TemporaryDeviceChanger(self.device):
            preds = self.infer.generate(data, **generate_kwargs)
        t_generate = time.time() - t0
        _record_timer("DocVLMPredictor.generate", t_generate)

        t0 = time.time()
        postprocess_kwargs = {}
        if kwargs.get('skip_special_tokens') is not None:
            postprocess_kwargs["skip_special_tokens"] = kwargs['skip_special_tokens']
        preds = self.processor.postprocess(preds, **postprocess_kwargs)
        t_postprocess = time.time() - t0
        _record_timer("DocVLMPredictor.postprocess", t_postprocess)

        result_dict = self._format_result_dict(preds, data)
        return result_dict

    DocVLMLocalPredictor.process = profiled_process

    orig_forward = PaddleOCRVLForConditionalGeneration.forward

    def profiled_forward(self, *args, **kwargs):
        t0 = time.time()
        has_pixel_values = kwargs.get('pixel_values') is not None
        has_past = kwargs.get('past_key_values') is not None

        if has_pixel_values and not has_past:
            phase = "prefill(with_vision)"
        elif has_past:
            phase = "decode"
        else:
            phase = "prefill(text_only)"

        result = orig_forward(self, *args, **kwargs)

        t_forward = time.time() - t0
        _record_timer(f"PaddleOCRVL.forward({phase})", t_forward)
        _forward_phase_times.append((phase, t_forward))

        if has_past and isinstance(result, paddle.Tensor) is False:
            try:
                from paddlex.inference.models.common.transformers.transformers.model_outputs import ModelOutput
                if isinstance(result, ModelOutput) and result.past_key_values is not None:
                    pkv = result.past_key_values
                    seq_len = pkv[0][0].shape[1] if pkv and pkv[0] else 0
                    step_num = len(_step_times)
                    _kv_cache_sizes.append((step_num, f"seq_len={seq_len}, num_layers={len(pkv)}"))
            except Exception:
                pass

        return result

    PaddleOCRVLForConditionalGeneration.forward = profiled_forward

    orig_generate = PaddleOCRVLForConditionalGeneration.generate

    def profiled_generate(self, inputs, **kwargs):
        print(f"\n  [PROF] PaddleOCRVL.generate() called")
        print(f"  [PROF]   max_new_tokens: {kwargs.get('max_new_tokens', 'default')}")
        print(f"  [PROF]   use_cache: {kwargs.get('use_cache', 'default')}")
        for k, v in inputs.items():
            if isinstance(v, paddle.Tensor):
                print(f"  [PROF]   input '{k}': shape={v.shape}, dtype={v.dtype}, place={v.place}")

        t0 = time.time()
        result = orig_generate(self, inputs, **kwargs)
        t_total = time.time() - t0
        _record_timer("PaddleOCRVL.generate(total)", t_total)

        if isinstance(result, paddle.Tensor):
            print(f"  [PROF]   generated tokens: {result.shape[-1]}")
            print(f"  [PROF]   output place: {result.place}")
        elif isinstance(result, tuple):
            ids = result[0]
            print(f"  [PROF]   generated tokens: {ids.shape[-1]}")
            print(f"  [PROF]   output place: {ids.place if isinstance(ids, paddle.Tensor) else 'N/A'}")

        return result

    PaddleOCRVLForConditionalGeneration.generate = profiled_generate

    orig_greedy_search = GenerationMixin.greedy_search

    def profiled_greedy_search(self, input_ids, logits_processors, max_length, pad_token_id, eos_token_id,
                                stopping_criteria=None, streamer=None, fast_ptq_sampling=False,
                                trunc_input=True, synced_gpus=False, **model_kwargs):
        batch_size, cur_len = input_ids.shape
        origin_len = cur_len
        print(f"\n  [PROF] greedy_search started: input_len={origin_len}, max_length={max_length}")
        print(f"  [PROF]   input_ids place: {input_ids.place}")
        if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
            print(f"  [PROF]   attention_mask shape: {model_kwargs['attention_mask'].shape}, place: {model_kwargs['attention_mask'].place}")
        if "past_key_values" in model_kwargs:
            pkv = model_kwargs["past_key_values"]
            print(f"  [PROF]   past_key_values present: {pkv is not None}")

        step_count = 0
        step_t0 = time.time()

        from paddlex.inference.models.common.transformers.generation.utils import (
            LogitsProcessorList, StoppingCriteriaList, get_unfinished_flag
        )
        from paddlex.inference.models.common.transformers.generation.stopping_criteria import validate_stopping_criteria

        logits_processors = logits_processors if logits_processors is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        unfinished_flag = paddle.full([batch_size, 1], True, dtype="bool")
        scores = paddle.full([batch_size, 1], 0.0, dtype=paddle.get_default_dtype())
        generate_end = False

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
                    unfinished_flag,
                    next_tokens,
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
            _step_times.append(step_time)
            step_count += 1

            if step_count <= 3 or step_count % 50 == 0 or generate_end:
                pkv_info = "None"
                if "past_key_values" in model_kwargs and model_kwargs["past_key_values"] is not None:
                    pkv = model_kwargs["past_key_values"]
                    pkv_info = f"layers={len(pkv)}, seq_len={pkv[0][0].shape[1] if pkv and pkv[0] else '?'}"
                detail = f"time={step_time:.4f}s, cur_len={cur_len}, kv_cache={pkv_info}"
                _generate_step_details.append(detail)
                if step_count <= 3 or step_count % 100 == 0 or generate_end:
                    print(f"  [PROF]   step {step_count}: {detail}")

            if generate_end and not synced_gpus:
                break

            model_kwargs = self.update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            if fast_ptq_sampling:
                break

        print(f"  [PROF] greedy_search finished: {step_count} steps, total={sum(_step_times):.3f}s")
        if streamer is not None:
            streamer.end()

        return input_ids[:, origin_len:] if trunc_input else input_ids, scores

    GenerationMixin.greedy_search = profiled_greedy_search

    orig_prod = paddle.Tensor.prod
    def counting_prod(self, *args, **kwargs):
        if self.place.is_gpu_place():
            _prod_call_count[0] += 1
        return orig_prod(self, *args, **kwargs)
    paddle.Tensor.prod = counting_prod

    print("[PROF] Performance profiling patches installed.")


BASE_DIR = Path(__file__).parent.resolve()
OUTPUT_DIR = BASE_DIR / "output_paddleocr_vl_gpu"
IMAGE_DIR = str(Path(__file__).parent.parent.resolve() / "tmp/42e59745cdb54b6fb2c635d7c11dbd43")
IMAGES = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])

PIPELINE_VERSION = "v1.6"


def main():
    install_profiling_patches()

    from paddleocr import PaddleOCRVL

    print("\n" + "=" * 60)
    print("  PaddleOCR-VL-1.6 - GPU Performance Profiling")
    print("=" * 60)
    print(f"[INFO] Images to process: {len(IMAGES)}")

    print(f"\n[LOAD] Initializing PaddleOCRVL pipeline with GPU ...")
    t0 = time.time()
    pipeline = PaddleOCRVL(
        pipeline_version=PIPELINE_VERSION,
        device="gpu",
    )
    t_load = time.time() - t0
    print(f"[OK] Pipeline initialized (GPU) in {t_load:.1f}s")

    model = pipeline.paddlex_pipeline.vl_rec_model.infer
    print(f"\n[INFO] Model dtype: {model.dtype if hasattr(model, 'dtype') else 'N/A'}")
    params_iter = model.parameters()
    first_param = next(iter(params_iter))
    print(f"[INFO] Model device: place={first_param.place}, dtype={first_param.dtype}")
    for name, param in model.named_parameters():
        print(f"[INFO]   {name}: shape={param.shape}, dtype={param.dtype}, place={param.place}")
        break

    print(f"\n[INFO] Visual encoder fp32 check:")
    if hasattr(model, 'visual'):
        for name, param in model.visual.named_parameters():
            print(f"[INFO]   visual.{name}: dtype={param.dtype}, place={param.place}")
            break

    for img_name in IMAGES:
        img_path = IMAGE_DIR + "/" + img_name
        img_out_dir = OUTPUT_DIR / Path(img_name).stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[PROC] Processing: {img_name}")
        _step_times.clear()
        _forward_phase_times.clear()
        _generate_step_details.clear()
        _kv_cache_sizes.clear()
        _prod_call_count[0] = 0
        _cpu_transfer_count[0] = 0

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

    print_perf_report()


if __name__ == "__main__":
    main()
