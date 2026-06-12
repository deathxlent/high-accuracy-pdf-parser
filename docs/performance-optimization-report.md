# PaddleOCR-VL-1.6 性能优化分析报告

> 基于 Quadro P1000 (SM 6.1 Pascal, 4GB VRAM) 基准测试 (475.8s) 和 profiling 数据

## 基准测试概况

| 指标 | 值 |
|------|-----|
| 模型 | PaddleOCR-VL-1.6 (PaddleX, paddlex-4.3.10) |
| GPU | Quadro P1000 (Pascal, SM 6.1, 4GB VRAM) |
| 精度 | fp32 (全量) |
| 总推理时间 | **475.8s** (~8 分钟/页) |
| VRAM 占用 | ~3.6GB/4GB |
| 模型架构 | SigLIP VE (27层) + Projector + Qwen2 LLM (0.5B, 24层) |

### 耗时分布

| 模块 | 耗时 | 占比 | 说明 |
|------|------|------|------|
| **VE (视觉编码器)** | **~220s** | ~46% | SigLIP 27 transformer layers — 最重 |
| **LLM (文本生成)** | ~150s | ~32% | Qwen2-0.5B 的 decode 阶段 |
| **预处理 & 版面检测** | ~50s | ~10% | PP-DocLayoutV3 检测 + 图像裁剪 |
| **OCR 后处理 & 合并** | ~55s | ~12% | 区域结果汇总，Markdown 生成 |

### 根因分析

**核心瓶颈：Quadro P1000 无 Tensor Cores，且全模型运行在 fp32。**

- KV cache 本身工作正常（profile_full_page.py 验证：每 block 仅 1 次 VE 调用）
- `max_new_tokens` 从 8192 降至 1024 已无明显影响
- FP32 对计算单元密集度的利用率低（Pascal 的 fp32 CUDA core 吞吐有限）
- 模型 ~3.6GB 逼近 VRAM 上限，限制了任何批处理能力

---

## 优化方案

### 方案 1：fp16 AMP 推理 ★★★★★ (推荐)

**原理：** 利用 Pascal SM 6.1 原生支持的 fp16 计算，将模型大部分操作迁移到 fp16，内存带宽需求减半，计算吞吐翻倍。

**关键信息：**
- PaddlePaddle 3.2.1 + CUDA 11.8 完全支持 `paddle.amp` 模块
- Quadro P1000 支持原生 fp16（不支持 bfloat16）
- 当前 `_keep_in_fp32_modules = ["visual", "mlp_AR"]` 强制将 VE 和 projector 保留在 fp32，需覆盖

**实施方式：**

```python
import paddle
from paddlex import PaddleOCRVL

pipeline = PaddleOCRVL(pipeline_version="v1.6", device="gpu")
model = pipeline.paddlex_pipeline.vl_rec_model.infer

# 1. 覆盖 fp32 强制保留配置 — 需要修改 paddlex 源码
#    model.py 中 _keep_in_fp32_modules = []  # 清空

# 2. 用 AMP decorate 包裹模型
model = paddle.amp.decorate(models=model, level='O2', master_weight=True)

# 3. 推理时使用 auto_cast
with paddle.amp.auto_cast(level='O2', dtype='float16'):
    output = pipeline.predict(img_path)
```

**预期效果：**

| 指标 | fp32 (当前) | fp16 (预期) | 提升 |
|------|------------|------------|------|
| VE 时间 | ~220s | ~70-100s | 2-3x |
| LLM 时间 | ~150s | ~75-100s | 1.5-2x |
| VRAM 占用 | 3.6GB | ~1.8GB | 50% |
| **总时间** | **~480s** | **~200-300s** | **1.6-2.4x** |
| 精度损失 | — | <0.5% | 可忽略 |

**风险：**
- `gelu_pytorch_tanh` 激活函数在 fp16 下有小概率下溢
- VE 和 projector 的一些 layer 可能在 fp16 下数值敏感（但通常 torch 模型可用 fp16 无损）
- 需要修改 paddlex 内部依赖库源码（`_libs/paddlex/paddlex-paddleocr-vl/paddlex/paddlex/pipelines/PaddleOCR-VL-1.6/model.py`）

---

### 方案 2：降低图片分辨率 ★★★☆☆ (中等推荐)

**原理：** 减小传入 VE 的图像区域的 patch 数，直接缩短 transformer attention 计算时间（复杂度 O(n²)）。

**背景数据：**

| 区域类型 | 典型 patch 数 | VE 耗时 |
|---------|--------------|--------|
| 标准 384×384 区域 | ~729 (27×27) | ~5.5s |
| 最大区域 (块 14) | ~2000+ | 135.7s |

**实施方式：**

```python
# 在 predict 时传入 max_pixels 参数
output = pipeline.predict(
    img_path,
    max_pixels=720*28*28  # 默认 ~1M (1280*28*28)，此处降低 ~44%
)
```

或修改 `_siglip.py` processor 中 `_preprocess` 方法的 `max_pixels` 阈值。

**预期效果：**

| 降低幅度 | VE 时间估计 | 总时间估计 | 精度影响 |
|---------|------------|-----------|---------|
| 不降低 (现状) | ~220s | ~480s | 基线 |
| 降低 30% | ~155s | ~410s | 轻微 |
| 降低 50% | ~110s | ~370s | 中等 |
| 降低 70% | ~65s | ~320s | 较大 |

**风险：**
- 小文字、密集表格、复杂公式的识别精度会下降
- SigLIP 的 `interpolate_pos_encoding=True` 支持分辨率外推，但非无限缩放
- 对大字为主的页面影响小，对小字密集页面影响大

---

### 方案 3：多页并行处理 ★★☆☆☆ (条件推荐)

**原理：** 对包含多页的文档（如扫描 PDF），同时启动多个 pipeline 实例并行推理。

**背景：**
- pipeline 内部按 block 串行处理（block 间无依赖）
- 同一页面内的 block 不并行（串行更稳定且内存可控）
- 不同页面之间完全独立，可并行

**实施方式：**

```python
from concurrent.futures import ThreadPoolExecutor

images = ["page_1.jpg", "page_2.jpg", "page_3.jpg", "page_4.jpg"]

with ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(pipeline.predict, images))
```

**VRAM 限制（关键瓶颈）：**

| 精度 | 单实例 VRAM | P1000 最大并行数 |
|------|------------|----------------|
| fp32 | ~3.6GB | 1 (无法并行) |
| fp16 | ~1.8GB | 2 |

**预期效果（需先做 fp16）：**

| 页数 | 串行 | 2 路并行 | 加速比 |
|------|------|---------|-------|
| 1 页 | ~250s (fp16) | — | 无变化 |
| 4 页 | ~1000s | ~500s | 2x |
| 10 页 | ~2500s | ~1250s | 2x |

**风险：**
- Python GIL 限制：`ThreadPoolExecutor` 在 CPU 密集型操作中受限，但 paddle GPU 调用会释放 GIL
- 更推荐 `multiprocessing` 避免 GIL，但会增加内存开销
- 多实例同时推理可能因 GPU 显存竞争导致 OOM

---

## 实施路线图（推荐优先级）

```
第一步: fp16 AMP 推理 (方案 1)
  └─ 最大收益，最小风险
  └─ 总时间: 480s → 200-300s
  └─ 为后续批处理释放 VRAM
         ↓
第二步: 多页并行 (方案 3)
  └─ 仅对多页文档有效
  └─ 需依赖第一步 (fp16 才有 VRAM 余量)
  └─ 多页吞吐: 2x 提升
         ↓
第三步: 降低分辨率 (方案 2, 可选)
  └─ 如果精度仍然可接受
  └─ 在 fp16 基础上再降 20-30%
  └─ 总时间可进一步压至 ~150-200s
```

## 结论

**单页场景的核心瓶颈是 fp32 精度 + Pascal GPU 缺少 Tensor Cores。** 三个优化方案中，fp16 AMP 推理的投入产出比最高，改动相对可控（仅需修改 paddlex 内部 model.py 的 `_keep_in_fp32_modules` 配置 + 添加 `auto_cast` 上下文），预估可将单页处理时间从约 8 分钟降至 3-5 分钟。

最彻底的解决方案仍是升级到支持 Tensor Cores 的 GPU（Volta SM 7.0+），届时 fp16 可提供 4-8x 的矩阵乘吞吐，预估总时间可降至 **30-60 秒/页**。
