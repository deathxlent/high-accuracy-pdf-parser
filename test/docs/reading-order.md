# 文档阅读顺序（Reading Order）获取指南

## 一、什么是阅读顺序

阅读顺序（Reading Order）是指文档中各个内容块（文本段落、标题、表格、图片、公式等）在逻辑上的排列次序。对于 PDF 文档来说，内容块的 **物理位置** 和 **逻辑阅读顺序** 并不总是一一对应，尤其是：

- **多栏布局**（报纸、杂志）：先读左栏从上到下，再读右栏从上到下
- **图文混排**：文字绕排图片时，阅读流可能被图片打断后继续
- **表格与标题**：表格的标题通常在表格之前，但物理位置可能在表格下方
- **页眉 / 页脚 / 脚注**：这些在物理上可能在页面顶部或底部，但不属于正文阅读流

> **正确获取阅读顺序是文档解析中最关键也最容易被忽视的环节。** 如果阅读顺序错乱，提取出来的文本会语义断裂，导致后续 RAG 检索到的内容与原文档含义南辕北辙。

## 二、整体方案：两阶段法

本项目采用 **两阶段法** 来获取阅读顺序：

```
第一阶段（Layout Detection）      第二阶段（Reading Order）
 ┌─────────────────────┐          ┌─────────────────────────┐
 │ 检测每个内容块的位置  │   ──►   │ 对所有内容块按阅读顺序排序 │
 │ 及类型（text/table/  │          │                         │
 │ picture/...）        │          │ 输入：bbox 列表 + 页面图片 │
 │                      │          │ 输出：每个 bbox 的 position │
 │ 输出：bbox 列表       │          │                         │
 └─────────────────────┘          └─────────────────────────┘
```

### 为什么不一步到位？

有开源方案（如 Surya 的 `LayoutPredictor`）可以同时输出 layout 类型和 reading order，但实际效果有限。本项目的策略是：

1. **用专门的 layout 模型（YOLOv10）**：它专为文档布局检测训练，能更好地区分 text、title、table、picture、formula 等类型
2. **将 layout 检测结果交给 reading order 模型（Surya ordering）**：由它根据内容块的位-置关系和视觉特征重新排序
3. **分离的好处**：可以分别替换升级 layout 模型和 reading order 模型，互不影响

## 三、核心代码解析

surya-ocr~=0.4.5

### 3.1 模型初始化

```python
from surya.model.ordering.model import load_model as order_load_model
from surya.model.ordering.processor import load_processor as order_load_processor

READING_ORDERS_MODEL = order_load_model()
READING_ORDERS_PROCESSOR = order_load_processor()
```

模型文件会自动下载到 HuggingFace 缓存目录（`~/.cache/huggingface/`）。但应该下载到本目录models下

### 3.2 Layout 检测

```python
def layout(image_path, model, page):
    """使用 YOLOv10 检测文档布局，返回内容块列表"""
    img = Image.open(image_path)
    width, height = img.size
    origin_result = model(source=image_path, conf=0.2, iou=0.8)
    detections = sv.Detections.from_ultralytics(origin_result[0])
    xyxys = detections.xyxy
    types = detections.data['class_name']

    layout_results = []
    for id in range(len(xyxys)):
        layout_results.append(LayoutResult(page, width, height, xyxys[id], types[id].lower()))

    # 对单列文本进行优化：将单列文本块的宽度扩展到页面宽度
    # （解决某些 layout 模型对单列文本的检测框过小的问题）
    height_ranges = [[r.bbox[1], r.bbox[3], r.type] for r in layout_results]
    single_col_indexes = _extract_single_text_column_index(height_ranges)
    for idx in single_col_indexes:
        layout_results[idx].bbox = [0, layout_results[idx].bbox[1], width, layout_results[idx].bbox[3]]

    return layout_results
```

**关键点**：

- YOLOv10 模型文件：`models/yolov10x_best.pt`（来自 [moured/YOLOv10-Document-Layout-Analysis](https://github.com/moured/YOLOv10-Document-Layout-Analysis)）
- 置信度阈值 `conf=0.2`, IoU 阈值 `iou=0.8`
- 检测类型：text、title、table、picture、formula、caption、section-header、list-item、footnote、page-header、page-footer 等

### 3.3 Reading Order 排序（核心）

```python
from surya.ordering import batch_ordering

def reading_orders(image_path, layout_results):
    """对 layout 检测结果进行阅读顺序排序"""
    try:
        model, processor = get_and_init_reading_order_models()

        bboxes = []
        img = Image.open(image_path)
        bbox_layout_map = {}

        # 收集所有 content block 的 bbox
        for layout in layout_results:
            bbox = [
                floor(layout.bbox[0]),
                floor(layout.bbox[1]),
                ceil(layout.bbox[2]),
                ceil(layout.bbox[3])
            ]
            bbox_layout_map[str(bbox)] = layout
            bboxes.append(bbox)

        # 调用 Surya batch_ordering
        order_boxes = batch_ordering([img], [bboxes], model, processor)

        # 将 Surya 返回的 position 映射回 layout_result
        for order_box in order_boxes[0].bboxes:
            bbox = order_box.bbox
            bbox_str = str([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            if bbox_str in bbox_layout_map:
                bbox_layout_map[bbox_str].reading_order = order_box.position

        # 按 reading_order 排序后返回
        result = sorted(bbox_layout_map.values(), key=lambda x: x.reading_order)
        return result

    except Exception as e:
        pass  # 如果 reading order 失败，回退到 layout 的原始顺序

    return layout_results
```

**关键逻辑**：

| 步骤            | 说明                                                      |
| ------------- | ------------------------------------------------------- |
| ① 收集 bbox     | 将 layout 检测结果中的每个内容块取整为 `[x1, y1, x2, y2]`              |
| ② 建立映射        | `bbox_layout_map[str(bbox)] = layout` —— 用字符串 key 做精确匹配 |
| ③ 调用 Surya    | `batch_ordering([img], [bboxes], model, processor)`     |
| ④ 回填 position | `order_box.position` 是 Surya 模型输出的阅读顺序编号（从 0 开始）        |
| ⑤ 排序输出        | `sorted(..., key=lambda x: x.reading_order)`            |

### 3.4 Surya `batch_ordering` API 详解

```python
order_boxes = batch_ordering(images, bbox_lists, model, processor)
```

**参数**：

| 参数           | 类型                      | 说明                                       |
| ------------ | ----------------------- | ---------------------------------------- |
| `images`     | `List[PIL.Image]`       | 页面图片列表，每页一张                              |
| `bbox_lists` | `List[List[List[int]]]` | 每页的 bbox 列表，每个 bbox 为 `[x1, y1, x2, y2]` |
| `model`      | `OrderingModel`         | Surya ordering 模型                        |
| `processor`  | `OrderingProcessor`     | Surya ordering 处理器                       |

**返回值**：

`order_boxes` 是一个列表（每页一个元素），每个元素包含：

| 字段                    | 类型                 | 说明                          |
| --------------------- | ------------------ | --------------------------- |
| `.bboxes`             | `List[OrderedBox]` | 排序后的 bbox 列表                |
| `.bboxes[i].bbox`     | `List[int]`        | `[x1, y1, x2, y2]`          |
| `.bboxes[i].position` | `int`              | 阅读顺序位置（0 表示第一个）             |
| `.bboxes[i].label`    | `str`              | 类型标签（仅 Surya 自己 layout 时才有） |

**注意**：Surya 的 `batch_ordering` 只负责**排序**，不重新检测 layout。它接收你提供的 bbox 列表，返回这些 bbox 的阅读顺序。

### 3.5 完整流程串联

```python
def _handle_single_page(doc, page_no, temp_path, saved_temp_debug=False):
    """处理单页 PDF"""

    # 1. 将 PDF 页渲染为图片
    image_path = f'{temp_path}{page_no}.png'

    # 2. 判断是否为扫描版 PDF
    page = doc[page_no]
    is_scanned_page = is_scanned_pdf_page(page)

    # 3. Layout 检测（YOLOv10）
    layout_results = layout(image_path, get_and_init_yolox(), page_no)

    # 4. Reading Order 排序（Surya）
    reading_orders_result = reading_orders(image_path, layout_results)

    # 5. 按阅读顺序依次提取内容
    for layout_result in reading_orders_result:
        extract_text_content(layout_result, ...)

    return result
```

### 3.6 可视化调试

项目提供了 `draw_bbox_to_images()` 函数，可以将 reading order 结果可视化：

```python
def draw_bbox_to_images(image_path, save_dir, layout_results):
    """在图片上绘制 bbox 和 reading order 编号"""
    # ...
    for layout_result in layout_results:
        if layout_result.reading_order is not None:
            type_text = f"{layout_result.reading_order}--{type_text}"
        _draw_text(text_position, type_text, draw)
```

启用 `--debug` 参数后，会在输出目录生成带 `_marked_order.png` 后缀的标注图，每个内容块上会显示 `序号--类型`，例如 `0--title`、`1--text`、`2--table`、`3--text`。

## 四、Surya 替代方案：一步到位

如果不想分两步（layout + ordering），也可以直接用 Surya 的 `LayoutPredictor` 同时完成 layout 检测和 reading order：

```python
from PIL import Image
from surya.layout import LayoutPredictor

image = Image.open("page.png")
layout_predictor = LayoutPredictor()

predictions = layout_predictor([image])

# predictions[0].bboxes 中每个 bbox 包含：
#   - bbox: [x1, y1, x2, y2]
#   - position: 阅读顺序
#   - label: 类型（Text, Title, Table, Picture 等）
#   - top_k: 各类别的置信度

for bbox in predictions[0].bboxes:
    print(f"Position {bbox.position}: {bbox.label} at {bbox.bbox}")
```

**优缺点对比**：

| 方案                            | 优点                           | 缺点                     |
| ----------------------------- | ---------------------------- | ---------------------- |
| YOLOv10 + Surya ordering（本项目） | layout 类型更准确，可替换任意 layout 模型 | 流程更复杂，需要处理两次           |
| Surya LayoutPredictor         | 简单，一次调用                      | layout 类型识别在某些场景不如专用模型 |

## 五、完整运行流程

```text
PDF 文件
   │
   ▼
┌──────────────────────┐
│ PyMuPDF 渲染为图片    │  每个 page_no.png
│ (scale=2)            │
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│ YOLOv10 Layout 检测   │  输出 LayoutResult 列表
│ (conf=0.2, iou=0.8)  │  (bbox + type + page)
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│ Surya batch_ordering  │  为每个 LayoutResult 赋值
│ (图片 + bbox 列表)    │  reading_order 字段
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│ 按 reading_order 排序  │  得到正确阅读顺序的内容块
└──────────────────────┘
   │
   ▼
┌──────────────────────┐
│ 按类型提取内容         │
│ text → PyMuPDF/Surya OCR│
│ table → TableTransformer│
│ picture → 保存图片     │
│ formula → 保存公式图片  │
└──────────────────────┘
   │
   ▼
┌──────────────────────────────────┐
│ 输出 RAG 友好的格式                │
│ text.json / table.json / image.json│
│ total.json / {file_name}.md       │
└──────────────────────────────────┘
```

## 六、模型来源

| 模型                | 来源                                                                                                                                              | 用途       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| YOLOv10x          | [moured/YOLOv10-Document-Layout-Analysis](https://github.com/moured/YOLOv10-Document-Layout-Analysis)                                           | 文档布局检测   |
| Surya Ordering    | [VikParuchuri/surya](https://github.com/VikParuchuri/surya)                                                                                     | 阅读顺序分析   |
| Surya OCR         | [VikParuchuri/surya](https://github.com/VikParuchuri/surya)                                                                                     | OCR 文字识别 |
| Table Transformer | [microsoft/table-transformer-structure-recognition-v1.1-pub](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-pub) | 表格结构识别   |

## 七、常见问题

### Q1: 为什么不用 Surya 自带的 layout + ordering？

Surya 的 `LayoutPredictor` 输出的 label 类型不如 YOLOv10 准确。本项目需要精确区分 `title`、`section-header`、`caption`、`picture`、`table`、`formula` 等类型来做差异化处理（如表格需要额外用 Table Transformer 解析内部结构），因此先用 YOLOv10 检测类型，再用 Surya 排序。

### Q2: 多栏版面效果如何？

对于 ≤3 栏的普通版面效果可接受。对于复杂的杂志/报纸版面，YOLOv10 的 layout 检测本身可能出错（把两栏内容识别成一个 text block），导致 Surya 排序时拿到的是错误粒度的 bbox。

### Q3: Reading Order 失败了怎么办？

`reading_orders()` 函数有异常保护：如果 Surya 调用失败（`except Exception`），会直接返回原始的 `layout_results`（即按 YOLOv10 的检测顺序），不会让流程中断。

### Q4: 如何升级到更好的 reading order 模型？

在 `get_and_init_reading_order_models()` 中替换模型加载逻辑即可，上游接口只要兼容 `batch_ordering(images, bboxes, model, processor)` 就能无缝接入。

### Q5: 读取顺序从 0 开始吗？

是的，Surya 返回的 `position` 从 0 开始，0 表示页面上第一个要读的内容块。

## 八、推荐的替代方案

如果需要更好的阅读顺序识别效果，可以考虑：

| 方案            | 说明                                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| **PaddleOCR** | 英文/中文场景下 OCR 效果更好，可结合其 layout 分析功能                                                                                |
| **Marker**    | [VikParuchuri/marker](https://github.com/VikParuchuri/marker) — Surya 作者的另一项目，端到端 PDF 转 markdown，内置 reading order |
| **DocTR**     | 端到端文档识别，支持布局分析                                                                                                    |
| **Nougat**    | Meta 的 PDF 理解模型，可输出 markdown                                                                                      |
| **商业方案**      | hellorag.com 的商业版在 reading order 上有显著提升                                                                           |
