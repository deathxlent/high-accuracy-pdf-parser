"""
Layout Model 独立测试脚本
=======================
用法:
    python scripts/test_layout.py <图片路径> [--output 输出路径]

功能:
    1. 加载项目中已配置的 YOLO layout 模型
    2. 对输入图片进行版面检测
    3. 打印模型返回的完整详细信息
    4. 将检测框标注在原图上并保存为新图片
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# 确保项目根目录在 sys.path 中，方便导入 backend 模块
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("layout_test")

# ── 颜色方案（BGR 格式，与 OpenCV 一致）────────────────────────────────────
CATEGORY_COLORS = {
    "Caption":        (255, 191, 0),      # 深天蓝
    "Footnote":       (128, 128, 128),    # 灰色
    "Formula":        (0, 165, 255),      # 橙色
    "List-item":      (0, 255, 255),      # 黄色
    "Page-footer":    (128, 0, 128),      # 紫色
    "Page-header":    (255, 0, 255),      # 粉红
    "Picture":        (0, 255, 0),        # 绿色
    "Section-header": (255, 128, 0),      # 青蓝
    "Table":          (255, 0, 0),        # 红色
    "Text":           (255, 255, 0),      # 青色（原为白色，浅色背景看不见）
    "Title":          (0, 0, 255),        # 蓝色
}
DEFAULT_COLOR = (200, 200, 200)  # 未知类别 -> 浅灰


class LayoutVisualizer:
    """版面检测结果可视化器"""

    def __init__(self):
        self.model = None

    def load_model(self):
        """加载 YOLO layout 模型（复用 backend 逻辑）"""
        from backend.services.layout_service import _get_model
        logger.info("正在加载 YOLO layout 模型 ...")
        self.model = _get_model()
        logger.info("模型加载完成。")

    def detect(self, image_path: str) -> list[dict]:
        """对图片执行版面检测，同时返回 YOLO 原生 results 用于打印详情"""
        from ultralytics import YOLO
        from backend.config import YOLO_IMG_SIZE

        model: YOLO = self.model
        logger.info(f"正在检测: {image_path}")
        results = model(image_path, imgsz=YOLO_IMG_SIZE, verbose=False)

        if not results:
            logger.warning("模型未返回任何结果！")
            return [], None

        result = results[0]
        elements = []

        # ── 打印原生结果详情 ────────────────────────────────────────────
        print("=" * 72)
        print("📦  YOLO 模型原生返回信息")
        print("=" * 72)

        # 1) 图片信息
        orig_shape = result.orig_shape  # (H, W)
        print(f"\n📷 输入尺寸 (H×W): {orig_shape[0]} × {orig_shape[1]}")
        if hasattr(result, "speed"):
            print(f"⚡ 推理耗时: {result.speed}")

        # 2) 检测框信息
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            logger.warning("未检测到任何目标。")
            return [], result

        print(f"\n🔍 共检测到 {len(boxes)} 个版面元素:")
        print("-" * 72)
        print(f"{'#':>3} | {'类别':<16} | {'置信度':>8} | {'x1':>7} {'y1':>7} {'x2':>7} {'y2':>7} | {'面积占比':>8}")
        print("-" * 72)

        from backend.services.layout_service import YOLO_CATEGORY_MAP

        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = YOLO_CATEGORY_MAP.get(cls_id, f"Unknown-{cls_id}")

            x1, y1, x2, y2 = xyxy
            area_ratio = ((x2 - x1) * (y2 - y1)) / (orig_shape[0] * orig_shape[1]) * 100

            print(f"{i+1:>3} | {cls_name:<16} | {conf:>7.3f} | "
                  f"{x1:>7.1f} {y1:>7.1f} {x2:>7.1f} {y2:>7.1f} | {area_ratio:>7.2f}%")

            elements.append({
                "element_type": cls_name,
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": conf,
                "reading_order": -1,
            })

        print("-" * 72)

        # 3) 按类别统计
        from collections import Counter
        type_counts = Counter(e["element_type"] for e in elements)
        print(f"\n📊 按类别统计:")
        for t, cnt in type_counts.most_common():
            print(f"    {t:<16} : {cnt} 个")

        # 4) 如有分类置信度详情 (YOLO 的 cls prob)
        if hasattr(boxes, "cls") and len(boxes.cls.shape) > 1 and boxes.cls.shape[1] > 1:
            print(f"\n📋 分类置信度分布 (所有类别):")
            for i in range(len(boxes)):
                cls_probs = boxes.cls[i].cpu().numpy()
                top3_idx = cls_probs.argsort()[-3:][::-1]
                print(f"   目标 #{i+1} Top-3:")
                for idx in top3_idx:
                    cat_name = YOLO_CATEGORY_MAP.get(idx, f"Unknown-{idx}")
                    print(f"      {cat_name:<16} : {cls_probs[idx]:.4f}")

        print("=" * 72)
        return elements, result

    def annotate(self, image_path: str, elements: list[dict], output_path: str):
        """将检测框 + 标签绘制在图片上，保存为 output_path"""
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.error("需要 opencv-python，请执行: pip install opencv-python")
            sys.exit(1)

        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图片: {image_path}")
            sys.exit(1)

        H, W = img.shape[:2]

        for idx, elem in enumerate(elements):
            x1, y1, x2, y2 = elem["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls_name = elem["element_type"]
            conf = elem["confidence"]

            color = CATEGORY_COLORS.get(cls_name, DEFAULT_COLOR)

            # ── 画检测框 ──────────────────────────────────────────────
            thickness = max(2, int(min(W, H) / 500))
            # 外边框（深色描边），确保在任何背景下都可见
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness + 2)
            # 内边框（类别色）
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # ── 标签文字 ──────────────────────────────────────────────
            label = f"{cls_name}  {conf:.2%}"
            font_scale = max(0.25, min(W, H) / 2000)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
            )

            # 标签背景（防止文字被内容遮挡）
            label_y1 = max(y1 - th - baseline - 4, 0)
            cv2.rectangle(
                img,
                (x1, label_y1),
                (x1 + tw + 6, y1),
                color,
                -1,  # 填充
            )
            # 文字（白色或黑色取决于背景亮度）
            text_color = (0, 0, 0) if np.mean(color) > 127 else (255, 255, 255)
            cv2.putText(
                img, label,
                (x1 + 3, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                max(1, thickness - 1),
                cv2.LINE_AA,
            )

            # ── 序号标注 ──────────────────────────────────────────────
            num_label = f"#{idx + 1}"
            cv2.putText(
                img, num_label,
                (x1 + 3, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                max(1, thickness - 2),
                cv2.LINE_AA,
            )

        # ── 图例（纯文字，不加背景框）────────────────────────────────
        legend_x = 15
        legend_y = 25
        cv2.putText(img, "图例", (legend_x, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for i, (cat, clr) in enumerate(CATEGORY_COLORS.items()):
            y = legend_y + (i + 1) * 20
            cv2.rectangle(img, (legend_x, y + 2), (legend_x + 14, y + 14), clr, -1)
            cv2.putText(img, cat, (legend_x + 20, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, img)
        logger.info(f"标注结果已保存: {output_path}")

    def run(self, image_path: str, output_path: str | None = None) -> list[dict]:
        """一键执行：加载模型 → 检测 → 打印 → 标注保存"""
        if self.model is None:
            self.load_model()

        elements, raw_result = self.detect(image_path)

        # 输出 JSON 格式（便于程序消费）
        print(f"\n💾 JSON 格式输出:")
        print(json.dumps(elements, ensure_ascii=False, indent=2))

        if elements and output_path:
            self.annotate(image_path, elements, output_path)

        return elements


def main():
    parser = argparse.ArgumentParser(
        description="Layout 模型独立测试 — 检测图片中的版面元素并可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_layout.py sample.jpg
  python scripts/test_layout.py sample.jpg --output result.jpg
  python scripts/test_layout.py sample.jpg -o result.jpg --no-display
        """,
    )
    parser.add_argument("image", type=str, help="输入图片路径")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="标注图片输出路径 (默认: 输入文件名加上 _annotated 后缀)")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        logger.error(f"文件不存在: {img_path}")
        sys.exit(1)

    # 默认输出路径
    output_path = args.output
    if output_path is None:
        stem = img_path.stem
        output_path = str(img_path.with_name(f"{stem}_annotated{img_path.suffix}"))

    visualizer = LayoutVisualizer()
    visualizer.run(str(img_path), output_path)

    print(f"\n✅ 完成！标注图片: {output_path}")


if __name__ == "__main__":
    main()
