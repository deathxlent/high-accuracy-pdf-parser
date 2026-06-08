import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from backend.config import MODELS_DIR, YOLO_MODEL_REPO, YOLO_MODEL_FILE, YOLO_IMG_SIZE

logger = logging.getLogger(__name__)

_model = None


def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def remove_overlapping_elements(elements: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    if len(elements) <= 1:
        return elements

    non_text_types = {t for t in YOLO_CATEGORY_MAP.values() if t != "Text"}

    def should_keep(elem, other):
        iou = _compute_iou(elem["bbox"], other["bbox"])
        if iou < iou_threshold:
            return True

        if elem["element_type"] == "Text" and other["element_type"] != "Text":
            return False
        elif elem["element_type"] != "Text" and other["element_type"] == "Text":
            return True
        else:
            return elem["confidence"] >= other["confidence"]

    kept = []
    for i, elem in enumerate(elements):
        keep = True
        for j, other in enumerate(elements):
            if i == j:
                continue
            if not should_keep(elem, other):
                keep = False
                break
        if keep:
            kept.append(elem)

    return kept

YOLO_CATEGORY_MAP = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def _get_model() -> YOLO:
    global _model
    if _model is not None:
        return _model

    model_path = MODELS_DIR / YOLO_MODEL_FILE
    if not model_path.exists():
        logger.info("Downloading YOLO26m model from HuggingFace Mirror...")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        downloaded = hf_hub_download(
            repo_id=YOLO_MODEL_REPO,
            filename=YOLO_MODEL_FILE,
            repo_type="model",
            local_dir=str(MODELS_DIR),
        )
        model_path = Path(downloaded)

    logger.info(f"Loading YOLO26m model from {model_path}")
    _model = YOLO(str(model_path))
    return _model


def detect_layout(image_path: str) -> list[dict]:
    model = _get_model()
    results = model(image_path, imgsz=YOLO_IMG_SIZE, verbose=False)

    elements = []
    if not results:
        return elements

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return elements

    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())

        element_type = YOLO_CATEGORY_MAP.get(cls_id, f"Unknown-{cls_id}")

        elements.append({
            "element_type": element_type,
            "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
            "confidence": conf,
            "reading_order": -1,
        })

    elements = remove_overlapping_elements(elements)
    return elements


def detect_layout_batch(image_paths: list[str]) -> list[list[dict]]:
    if not image_paths:
        return []

    model = _get_model()
    results = model(image_paths, imgsz=YOLO_IMG_SIZE, verbose=False)

    all_elements = []
    for result in results:
        elements = []
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                box = boxes[i]
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                element_type = YOLO_CATEGORY_MAP.get(cls_id, f"Unknown-{cls_id}")
                elements.append({
                    "element_type": element_type,
                    "bbox": (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    "confidence": conf,
                    "reading_order": -1,
                })
        elements = remove_overlapping_elements(elements)
        all_elements.append(elements)

    return all_elements
