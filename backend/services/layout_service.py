import os
import logging
from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from backend.config import MODELS_DIR, YOLO_MODEL_REPO, YOLO_MODEL_FILE, YOLO_IMG_SIZE

logger = logging.getLogger(__name__)

_model = None

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

    return elements
