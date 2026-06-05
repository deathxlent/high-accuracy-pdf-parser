import logging
from PIL import Image
from backend.services.layout_service import detect_layout

logger = logging.getLogger(__name__)

_manager = None
_layout_predictor = None


def _get_layout_predictor():
    global _manager, _layout_predictor
    if _layout_predictor is not None:
        return _layout_predictor

    try:
        from surya.inference import SuryaInferenceManager
        from surya.layout import LayoutPredictor

        logger.info("Initializing Surya inference manager for reading order...")
        _manager = SuryaInferenceManager()
        _layout_predictor = LayoutPredictor(_manager)
        logger.info("Surya layout predictor loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Surya: {e}. Will use fallback reading order.")
        _layout_predictor = None

    return _layout_predictor


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


def _fallback_reading_order(elements: list[dict]) -> list[dict]:
    sorted_elements = sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))
    for i, elem in enumerate(sorted_elements):
        elem["reading_order"] = i
    return sorted_elements


def assign_reading_order(elements: list[dict], image_path: str) -> list[dict]:
    if not elements:
        return elements

    predictor = _get_layout_predictor()

    if predictor is None:
        logger.info("Using fallback reading order (top-to-bottom, left-to-right)")
        return _fallback_reading_order(elements)

    try:
        image = Image.open(image_path)
        layout_results = predictor([image])

        if not layout_results or not layout_results[0].bboxes:
            logger.warning("Surya returned no layout results, using fallback")
            return _fallback_reading_order(elements)

        surya_boxes = []
        for bbox_info in layout_results[0].bboxes:
            surya_boxes.append({
                "bbox": tuple(bbox_info.bbox) if hasattr(bbox_info, 'bbox') else tuple(bbox_info.polygon[:2] + bbox_info.polygon[2:4]) if hasattr(bbox_info, 'polygon') else (0, 0, 0, 0),
                "position": bbox_info.position if hasattr(bbox_info, 'position') else 0,
            })

        surya_boxes.sort(key=lambda s: s["position"])

        for elem in elements:
            best_iou = 0.0
            best_position = len(elements)
            for sbox in surya_boxes:
                iou = _compute_iou(elem["bbox"], sbox["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_position = sbox["position"]
            elem["reading_order"] = best_position

        elements.sort(key=lambda e: e["reading_order"])
        for i, elem in enumerate(elements):
            elem["reading_order"] = i

        return elements

    except Exception as e:
        logger.error(f"Surya reading order failed: {e}, using fallback")
        return _fallback_reading_order(elements)
