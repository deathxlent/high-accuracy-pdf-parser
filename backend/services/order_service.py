import logging
from PIL import Image
from backend.services.layout_service import detect_layout

logger = logging.getLogger(__name__)

_manager = None


def _get_ordering_manager():
    global _manager
    if _manager is not None:
        return _manager

    try:
        from surya.inference import SuryaInferenceManager

        logger.info("Initializing Surya inference manager for ordering...")
        _manager = SuryaInferenceManager()
        logger.info("Surya inference manager loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load Surya manager: {e}. Will use fallback reading order.")
        _manager = None

    return _manager


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

    manager = _get_ordering_manager()

    if manager is None:
        logger.info("Using fallback reading order (top-to-bottom, left-to-right)")
        return _fallback_reading_order(elements)

    try:
        from surya.ordering import batch_ordering

        image = Image.open(image_path)
        image_size = image.size

        bboxes = []
        for elem in elements:
            bbox = elem["bbox"]
            x0, y0, x1, y1 = bbox
            x0 = max(0, min(1, x0 / image_size[0]))
            y0 = max(0, min(1, y0 / image_size[1]))
            x1 = max(0, min(1, x1 / image_size[0]))
            y1 = max(0, min(1, y1 / image_size[1]))
            bboxes.append([x0, y0, x1, y1])

        ordering_results = batch_ordering([image], [bboxes], manager)

        if not ordering_results or not ordering_results[0].bboxes:
            logger.warning("Surya batch_ordering returned no results, using fallback")
            return _fallback_reading_order(elements)

        surya_boxes = []
        for bbox_info in ordering_results[0].bboxes:
            if hasattr(bbox_info, 'bbox'):
                bbox = tuple(bbox_info.bbox)
            elif hasattr(bbox_info, 'polygon'):
                poly = bbox_info.polygon
                bbox = (poly[0][0], poly[0][1], poly[2][0], poly[2][1])
            else:
                bbox = (0, 0, 0, 0)

            position = bbox_info.position if hasattr(bbox_info, 'position') else 0
            surya_boxes.append({"bbox": bbox, "position": position})

        surya_boxes.sort(key=lambda s: s["position"])

        for elem in elements:
            best_iou = 0.0
            best_position = len(elements)

            elem_bbox = elem["bbox"]
            elem_norm = (
                elem_bbox[0] / image_size[0],
                elem_bbox[1] / image_size[1],
                elem_bbox[2] / image_size[0],
                elem_bbox[3] / image_size[1],
            )

            for sbox in surya_boxes:
                iou = _compute_iou(elem_norm, sbox["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_position = sbox["position"]

            elem["reading_order"] = best_position

        elements.sort(key=lambda e: e["reading_order"])
        for i, elem in enumerate(elements):
            elem["reading_order"] = i

        return elements

    except Exception as e:
        logger.error(f"Surya batch_ordering failed: {e}, using fallback")
        return _fallback_reading_order(elements)


def assign_reading_order_batch(pages_elements: list[list[dict]], image_paths: list[str]) -> list[list[dict]]:
    manager = _get_ordering_manager()

    if manager is None:
        logger.info("Using fallback reading order for all pages")
        return [_fallback_reading_order(elems) for elems in pages_elements]

    try:
        from surya.ordering import batch_ordering

        all_images = []
        all_bboxes = []
        valid_indices = []

        for idx, (elements, image_path) in enumerate(zip(pages_elements, image_paths)):
            if not elements:
                continue

            image = Image.open(image_path)
            image_size = image.size
            all_images.append(image)

            page_bboxes = []
            for elem in elements:
                bbox = elem["bbox"]
                x0, y0, x1, y1 = bbox
                x0 = max(0, min(1, x0 / image_size[0]))
                y0 = max(0, min(1, y0 / image_size[1]))
                x1 = max(0, min(1, x1 / image_size[0]))
                y1 = max(0, min(1, y1 / image_size[1]))
                page_bboxes.append([x0, y0, x1, y1])
            all_bboxes.append(page_bboxes)
            valid_indices.append(idx)

        if not all_images:
            return [_fallback_reading_order(elems) for elems in pages_elements]

        ordering_results = batch_ordering(all_images, all_bboxes, manager)

        result = [_fallback_reading_order(elems) for elems in pages_elements]

        for res_idx, page_idx in enumerate(valid_indices):
            if res_idx >= len(ordering_results) or not ordering_results[res_idx].bboxes:
                continue

            elements = pages_elements[page_idx]
            image_path = image_paths[page_idx]
            image = Image.open(image_path)
            image_size = image.size

            surya_boxes = []
            for bbox_info in ordering_results[res_idx].bboxes:
                if hasattr(bbox_info, 'bbox'):
                    bbox = tuple(bbox_info.bbox)
                elif hasattr(bbox_info, 'polygon'):
                    poly = bbox_info.polygon
                    bbox = (poly[0][0], poly[0][1], poly[2][0], poly[2][1])
                else:
                    bbox = (0, 0, 0, 0)
                position = bbox_info.position if hasattr(bbox_info, 'position') else 0
                surya_boxes.append({"bbox": bbox, "position": position})

            surya_boxes.sort(key=lambda s: s["position"])

            for elem in elements:
                best_iou = 0.0
                best_position = len(elements)
                elem_bbox = elem["bbox"]
                elem_norm = (
                    elem_bbox[0] / image_size[0],
                    elem_bbox[1] / image_size[1],
                    elem_bbox[2] / image_size[0],
                    elem_bbox[3] / image_size[1],
                )

                for sbox in surya_boxes:
                    iou = _compute_iou(elem_norm, sbox["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_position = sbox["position"]

                elem["reading_order"] = best_position

            elements.sort(key=lambda e: e["reading_order"])
            for i, elem in enumerate(elements):
                elem["reading_order"] = i

            result[page_idx] = elements

        return result

    except Exception as e:
        logger.error(f"Surya batch_ordering failed: {e}, using fallback")
        return [_fallback_reading_order(elems) for elems in pages_elements]
