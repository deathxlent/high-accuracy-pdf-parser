import logging
import os
import tempfile
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_engine = None
_formula_engine = None


def _get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    try:
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR engine...")
        _ocr_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            show_log=False,
        )
        logger.info("PaddleOCR engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR: {e}")
        raise

    return _ocr_engine


def _get_formula_engine():
    global _formula_engine
    if _formula_engine is not None:
        return _formula_engine

    try:
        from paddleocr import PaddleOCR
        logger.info("Initializing PaddleOCR formula engine...")
        _formula_engine = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            show_log=False,
        )
        logger.info("PaddleOCR formula engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to initialize formula engine: {e}")
        raise

    return _formula_engine


def _crop_and_save_image(image_path: str, bbox: tuple = None) -> str:
    img = Image.open(image_path)

    if bbox is not None:
        x0, y0, x1, y1 = [int(v) for v in bbox]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.width, x1)
        y1 = min(img.height, y1)
        if x1 <= x0 or y1 <= y0:
            return ""
        img = img.crop((x0, y0, x1, y1))

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = tmp.name
    img.save(tmp_path, "PNG")
    tmp.close()

    return tmp_path


def _extract_text_from_ocr_result(result) -> str:
    if not result or not result[0]:
        return ""

    lines = []
    for line in result[0]:
        if line and len(line) >= 2:
            text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
            if text.strip():
                lines.append(text.strip())

    return "\n".join(lines)


def ocr_region(image_path: str, bbox: tuple[float, float, float, float] = None) -> str:
    ocr = _get_ocr_engine()
    tmp_path = _crop_and_save_image(image_path, bbox)

    if not tmp_path:
        return ""

    try:
        result = ocr.ocr(tmp_path, cls=False)
        return _extract_text_from_ocr_result(result)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ocr_formula(image_path: str, bbox: tuple[float, float, float, float] = None) -> str:
    engine = _get_formula_engine()
    tmp_path = _crop_and_save_image(image_path, bbox)

    if not tmp_path:
        return ""

    try:
        result = engine.ocr(tmp_path, cls=False)
        if not result or not result[0]:
            return ""

        latex_parts = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                if text.strip():
                    latex_parts.append(text.strip())

        latex = " ".join(latex_parts)
        if latex and not latex.startswith("$"):
            latex = f"${latex}$"
        return latex
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ocr_batch(image_path: str, bboxes: list[tuple]) -> list[str]:
    if not bboxes:
        return []

    ocr = _get_ocr_engine()
    tmp_paths = []
    results = []

    try:
        for bbox in bboxes:
            tmp_path = _crop_and_save_image(image_path, bbox)
            tmp_paths.append(tmp_path)

        for tmp_path in tmp_paths:
            if not tmp_path:
                results.append("")
                continue

            try:
                result = ocr.ocr(tmp_path, cls=False)
                text = _extract_text_from_ocr_result(result)
                results.append(text)
            except Exception as e:
                logger.error(f"Batch OCR failed for a region: {e}")
                results.append("")

        return results
    finally:
        for tmp_path in tmp_paths:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def ocr_batch_multi_image(image_bbox_pairs: list[tuple[str, tuple]]) -> list[str]:
    if not image_bbox_pairs:
        return []

    ocr = _get_ocr_engine()
    tmp_paths = []
    results = []

    try:
        for image_path, bbox in image_bbox_pairs:
            tmp_path = _crop_and_save_image(image_path, bbox)
            tmp_paths.append(tmp_path)

        for tmp_path in tmp_paths:
            if not tmp_path:
                results.append("")
                continue

            try:
                result = ocr.ocr(tmp_path, cls=False)
                text = _extract_text_from_ocr_result(result)
                results.append(text)
            except Exception as e:
                logger.error(f"Multi-image batch OCR failed: {e}")
                results.append("")

        return results
    finally:
        for tmp_path in tmp_paths:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
