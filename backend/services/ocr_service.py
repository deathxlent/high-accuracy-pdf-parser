import logging
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


def ocr_region(image_path: str, bbox: tuple[float, float, float, float] = None) -> str:
    ocr = _get_ocr_engine()
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

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path, "PNG")

    try:
        result = ocr.ocr(tmp_path, cls=False)
        if not result or not result[0]:
            return ""

        lines = []
        for line in result[0]:
            if line and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                if text.strip():
                    lines.append(text.strip())

        return "\n".join(lines)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def ocr_formula(image_path: str, bbox: tuple[float, float, float, float] = None) -> str:
    engine = _get_formula_engine()
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

    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        img.save(tmp_path, "PNG")

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
