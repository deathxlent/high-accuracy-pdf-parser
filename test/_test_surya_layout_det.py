import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
import numpy as np

img_path = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"

print("Testing surya-ocr LayoutPredictor...")
try:
    from surya.layout import LayoutPredictor
    predictor = LayoutPredictor()
    image = Image.open(img_path)
    result = predictor([image])
    print("Layout result type:", type(result))
    print("Layout result len:", len(result))
    if result:
        r = result[0]
        print("First result type:", type(r))
        attrs = [x for x in dir(r) if not x.startswith('_')]
        print("First result attrs:", attrs)
        if hasattr(r, 'bboxes'):
            print("Bboxes count:", len(r.bboxes))
            for i, bbox in enumerate(r.bboxes[:5]):
                print("  bbox", i, ":", bbox)
                if hasattr(bbox, 'position'):
                    print("    position:", bbox.position)
except Exception as e:
    print("LayoutPredictor failed:", e)
    import traceback
    traceback.print_exc()

print()
print("Testing surya-ocr DetectionPredictor...")
try:
    from surya.detection import DetectionPredictor
    predictor = DetectionPredictor()
    image = Image.open(img_path)
    result = predictor([image])
    print("Detection result type:", type(result))
    print("Detection result len:", len(result))
    if result:
        r = result[0]
        print("First result type:", type(r))
        attrs = [x for x in dir(r) if not x.startswith('_')]
        print("First result attrs:", attrs)
        if hasattr(r, 'bboxes'):
            print("Bboxes count:", len(r.bboxes))
except Exception as e:
    print("DetectionPredictor failed:", e)
    import traceback
    traceback.print_exc()
