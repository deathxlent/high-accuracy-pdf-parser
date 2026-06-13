from PIL import Image
from surya.layout import LayoutPredictor
import json

test_img = r"c:\ws\high accuracy pdf parser\tmp\3413826ca8d84c28ad51df7feb91d06d\page_1.jpg"

print("Loading LayoutPredictor...")
predictor = LayoutPredictor()
print("Predictor loaded!")

print(f"\nProcessing image: {test_img}")
image = Image.open(test_img)
results = predictor([image])

print(f"\nNumber of results: {len(results)}")
if results:
    result = results[0]
    print(f"Result type: {type(result)}")
    print(f"Result dir: {[x for x in dir(result) if not x.startswith('_')]}")
    
    print(f"\nNumber of bboxes: {len(result.bboxes)}")
    for i, bbox in enumerate(result.bboxes[:5]):
        print(f"\n  BBox #{i}:")
        print(f"    type: {type(bbox)}")
        print(f"    dir: {[x for x in dir(bbox) if not x.startswith('_')]}")
        if hasattr(bbox, 'bbox'):
            print(f"    bbox: {bbox.bbox}")
        if hasattr(bbox, 'position'):
            print(f"    position: {bbox.position}")
        if hasattr(bbox, 'label'):
            print(f"    label: {bbox.label}")
