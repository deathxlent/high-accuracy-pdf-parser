import fitz
from PIL import Image

doc = fitz.open("tmp/9214bfa4e7c8420297fc35ebc7fbc923.pdf")
page = doc[0]

# Table region (first small table) - what PyMuPDF returns
rect1 = fitz.Rect(188, 457, 1420, 694)
text1 = page.get_text("text", clip=rect1).strip()
print("=== PyMuPDF text for Table 1 (garbled) ===")
print(text1[:300])

# Second big table
rect2 = fitz.Rect(186, 1448, 1609, 2134)
text2 = page.get_text("text", clip=rect2).strip()
print("\n=== PyMuPDF text for Table 2 (garbled) ===")
print(text2[:500])

# Try OCR on first table region
print("\n=== Rendering table region to image for OCR ===")
mat = fitz.Matrix(200/72, 200/72)
pix = page.get_pixmap(matrix=mat, clip=rect1)
img_path = "tmp/table_region_test.png"
pix.save(img_path)

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
result = ocr.ocr(img_path, cls=False)
print("\n=== PaddleOCR result ===")
if result and result[0]:
    for line in result[0]:
        txt = line[1][0] if line[1] else ""
        print(f"  [{txt}]")
else:
    print("  No OCR results")

doc.close()
