"""Analyze result.jpg for gray-ish bounding boxes"""
import cv2
import numpy as np

img = cv2.imread('result.jpg')
h, w = img.shape[:2]
print(f"Image size: {w}x{h}")

elements_info = [
    ("Table",       186, 1448, 1609, 2134),
    ("Text",        251, 710, 1398, 1001),
    ("Text",        248, 1163, 1400, 1261),
    ("Table",       188, 457, 1420, 694),
    ("Text",        247, 216, 1276, 248),
    ("Text",        318, 281, 1227, 313),
    ("Text",        317, 1305, 565, 1337),
    ("Text",        318, 1099, 573, 1129),
    ("Picture",     252, 83, 352, 155),
    ("Page-footer", 839, 2177, 863, 2194),
    ("Text",        471, 1369, 1223, 1403),
    ("Text",        1205, 411, 1381, 442),
    ("Text",        361, 346, 1318, 379),
    ("Text",        239, 217, 1285, 248),
    ("Text",        398, 124, 1399, 153),
    ("Page-header", 398, 124, 1399, 153),
    ("Section-header", 471, 1369, 1223, 1403),
]

print("\nDrawn box border colors at each element:")
for name, x1, y1, x2, y2 in elements_info:
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)

    border_points = [
        ((x1+x2)//2, y1+3), ((x1+x2)//2, y2-3),
        (x1+3, (y1+y2)//2), (x2-3, (y1+y2)//2),
    ]
    colors = []
    for px, py in border_points:
        px, py = int(px), int(py)
        if 0 <= px < w and 0 <= py < h:
            colors.append(tuple(img[py, px]))

    avg_b = sum(c[0] for c in colors) / len(colors)
    avg_g = sum(c[1] for c in colors) / len(colors)
    avg_r = sum(c[2] for c in colors) / len(colors)
    gray_diff = abs(avg_b - avg_g) + abs(avg_g - avg_r)
    marker = " <-- GRAY?" if gray_diff < 30 else ""
    print(f"  {name:20s}: avg BGR=({avg_b:6.0f},{avg_g:6.0f},{avg_r:6.0f}) diff={gray_diff:.0f}{marker}")

# Check for pure gray pixels in bottom quarter
print("\nGray pixel scan in bottom region:")
bl = img[3*h//4:, :, :]
for py in range(0, bl.shape[0], 10):
    for px in range(0, bl.shape[1], 10):
        b, g, r = int(bl[py, px, 0]), int(bl[py, px, 1]), int(bl[py, px, 2])
        if abs(b-g) < 15 and abs(g-r) < 15 and 130 < b < 210:
            print(f"  Gray at ({px},{py+3*h//4}): BGR=({b},{g},{r})")
