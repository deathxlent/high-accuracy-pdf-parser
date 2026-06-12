"""Analyze saved PaddleOCR GPU result JSON"""
import os, json

os.environ["PADDLE_PDX_CACHE_HOME"] = "C:\\paddlex_cache"

json_path = r"C:\ws\high accuracy pdf parser\test\output_ocr_gpu\page_1_res.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

rec_texts = data.get("rec_texts", [])
rec_scores = data.get("rec_scores", [])

print(f"Total text blocks: {len(rec_texts)}")
print()
for text, score in zip(rec_texts, rec_scores):
    if text and text.strip():
        print(f"  [{score:.4f}] {text}")

if rec_scores:
    avg_score = sum(rec_scores) / len(rec_scores)
    print()
    print(f"Average confidence: {avg_score:.4f}")
    print(f"Highest: {max(rec_scores):.4f}")
    print(f"Lowest:  {min(rec_scores):.4f}")
    leq05 = len([s for s in rec_scores if s <= 0.5])
    leq08 = len([s for s in rec_scores if s <= 0.8])
    leq09 = len([s for s in rec_scores if s <= 0.9])
    gt09 = len([s for s in rec_scores if s > 0.9])
    print(f"Conf dist: <=0.5:{leq05} <=0.8:{leq08} <=0.9:{leq09} >0.9:{gt09}")

print(f"Det polygons: {len(data.get('dt_polys', []))}")
print(f"Rec boxes:    {len(data.get('rec_boxes', []))}")
print(f"Input:        {data.get('input_path', 'N/A')}")
