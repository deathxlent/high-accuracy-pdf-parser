import fitz

doc = fitz.open("tmp/9214bfa4e7c8420297fc35ebc7fbc923.pdf")
print(f"Pages: {len(doc)}")
page = doc[0]
print(f"Size: {page.rect.width:.1f} x {page.rect.height:.1f}")

text = page.get_text("text")
print(f"\n=== Plain text ({len(text)} chars) ===")
print(text[:3000])

print("\n" + "="*60)

blocks = page.get_text("dict")["blocks"]
print(f"\n=== Text blocks ({len(blocks)}) ===")
for i, block in enumerate(blocks):
    if block["type"] == 0:
        for line in block.get("lines", []):
            line_text = "".join(span["text"] for span in line["spans"])
            bbox = line["bbox"]
            if line_text.strip():
                print(f"  [{line_text.strip()}]  at ({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f})")
    elif block["type"] == 1:
        print(f"  IMAGE at {block['bbox']}")

doc.close()
