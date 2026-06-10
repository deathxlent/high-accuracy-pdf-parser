import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)

model_dir = str(PROJECT_ROOT / "models" / "official_models" / "PaddleOCR-VL-1.6")
test_img_dir = PROJECT_ROOT / "tmp" / "42e59745cdb54b6fb2c635d7c11dbd43"

try:
    from paddlex.inference.models.common.tokenizer import LlamaTokenizer
    from paddlex.inference.models.common.tokenizer.tokenizer_utils import ChatTemplate
    from paddlex.inference.models.doc_vlm.processors import (
        PaddleOCRVLProcessor,
        SiglipImageProcessor,
    )

    print("Building processor with patched code...")
    image_processor = SiglipImageProcessor.from_pretrained(model_dir)
    vocab_file = str(Path(model_dir, "tokenizer.model"))
    tokenizer = LlamaTokenizer.from_pretrained(model_dir, vocab_file=vocab_file)
    chat_template_file = Path(model_dir, "chat_template.jinja")
    tokenizer.chat_template = ChatTemplate._compile_jinja_template(
        chat_template_file.read_text(encoding="utf-8")
    )
    processor = PaddleOCRVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    print("Processor created successfully.")

    print("\nLoading test image...")
    img_files = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    test_img = img_files[0]
    print(f"Test image: {test_img}")

    print("\nCalling processor.preprocess() (should use patched code)...")
    input_data = [{"image": str(test_img), "query": "OCR:"}]
    result = processor.preprocess(input_data)

    print(f"\nResult keys: {result.keys()}")

    if "input_ids" in result:
        input_ids = result["input_ids"].numpy()
        print(f"Input IDs shape: {input_ids.shape}")
        
        image_token_id = 100295
        n_image_tokens = (input_ids[0] == image_token_id).sum()
        print(f"Number of image tokens (id={image_token_id}): {n_image_tokens}")
        
        if n_image_tokens > 0:
            print("SUCCESS: Image tokens are present!")
        else:
            print("FAILED: No image tokens found!")

    if "pixel_values" in result:
        print(f"Pixel values shape: {result['pixel_values'].shape}")

    if "image_grid_thw" in result:
        print(f"Image grid THW: {result['image_grid_thw'].numpy()}")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
