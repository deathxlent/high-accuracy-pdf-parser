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
print(f"Model dir: {model_dir}")
print(f"chat_template.jinja exists: {Path(model_dir, 'chat_template.jinja').exists()}")

try:
    from paddlex.inference.models.common.tokenizer import LlamaTokenizer
    from paddlex.inference.models.common.tokenizer.tokenizer_utils import ChatTemplate
    from paddlex.inference.models.doc_vlm.processors import (
        PaddleOCRVLProcessor,
        SiglipImageProcessor,
    )

    print("\nBuilding image_processor...")
    image_processor = SiglipImageProcessor.from_pretrained(model_dir)
    print(f"Image processor created: {type(image_processor)}")

    print("\nBuilding tokenizer...")
    vocab_file = str(Path(model_dir, "tokenizer.model"))
    tokenizer = LlamaTokenizer.from_pretrained(model_dir, vocab_file=vocab_file)
    print(f"Tokenizer created: {type(tokenizer)}")

    print("\nLoading chat_template...")
    chat_template_file = Path(model_dir, "chat_template.jinja")
    chat_template_content = chat_template_file.read_text(encoding="utf-8")
    print(f"chat_template content (first 200 chars): {chat_template_content[:200]}")

    print("\nCompiling chat_template...")
    compiled_template = ChatTemplate._compile_jinja_template(chat_template_content)
    print(f"Compiled template type: {type(compiled_template)}")
    tokenizer.chat_template = compiled_template
    print(f"Tokenizer has chat_template: {hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None}")

    print("\nCreating PaddleOCRVLProcessor...")
    processor = PaddleOCRVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    print(f"Processor created: {type(processor)}")

    print("\nTesting apply_chat_template...")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\nGenerated prompt:\n{repr(prompt)}")
    print(f"\nHas <|IMAGE_PLACEHOLDER|>: {'<|IMAGE_PLACEHOLDER|>' in prompt}")
    print(f"Has <|IMAGE_START|>: {'<|IMAGE_START|>' in prompt}")
    print(f"Has <|IMAGE_END|>: {'<|IMAGE_END|>' in prompt}")

    print("\nNow testing processor.preprocess with a real image...")
    from PIL import Image
    import numpy as np

    test_img_dir = PROJECT_ROOT / "tmp" / "42e59745cdb54b6fb2c635d7c11dbd43"
    img_files = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if img_files:
        test_img = img_files[0]
        print(f"Using test image: {test_img}")
        
        input_data = [{"image": str(test_img), "query": "OCR:"}]
        processed = processor.preprocess(input_data)
        print(f"\nProcessed output keys: {processed.keys()}")
        
        if "input_ids" in processed:
            input_ids = processed["input_ids"].numpy()
            print(f"Input IDs shape: {input_ids.shape}")
            image_token_id = 100295
            n_image_tokens = (input_ids[0] == image_token_id).sum()
            print(f"Number of image tokens (id={image_token_id}): {n_image_tokens}")
        
        if "image_embeds" in processed:
            print(f"Image embeds shape: {processed['image_embeds'].shape}")

    else:
        print("No test images found!")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
