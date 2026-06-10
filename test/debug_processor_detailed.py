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
    from paddlex.inference.models.doc_vlm.processors.common import fetch_image

    print("=" * 60)
    print("Step 1: Building components")
    print("=" * 60)

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

    print(f"processor.image_token = {repr(processor.image_token)}")
    print(f"tokenizer.image_token = {repr(getattr(tokenizer, 'image_token', 'NOT FOUND'))}")

    print("\n" + "=" * 60)
    print("Step 2: Loading test image")
    print("=" * 60)

    img_files = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    test_img = img_files[0]
    print(f"Test image: {test_img}")

    input_dicts = [{"image": str(test_img), "query": "OCR:"}]

    print("\n" + "=" * 60)
    print("Step 3: Creating text prompt")
    print("=" * 60)

    text = []
    for input_dict in input_dicts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "placeholder"},
                    {"type": "text", "text": input_dict["query"]},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        text.append(prompt)
    
    print(f"Generated text[0]:\n{repr(text[0])}")
    print(f"\nprocessor.image_token in text[0]: {processor.image_token in text[0]}")

    print("\n" + "=" * 60)
    print("Step 4: Processing image")
    print("=" * 60)

    images = [fetch_image(input_dict["image"]) for input_dict in input_dicts]
    print(f"Image loaded: size={images[0].size}, mode={images[0].mode}")

    image_inputs = image_processor(images=images, size=None, return_tensors="pd")
    image_grid_thw = image_inputs["image_grid_thw"]
    print(f"image_grid_thw = {image_grid_thw}")
    print(f"image_grid_thw[0].prod() = {int(image_grid_thw[0].prod())}")
    print(f"merge_size = {image_processor.merge_size}")
    n_placeholders = int(image_grid_thw[0].prod()) // image_processor.merge_size // image_processor.merge_size
    print(f"Number of placeholders needed: {n_placeholders}")

    print("\n" + "=" * 60)
    print("Step 5: Replacing image tokens")
    print("=" * 60)

    text_before = text[0]
    print(f"Before replacement:")
    print(f"  Contains '{processor.image_token}': {processor.image_token in text_before}")
    print(f"  Count: {text_before.count(processor.image_token)}")

    import copy
    index = 0
    while processor.image_token in text[0]:
        print(f"\nReplacing occurrence {index + 1}...")
        text[0] = text[0].replace(
            processor.image_token,
            "<|placeholder|>" * n_placeholders,
            1,
        )
        index += 1
    
    print(f"\nAfter first replacement step:")
    print(f"  Contains '<|placeholder|>': {'<|placeholder|>' in text[0]}")
    print(f"  Count: {text[0].count('<|placeholder|>')}")

    text[0] = text[0].replace("<|placeholder|>", processor.image_token)
    
    print(f"\nAfter second replacement step:")
    print(f"  Contains '{processor.image_token}': {processor.image_token in text[0]}")
    print(f"  Count: {text[0].count(processor.image_token)}")

    print(f"\nFinal text (first 500 chars):\n{repr(text[0][:500])}")

    print("\n" + "=" * 60)
    print("Step 6: Tokenizing text")
    print("=" * 60)

    text_kwargs = {
        "padding": False,
        "return_tensors": "pd",
    }
    text_inputs = tokenizer(text, **text_kwargs)
    input_ids = text_inputs["input_ids"].numpy()
    print(f"Input IDs shape: {input_ids.shape}")

    image_token_id = 100295
    n_image_tokens = (input_ids[0] == image_token_id).sum()
    print(f"Number of image tokens (id={image_token_id}): {n_image_tokens}")

    print(f"\nAll token IDs: {input_ids[0].tolist()}")
    print(f"\nDecoded tokens:")
    for tid in input_ids[0]:
        decoded = tokenizer.decode([int(tid)])
        print(f"  {tid}: {repr(decoded)}")

    print("\n" + "=" * 60)
    print("Step 7: Check if image_token_id exists in tokenizer vocab")
    print("=" * 60)

    if hasattr(tokenizer, 'get_vocab'):
        vocab = tokenizer.get_vocab()
        if processor.image_token in vocab:
            print(f"processor.image_token '{processor.image_token}' id: {vocab[processor.image_token]}")
        else:
            print(f"processor.image_token '{processor.image_token}' NOT IN VOCAB!")
        
        for special_token in ["<|IMAGE_START|>", "<|IMAGE_END|>", "<|IMAGE_PLACEHOLDER|>"]:
            if special_token in vocab:
                print(f"  '{special_token}' id: {vocab[special_token]}")
            else:
                print(f"  '{special_token}' NOT IN VOCAB!")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
