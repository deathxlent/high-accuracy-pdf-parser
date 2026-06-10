import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import paddle
paddle.device.set_device("cpu")

from paddlex.inference.models.doc_vlm.processors import PaddleOCRVLProcessor
from paddlex.inference.models.doc_vlm import PaddleOCRVLForConditionalGeneration
import cv2

model_dir = PROJECT_ROOT / "models" / "official_models" / "PaddleOCR-VL-1.6"

print("Loading processor and tokenizer...")
processor = PaddleOCRVLProcessor.from_pretrained(str(model_dir))

print("\nTesting apply_chat_template:")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "placeholder"},
            {"type": "text", "text": "OCR:"},
        ],
    }
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False)
print(f"Prompt:\n{repr(prompt)}")
print(f"\nHas image_token? {processor.image_token in prompt}")
print(f"image_token = {repr(processor.image_token)}")

print("\nTokenizing prompt:")
tokens = processor.tokenizer(prompt, return_tensors="pd")
input_ids = tokens["input_ids"]
print(f"input_ids shape: {input_ids.shape}")
print(f"input_ids: {input_ids}")

from paddlex.inference.models.doc_vlm.modeling.paddleocr_vl import PaddleOCRVLConfig
config = PaddleOCRVLConfig.from_pretrained(str(model_dir))
print(f"\nconfig.image_token_id = {config.image_token_id}")

n_image_tokens = (input_ids == config.image_token_id).sum().item()
print(f"n_image_tokens in prompt: {n_image_tokens}")
