import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)

from paddlex.inference.models import PaddlePredictorOption
from paddlex.inference.models.doc_vlm.predictor import DocVLMPredictor

model_dir = str(PROJECT_ROOT / "models" / "official_models" / "PaddleOCR-VL-1.6")
print(f"Model dir: {model_dir}")
print(f"chat_template.jinja exists: {Path(model_dir, 'chat_template.jinja').exists()}")

try:
    print("\nCreating DocVLMPredictor...")
    pp_option = PaddlePredictorOption(device="cpu")
    predictor = DocVLMPredictor(
        model_dir=model_dir,
        model_name="PaddleOCR-VL-1.6-0.9B",
        pp_option=pp_option,
    )
    print(f"Predictor created: {predictor}")
    print(f"Processor type: {type(predictor.processor)}")
    print(f"Tokenizer type: {type(predictor.processor.tokenizer)}")
    print(f"Tokenizer has chat_template: {hasattr(predictor.processor.tokenizer, 'chat_template') and predictor.processor.tokenizer.chat_template is not None}")

    if hasattr(predictor.processor.tokenizer, 'chat_template') and predictor.processor.tokenizer.chat_template:
        print(f"chat_template (first 200 chars): {str(predictor.processor.tokenizer.chat_template)[:200]}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "placeholder"},
                {"type": "text", "text": "OCR:"},
            ],
        }
    ]
    
    prompt = predictor.processor.tokenizer.apply_chat_template(messages, tokenize=False)
    print(f"\nGenerated prompt:\n{repr(prompt)}")
    print(f"\nHas <|IMAGE_PLACEHOLDER|>: {'<|IMAGE_PLACEHOLDER|>' in prompt}")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
