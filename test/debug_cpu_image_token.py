import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models" / "paddlex_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PADDLE_PDX_CACHE_HOME"] = str(MODELS_DIR.parent)
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

import paddle

print("=" * 60)
print("Debug: PaddleOCR-VL Image Token Mismatch")
print("=" * 60)

print(f"\n[1] Paddle version: {paddle.__version__}")
print(f"    Compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")
print(f"    CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

try:
    test_tensor = paddle.to_tensor([2, 3, 4])
    print(f"\n[2] Test tensor: {test_tensor}")
    print(f"    Tensor place: {test_tensor.place}")
    print(f"    Tensor prod(): {test_tensor.prod()}")
    print(f"    int(prod()): {int(test_tensor.prod())}")
    print(f"    numpy().prod(): {test_tensor.numpy().prod()}")
    print(f"    int(numpy().prod()): {int(test_tensor.numpy().prod())}")
except Exception as e:
    print(f"    Error: {e}")

print(f"\n[3] Loading model and processor...")
MODEL_PATH = PROJECT_ROOT / "models" / "official_models" / "PaddleOCR-VL-1.6"
print(f"    Model path: {MODEL_PATH}")

if not MODEL_PATH.exists():
    print(f"    ERROR: Model path not found!")
    sys.exit(1)

try:
    from paddlex.inference.models.doc_vlm.processors.paddleocr_vl import PaddleOCRVLProcessor
    from paddlex.inference.models.doc_vlm.processors.common import SiglipImageProcessor
    from paddlex.inference.models.common.tokenizer import LlamaTokenizer
    from paddlex.inference.models.common.tokenizer.tokenizer_utils import ChatTemplate
    
    image_processor = SiglipImageProcessor.from_pretrained(str(MODEL_PATH))
    vocab_file = str(MODEL_PATH / "tokenizer.model")
    tokenizer = LlamaTokenizer.from_pretrained(str(MODEL_PATH), vocab_file=vocab_file)
    
    chat_template_file = MODEL_PATH / "chat_template.jinja"
    if chat_template_file.exists():
        tokenizer.chat_template = ChatTemplate._compile_jinja_template(
            chat_template_file.read_text(encoding="utf-8")
        )
        print(f"    Chat template loaded from: {chat_template_file}")
    else:
        print(f"    WARNING: chat_template.jinja not found!")
    
    processor = PaddleOCRVLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    print(f"    Processor created successfully")
    print(f"    Image token: {repr(processor.image_token)}")
    print(f"    Merge size: {image_processor.merge_size}")
    
except Exception as e:
    print(f"    ERROR loading processor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n[4] Testing with test image...")
IMAGE_DIR = PROJECT_ROOT / "tmp" / "42e59745cdb54b6fb2c635d7c11dbd43"
test_images = sorted([f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f.startswith('page_')])

if not test_images:
    print(f"    No test images found in {IMAGE_DIR}")
    sys.exit(1)

test_img = IMAGE_DIR / test_images[0]
print(f"    Test image: {test_img}")

try:
    test_data = [{"image": str(test_img), "query": "OCR:"}]
    
    print(f"\n[5] Running processor.preprocess()...")
    result = processor.preprocess(test_data)
    
    print(f"\n[6] Result keys: {list(result.keys())}")
    
    if "input_ids" in result:
        input_ids = result["input_ids"]
        print(f"    input_ids shape: {input_ids.shape}")
        print(f"    input_ids type: {type(input_ids)}")
        
        if hasattr(input_ids, 'place'):
            print(f"    input_ids place: {input_ids.place}")
        
        image_token_id = tokenizer.convert_tokens_to_ids(processor.image_token)
        print(f"    Image token ID: {image_token_id}")
        
        n_image_tokens = (input_ids == image_token_id).sum().item()
        print(f"    Number of image tokens in input_ids: {n_image_tokens}")
    
    if "image_grid_thw" in result:
        image_grid_thw = result["image_grid_thw"]
        print(f"\n    image_grid_thw: {image_grid_thw}")
        print(f"    image_grid_thw type: {type(image_grid_thw)}")
        
        if hasattr(image_grid_thw, 'place'):
            print(f"    image_grid_thw place: {image_grid_thw.place}")
        
        if hasattr(image_grid_thw, 'shape'):
            print(f"    image_grid_thw shape: {image_grid_thw.shape}")
        
        print(f"\n[7] Testing prod() on image_grid_thw...")
        for i in range(len(image_grid_thw)):
            grid = image_grid_thw[i]
            print(f"\n    Grid {i}: {grid}")
            print(f"      type: {type(grid)}")
            
            prod_result = grid.prod()
            print(f"      prod(): {prod_result}")
            print(f"      int(prod()): {int(prod_result)}")
            
            np_prod = grid.numpy().prod()
            print(f"      numpy().prod(): {np_prod}")
            print(f"      int(numpy().prod()): {int(np_prod)}")
            
            n_tokens = int(grid.prod()) // image_processor.merge_size // image_processor.merge_size
            print(f"      Calculated image tokens (using prod()): {n_tokens}")
            
            n_tokens_np = int(grid.numpy().prod()) // image_processor.merge_size // image_processor.merge_size
            print(f"      Calculated image tokens (using numpy().prod()): {n_tokens_np}")
    
    print(f"\n[8] Checking decoded text for image tokens...")
    if "input_ids" in result:
        decoded = tokenizer.decode(result["input_ids"][0])
        n_image_in_text = decoded.count(processor.image_token)
        print(f"    Image tokens in decoded text: {n_image_in_text}")
        if n_image_in_text > 0:
            print(f"    First 200 chars of decoded text: {repr(decoded[:200])}")
        else:
            print(f"    WARNING: No image tokens found in decoded text!")
            print(f"    Full decoded text (first 500 chars): {repr(decoded[:500])}")
    
    print(f"\n[9] Checking if image_token exists in processor's text...")
    test_text = f"User: {processor.image_token}\nOCR:\nAssistant:"
    print(f"    Test text contains image_token: {processor.image_token in test_text}")
    
except Exception as e:
    print(f"\n    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Debug complete")
print("=" * 60)
