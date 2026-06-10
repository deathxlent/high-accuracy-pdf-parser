import os
import sys
import json

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_dir = r"c:\ws\high accuracy pdf parser\models\official_models\PaddleOCR-VL-1.6"

tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
with open(tokenizer_config_path, "r", encoding="utf-8") as f:
    tokenizer_config = json.load(f)

print("Has chat_template:", "chat_template" in tokenizer_config)

if "chat_template" in tokenizer_config:
    print("chat_template:", tokenizer_config["chat_template"])
else:
    print("chat_template NOT FOUND in tokenizer_config.json!")
    print("\nKeys in tokenizer_config:", list(tokenizer_config.keys())[:20])

added_tokens_path = os.path.join(model_dir, "added_tokens.json")
if os.path.exists(added_tokens_path):
    with open(added_tokens_path, "r", encoding="utf-8") as f:
        added_tokens = json.load(f)
    print("\nImage token in added_tokens:", "<|IMAGE_PLACEHOLDER|>" in added_tokens)
    if "<|IMAGE_PLACEHOLDER|>" in added_tokens:
        print("Image token id:", added_tokens["<|IMAGE_PLACEHOLDER|>"])

try:
    from paddlenlp.transformers import AutoTokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    print("\nTokenizer has apply_chat_template:", hasattr(tokenizer, "apply_chat_template"))
    print("Tokenizer has chat_template:", hasattr(tokenizer, "chat_template"))
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        print("chat_template:", tokenizer.chat_template[:200])

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

    tokens = tokenizer(prompt, return_tensors="np")
    input_ids = tokens["input_ids"][0]
    print(f"\nTotal tokens: {len(input_ids)}")
    
    image_token_id = 100295
    n_image_tokens = (input_ids == image_token_id).sum()
    print(f"Number of image tokens (id={image_token_id}): {n_image_tokens}")

except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
