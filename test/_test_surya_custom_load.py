import sys
import os
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import torch
from safetensors.torch import load_file

MODELS_DIR = Path(r"c:\ws\high accuracy pdf parser\models")
SURYA_ORDER_DIR = MODELS_DIR / "surya_order"

print("=== Testing custom model loading ===")

from transformers import VisionEncoderDecoderConfig, AutoModel, AutoModelForCausalLM
from surya.model.ordering.config import MBartOrderConfig, VariableDonutSwinConfig
from surya.model.ordering.decoder import MBartOrder
from surya.model.ordering.encoder import VariableDonutSwinModel
from surya.model.ordering.encoderdecoder import OrderVisionEncoderDecoderModel

print("Loading config...")
config = VisionEncoderDecoderConfig.from_pretrained(str(SURYA_ORDER_DIR))

decoder_config = vars(config.decoder)
decoder = MBartOrderConfig(**decoder_config)
config.decoder = decoder

encoder_config = vars(config.encoder)
encoder = VariableDonutSwinConfig(**encoder_config)
config.encoder = encoder

AutoModel.register(MBartOrderConfig, MBartOrder)
AutoModelForCausalLM.register(MBartOrderConfig, MBartOrder)
AutoModel.register(VariableDonutSwinConfig, VariableDonutSwinModel)

print("Creating model...")
model = OrderVisionEncoderDecoderModel(config)
print(f"✓ Model created (instantiated with random weights)")

print("Loading state dict from safetensors...")
state_dict = load_file(str(SURYA_ORDER_DIR / "model.safetensors"))
print(f"✓ State dict loaded: {len(state_dict)} tensors")

print("Loading state dict into model...")
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"✓ State dict loaded into model")
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if missing:
    print(f"  First 10 missing: {missing[:10]}")
if unexpected:
    print(f"  First 10 unexpected: {unexpected[:10]}")

print("Moving model to CUDA...")
model = model.to('cuda')
model = model.eval()
print(f"✓ Model on CUDA, dtype={model.dtype}")

print("\n=== Testing processor loading ===")
from surya.model.ordering.processor import OrderImageProcessor
processor = OrderImageProcessor.from_pretrained(str(SURYA_ORDER_DIR))
print(f"✓ Processor loaded")
print(f"  Type: {type(processor)}")

print("\n=== All tests passed! ===")
