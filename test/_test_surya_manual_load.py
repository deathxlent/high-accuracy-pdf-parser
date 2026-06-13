import sys
import os
sys.path.insert(0, r"c:\ws\high accuracy pdf parser")

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from pathlib import Path
import torch
from safetensors.torch import load_file

MODELS_DIR = Path(r"c:\ws\high accuracy pdf parser\models")
SURYA_ORDER_DIR = MODELS_DIR / "surya_order"

print("=== Testing manual safetensors load ===")

print(f"Loading model.safetensors...")
try:
    state_dict = load_file(str(SURYA_ORDER_DIR / "model.safetensors"))
    print(f"✓ Loaded successfully, {len(state_dict)} tensors")
    for k, v in list(state_dict.items())[:5]:
        print(f"  {k}: {v.shape}, dtype={v.dtype}")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n=== Testing with transformers from_pretrained with local files ===")
try:
    from transformers import VisionEncoderDecoderConfig
    from surya.model.ordering.model import OrderVisionEncoderDecoderModel
    from surya.model.ordering.encoderdecoder import MBartOrderConfig, VariableDonutSwinConfig, MBartOrder, VariableDonutSwinModel
    from transformers import AutoModel, AutoModelForCausalLM

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
    print(f"✓ Model created")

    print("Loading state_dict...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"✓ State dict loaded")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    model = model.to('cuda')
    model = model.eval()
    print(f"✓ Model moved to CUDA")

except Exception as e:
    print(f"✗ Failed: {e}")
    import traceback
    traceback.print_exc()
