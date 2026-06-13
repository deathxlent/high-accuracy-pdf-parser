import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModel, AutoModelForVision2Seq, AutoProcessor, AutoImageProcessor
import torch

MODEL_REPO = "vikp/surya_order"

print("Attempting to load surya_order model...")

try:
    model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True)
    print("AutoModel loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"AutoModel failed: {e}")

try:
    model = AutoModelForVision2Seq.from_pretrained(MODEL_REPO, trust_remote_code=True)
    print("AutoModelForVision2Seq loaded successfully!")
    print(f"Model type: {type(model)}")
except Exception as e:
    print(f"AutoModelForVision2Seq failed: {e}")

try:
    processor = AutoProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
    print("AutoProcessor loaded successfully!")
    print(f"Processor type: {type(processor)}")
except Exception as e:
    print(f"AutoProcessor failed: {e}")

try:
    img_processor = AutoImageProcessor.from_pretrained(MODEL_REPO, trust_remote_code=True)
    print("AutoImageProcessor loaded successfully!")
    print(f"Image processor type: {type(img_processor)}")
except Exception as e:
    print(f"AutoImageProcessor failed: {e}")
