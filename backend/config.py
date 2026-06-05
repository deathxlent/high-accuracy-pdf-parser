import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

TMP_DIR = BASE_DIR / "tmp"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data.db"

TMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

HF_MIRROR_URL = "https://hf-mirror.com"
YOLO_MODEL_REPO = "Armaggheddon/yolo26-document-layout"
YOLO_MODEL_FILE = "yolo26m_doc_layout.pt"
YOLO_IMG_SIZE = 1280

SCAN_TEXT_THRESHOLD = 50

TABLE_STRATEGY = "lines_strict"
