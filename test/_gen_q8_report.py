"""Generate Q8_0 report using the v3 report generator."""
import sys
sys.path.insert(0, r'C:\ws\high accuracy pdf parser\test')
from _gen_vl_report3 import generate_report

raw_path = r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_q8\extracted.txt'
# We need the raw binary file, not the extracted text
raw_path = r'C:\ws\high accuracy pdf parser\test\output_ocr_vl_gguf\q8_raw.txt'

generate_report(raw_path, 'output_ocr_vl_q8')
