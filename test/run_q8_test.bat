@echo off
setlocal
cd /d C:\ws\high accuracy pdf parser
echo Starting Q8_0 test at %TIME% > test\output_ocr_vl_q8\run_log.txt
G:\llamacpp\llama-cli.exe -m G:\llamacpp\models\PaddleOCR-VL-1.6.Q8_0.gguf --mmproj G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg --temp 0 -p OCR: -n 200 --no-display-prompt 1> test\output_ocr_vl_q8\llama_cli_raw.txt 2> test\output_ocr_vl_q8\stderr.txt
echo Exit code: %ERRORLEVEL% >> test\output_ocr_vl_q8\run_log.txt
echo Done at %TIME% >> test\output_ocr_vl_q8\run_log.txt
