@echo off
echo Starting Q8_0 test at %TIME%
G:\llamacpp\llama-cli.exe -m G:\llamacpp\models\PaddleOCR-VL-1.6.Q8_0.gguf --mmproj G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg --temp 0 -p "Table Recognition:" -n 1000 --no-display-prompt -ngl 99 1> test\output_ocr_vl_gguf\q8_raw.txt 2>&1
echo Exit: %ERRORLEVEL% at %TIME%
