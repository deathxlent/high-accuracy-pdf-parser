@echo off
echo Starting Q4_K_M Table Recognition test at %TIME%
G:\llamacpp\llama-cli.exe -m G:\llamacpp\models\PaddleOCR-VL-1.6.Q4_K_M.gguf --mmproj G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg --temp 0 -p "Table Recognition:" -n 500 --no-display-prompt 1> test\output_ocr_vl_gguf\q4_table_output.txt 2>&1
echo Exit code: %ERRORLEVEL%
echo Done at %TIME%
