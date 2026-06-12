@echo off
echo Starting Q4_K_M v2 at %TIME% > test\output_ocr_vl_gguf\run_v2_log.txt
G:\llamacpp\llama-cli.exe -m G:\llamacpp\models\PaddleOCR-VL-1.6.Q4_K_M.gguf --mmproj G:\llamacpp\models\PaddleOCR-VL-1.6-GGUF-mmproj.gguf --image tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg --temp 0 -p "Table Recognition:" -n 800 --no-display-prompt 1> test\output_ocr_vl_gguf\llama_cli_raw_v2.txt 2> test\output_ocr_vl_gguf\stderr_v2.txt
echo Exit: %ERRORLEVEL% at %TIME% >> test\output_ocr_vl_gguf\run_v2_log.txt
