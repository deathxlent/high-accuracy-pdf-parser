"""Run PaddleOCR-VL GGUF model and capture output with timing."""
import subprocess, sys, os, time, re

MODEL_DIR = r"G:\llamacpp\models"
MMPROJ = os.path.join(MODEL_DIR, "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
IMAGE = r"C:\ws\high accuracy pdf parser\tmp\42e59745cdb54b6fb2c635d7c11dbd43\page_1.jpg"
LLAMA_CLI = r"G:\llamacpp\llama-cli.exe"

BASE_OUT = r"C:\ws\high accuracy pdf parser\test"

def run_test(model_name, model_file, output_subdir, quant_label, model_size):
    """Run llama-cli with given model, capture output and timing."""
    model_path = os.path.join(MODEL_DIR, model_file)
    out_dir = os.path.join(BASE_OUT, output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None
    model_mb = os.path.getsize(model_path) // (1024*1024)

    cmd = [
        LLAMA_CLI,
        "-m", model_path,
        "--mmproj", MMPROJ,
        "--image", IMAGE,
        "--temp", "0",
        "-p", "OCR:",
        "--no-display-prompt"
    ]

    print(f"=== Running {model_name} ({model_mb} MB) ===")
    print(f"Command: {' '.join(cmd)}")
    sys.stdout.flush()

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=600,  # 10 min timeout
            text=False  # keep as bytes
        )
        elapsed = time.time() - start
        stdout = result.stdout
        stderr = result.stderr
        retcode = result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"TIMEOUT after {elapsed:.0f}s")
        return None
    except Exception as e:
        elapsed = time.time() - start
        print(f"ERROR: {e}")
        return None

    # Save raw output
    with open(os.path.join(out_dir, "llama_cli_raw.txt"), "wb") as f:
        f.write(stdout)
    if stderr:
        with open(os.path.join(out_dir, "stderr.txt"), "w", encoding="utf-8", errors="replace") as f:
            f.write(stderr.decode("utf-8", errors="replace"))

    # Save timing
    with open(os.path.join(out_dir, "elapsed.txt"), "w") as f:
        f.write(f"{elapsed:.1f}")

    print(f"Done in {elapsed:.1f}s (exit code {retcode})")
    print(f"Stdout: {len(stdout)} bytes, Stderr: {len(stderr)} bytes")

    # Show preview
    text = stdout.decode("utf-8", errors="replace")
    # Extract OCR response part (after "OCR:")
    clean = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)
    if "OCR:" in clean:
        resp = clean.split("OCR:", 1)[1].strip()
    else:
        resp = clean.strip()
    resp = re.sub(r'\n\[ Prompt:.*$', '', resp, flags=re.DOTALL).strip()
    resp = re.sub(r'\n>.*$', '', resp, flags=re.DOTALL).strip()

    print(f"OCR response ({len(resp)} chars):")
    print(resp[:200])
    print("---")

    return {
        "stdout": stdout,
        "elapsed": elapsed,
        "retcode": retcode,
        "resp": resp,
        "out_dir": out_dir,
        "model_name": model_name,
        "model_size": model_size,
        "quant_label": quant_label,
        "model_file": model_file,
        "model_mb": model_mb
    }


def generate_report(result):
    """Generate HTML report using the _gen_vl_report3 module."""
    if result is None:
        return

    # Build parameters
    ocr_text = result["resp"]
    out_dir = result["out_dir"]
    model_name = result["model_name"]
    model_size = result["model_size"]
    quant_label = result["quant_label"]
    model_file = result["model_file"]
    elapsed = result["elapsed"]

    # Extract performance from raw output
    raw_text = result["stdout"].decode("utf-8", errors="replace")
    perf = re.search(r'\[ Prompt:\s*([\d.]+)\s*t/s \| Generation:\s*([\d.]+)\s*t/s', raw_text)

    # Import the report builder from _gen_vl_report3
    sys.path.insert(0, os.path.join(BASE_OUT))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "report_gen",
        os.path.join(BASE_OUT, "_gen_vl_report3.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.build_report(
        ocr_text, perf,
        model_name=model_file,
        model_size=model_size,
        total_size=f"{result['model_mb'] + 841} MB",
        quant=quant_label,
        title_suffix=f"GGUF ({quant_label})",
        output_dir=os.path.basename(out_dir),
        parse_time_s=elapsed
    )


if __name__ == "__main__":
    # Determine which test to run
    test_name = sys.argv[1] if len(sys.argv) > 1 else "all"

    configs = []

    if test_name in ("f16", "all"):
        configs.append({
            "model_name": "PaddleOCR-VL-1.6 F16",
            "model_file": "PaddleOCR-VL-1.6.f16.gguf",
            "output_subdir": "output_ocr_vl_f16",
            "quant_label": "F16",
            "model_size": "892 MB"
        })

    if test_name in ("q8", "q8_0", "all"):
        configs.append({
            "model_name": "PaddleOCR-VL-1.6 Q8_0",
            "model_file": "PaddleOCR-VL-1.6.Q8_0.gguf",
            "output_subdir": "output_ocr_vl_q8",
            "quant_label": "Q8_0",
            "model_size": "473 MB"
        })

    for cfg in configs:
        result = run_test(**cfg)
        if result:
            generate_report(result)
        print()
