import os
import requests
from tqdm import tqdm

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# Warning: These models are for educational/research purposes.
INSWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
GFPGAN_URL = "https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx"
CODEFORMER_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/codeformer.onnx"

# Minimum valid file sizes (bytes) — anything smaller is a corrupt/error page download
MIN_SIZES = {
    "inswapper_128.onnx": 500_000_000,   # ~529 MB
    "GFPGANv1.4.onnx":   300_000_000,    # ~324 MB
    "codeformer.onnx":    350_000_000,    # ~359 MB
}

def download_file(url, dest_path):
    filename = os.path.basename(dest_path)
    min_size = MIN_SIZES.get(filename, 10 * 1024 * 1024)

    if os.path.exists(dest_path):
        current_size = os.path.getsize(dest_path)
        if current_size >= min_size:
            print(f"[INFO] File already valid ({current_size // (1024*1024)} MB): {dest_path}")
            return
        else:
            print(f"[WARNING] {filename} exists but is corrupt ({current_size} bytes). Re-downloading...")
            try:
                os.remove(dest_path)
            except Exception:
                pass

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"[INFO] Downloading {url} to {dest_path}...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 Megabyte

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    temp_path = dest_path + ".tmp"
    with open(temp_path, 'wb') as file:
        for data in response.iter_content(block_size):
            if data:
                progress_bar.update(len(data))
                file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("[ERROR] Download incomplete.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return

    # Ensure old dest is gone, then rename temp -> dest
    import gc, time, shutil
    gc.collect()
    time.sleep(0.5)
    if os.path.exists(dest_path):
        try:
            os.remove(dest_path)
        except Exception:
            pass

    for attempt in range(5):
        try:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            os.rename(temp_path, dest_path)
            break
        except PermissionError:
            time.sleep(1.0)
    else:
        # Fallback to copy + remove
        shutil.copy2(temp_path, dest_path)
        try:
            os.remove(temp_path)
        except Exception:
            pass

    print(f"[INFO] Download completed: {dest_path}")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    inswapper_path = os.path.join(MODELS_DIR, "inswapper_128.onnx")
    gfpgan_path = os.path.join(MODELS_DIR, "GFPGANv1.4.onnx")
    codeformer_path = os.path.join(MODELS_DIR, "codeformer.onnx")

    download_file(INSWAPPER_URL, inswapper_path)
    download_file(GFPGAN_URL, gfpgan_path)
    download_file(CODEFORMER_URL, codeformer_path)
    print("[INFO] Model download check complete.")
