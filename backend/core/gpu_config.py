"""
Centralized GPU environment configuration and strict ONNX Runtime session factory.

Sets environment variables to enforce dedicated NVIDIA GPU usage and provides
create_dgpu_session() which requires CUDAExecutionProvider with explicit VRAM
management. Falls back gracefully to CPU with a loud warning when CUDA is
genuinely unavailable (e.g. Google Colab CPU runtime, CI).
"""

import os
import logging

logger = logging.getLogger(__name__)

_configured = False

# Auto-configure environment immediately upon importing gpu_config
# so that all CUDA/cuDNN DLLs and device configs are registered before any ORT/Torch calls.


def _register_nvidia_dlls():
    """
    On Windows, register all DLL paths from installed nvidia-* packages
    (e.g., nvidia-cublas-cu12, nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12)
    so that onnxruntime-gpu can locate cublasLt64_12.dll, cudnn64_9.dll, etc.
    """
    import sys
    import site

    dirs_to_add = []
    # Search site-packages
    site_dirs = site.getsitepackages() + [site.getusersitepackages()]
    for base in site_dirs:
        nvidia_root = os.path.join(base, "nvidia")
        if os.path.isdir(nvidia_root):
            for pkg in os.listdir(nvidia_root):
                pkg_dir = os.path.join(nvidia_root, pkg)
                for sub in ("bin", "lib"):
                    cand = os.path.join(pkg_dir, sub)
                    if os.path.isdir(cand) and cand not in dirs_to_add:
                        dirs_to_add.append(cand)

    # Register each directory with os.add_dll_directory and prepend to PATH
    for d in dirs_to_add:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
    if dirs_to_add:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = ";".join(dirs_to_add) + ";" + current_path
        logger.info(f"[GPU_CONFIG] Registered {len(dirs_to_add)} NVIDIA CUDA/cuDNN DLL directories.")


def configure_environment():
    """
    Set environment variables that force the OS and ONNX/PyTorch to use the
    dedicated NVIDIA GPU. Must be called BEFORE importing torch, onnxruntime,
    or cv2 in the main entry point.
    """
    global _configured
    if _configured:
        return
    _configured = True

    # Register NVIDIA pip package DLL directories on Windows
    _register_nvidia_dlls()

    # Ensure only the high-performance dedicated NVIDIA GPU is visible
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    # Prevent CPU thread over-subscription and OpenBLAS thread allocation exhaustion
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    logger.info("[GPU_CONFIG] Environment configured for dedicated GPU.")


def get_ort_providers(vram_limit_mb: int = 1400):
    """
    Return the best available ONNX Runtime providers list.
    Prefers CUDAExecutionProvider with optimised settings; falls back to CPU.
    """
    import onnxruntime as ort

    available = ort.get_available_providers()

    if "CUDAExecutionProvider" in available:
        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "cudnn_conv_use_max_workspace": "0",
            "do_copy_in_default_stream": True,
        }
        logger.info("[GPU_CONFIG] Using CUDAExecutionProvider (dGPU, dynamic shared VRAM).")
        return [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]

    logger.warning(
        "[GPU_CONFIG] CUDAExecutionProvider NOT available! "
        "Falling back to CPUExecutionProvider. Performance will be degraded. "
        "To fix: pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu"
    )
    return ["CPUExecutionProvider"]


def create_dgpu_session(model_path: str, vram_limit_mb: int = 1400):
    """
    Create an onnxruntime.InferenceSession with strict GPU preference
    and graph-level optimisations enabled.
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = get_ort_providers(vram_limit_mb=vram_limit_mb)

    session = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers
    )

    active = session.get_providers()
    logger.info(f"[GPU_CONFIG] Session providers for {os.path.basename(model_path)}: {active}")
    return session


# Automatically configure environment upon import
configure_environment()
