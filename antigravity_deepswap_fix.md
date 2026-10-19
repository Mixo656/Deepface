# DeepSwap & Antigravity Pipeline Optimization & Tracking Stabilization Guide

## Executive Summary
This guide provides the exact architectural fixes, environment variables, and drop-in code modules to resolve three major pipeline flaws:
1. **Hardware Misallocation:** Execution defaulting to Intel/AMD integrated GPU (iGPU) and CPU, triggering intense thermal throttling, fan noise, and unviable processing speeds (< 2 FPS).
2. **Boundary/Edge Coordinate Jitter:** Severe flickering and detection dropping whenever faces approach, clip, or cross image borders.
3. **Mouth Motion & Occlusion Artifacts:** Texture tearing, distorted teeth, and visual glitches during wide mouth opening, chewing, biting food, or foreground hand/object occlusions.

---

## Part 1: Enforcing Dedicated GPU (dGPU) & Eliminating Thermal Throttling

### 1.1 Root Causes of CPU/iGPU Fallback
- **Execution Provider Fallback in ONNX Runtime:** By default, ONNX Runtime prioritizes providers in the list passed to it. If `CUDAExecutionProvider` is missing or improperly configured with matching cuDNN/CUDA libraries, it silently falls back to `CPUExecutionProvider` without crashing, causing 100% CPU core utilization.
- **Hybrid Graphics (NVIDIA Optimus / Dynamic Switchable Graphics):** In dual-GPU laptops (e.g., Intel Iris Xe / AMD Radeon + NVIDIA RTX/GTX), the OS treats standard command-line Python scripts as low-power applications and assigns them to the iGPU.
- **PyTorch Default Device:** PyTorch tensors and models remain on CPU unless explicitly shifted to `cuda:0`.

### 1.2 System Environment Variables
Place these declarations at the very beginning of the entry-point script before importing `torch`, `onnxruntime`, or `cv2`:

```python
import os
import sys

# Ensure only the high-performance dedicated NVIDIA GPU is visible
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Enable optimized cuDNN convolution algorithms and prevent CPU thread fighting
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
```

### 1.3 Strict ONNX Runtime Provider Configuration
Ensure your ONNX session setup strictly requires `CUDAExecutionProvider` and enables explicit VRAM memory management:

```python
import onnxruntime as ort

def create_dgpu_session(model_path: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "[CRITICAL] CUDAExecutionProvider not detected! \n"
            "Your environment is defaulting to CPU/iGPU. Verify CUDA Toolkit and install: \n"
            "pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu"
        )

    cuda_options = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,  # Allocate up to 4GB VRAM buffer
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": True,
    }

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    return ort.InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=[("CUDAExecutionProvider", cuda_options)]
    )
```

### 1.4 PyTorch Device & FP16 Precision

```python
import torch

def configure_pytorch_dgpu():
    if not torch.cuda.is_available():
        raise RuntimeError("[ERROR] PyTorch cannot locate NVIDIA CUDA device!")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[SUCCESS] Running on dGPU: {torch.cuda.get_device_name(0)}")
    return device
```

---

## Part 2: Edge-of-Frame Jitter & Boundary Tracking Stabilization

### 2.1 The Cause of Edge Flickering
When a subject's head reaches the borders of the frame, face detectors (e.g. RetinaFace, SCRFD, YOLO-face) fail to find bounding boxes or predict inconsistent 5-point facial landmarks because the receptive field is abruptly cut off. When landmarks fluctuate between frames $t$ and $t+1$, the affine alignment matrix vibrates violently, creating severe jitter.

### 2.2 Fix A: Dynamic Canvas Padding (Reflection)
Expand the image canvas with reflection borders before feeding into the face detector, then translate coordinate landmarks back to original video coordinates:

```python
import cv2
import numpy as np

def pad_frame_for_detection(frame: np.ndarray, pad_ratio: float = 0.20):
    h, w = frame.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    padded = cv2.copyMakeBorder(
        frame, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_REFLECT_101
    )
    return padded, (pad_h, pad_w)

def unpad_landmarks(landmarks: np.ndarray, pad_offsets: tuple) -> np.ndarray:
    pad_h, pad_w = pad_offsets
    corrected = landmarks.copy()
    corrected[:, 0] -= pad_w
    corrected[:, 1] -= pad_h
    return corrected
```

### 2.3 Fix B: Temporal Exponential Smoothing (EMA) with Frame Persistence
Smooth landmark and bounding box transitions across frames. If a face is lost for 1-3 frames at the boundary, maintain the previous valid coordinates rather than dropping the face swap:

```python
class LandmarkStabilizer:
    def __init__(self, alpha: float = 0.65, max_drop_frames: int = 3):
        self.alpha = alpha
        self.max_drop_frames = max_drop_frames
        self.prev_landmarks = None
        self.drop_count = 0

    def update(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None:
            if self.prev_landmarks is not None and self.drop_count < self.max_drop_frames:
                self.drop_count += 1
                return self.prev_landmarks  # Persist face position over temporary drop
            self.prev_landmarks = None
            return None

        self.drop_count = 0
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks.astype(np.float32)
            return self.prev_landmarks

        # Velocity-adaptive smoothing
        movement = np.linalg.norm(landmarks - self.prev_landmarks)
        smoothing_factor = self.alpha if movement < 30.0 else 0.85

        smoothed = smoothing_factor * landmarks + (1.0 - smoothing_factor) * self.prev_landmarks
        self.prev_landmarks = smoothed
        return smoothed
```

---

## Part 3: Mouth Opening, Eating, Biting & Occlusion Fixes

### 3.1 The Cause of Mouth & Biting Distortions
- **Rigid 5-Point Affine Alignment:** Standard face swap pipelines align faces using fixed coordinates (2 eyes, nose tip, 2 mouth corners). When a person opens their mouth widely or bites down on food, the vertical distance from nose to chin changes significantly. The affine matrix distorts the mouth cavity to fit a static closed-mouth template.
- **Foreign Object Overlap:** When biting an object (apple, fork, sandwich, fingers), the generator overlays artificial lips and jaw skin directly over the food item, causing severe tearing and glitchy flickering.

### 3.2 Solution: Semantic Segmentation & Occlusion-Preserving Blending
Use a face parsing network (such as BiSeNet or Face Parsing MobileNet) to generate distinct semantic masks: face skin, lips, inner mouth (teeth + cavity), and foreground occlusions:

```python
import cv2
import numpy as np

def occlusion_aware_blend(
    target_crop: np.ndarray,
    swapped_crop: np.ndarray,
    face_skin_mask: np.ndarray,
    mouth_cavity_mask: np.ndarray = None,
    occlusion_mask: np.ndarray = None,
    feather_amount: int = 19
) -> np.ndarray:
    """
    Blends the swapped face while preserving native mouth cavity, teeth, and foreground objects.
    """
    # Start with binary skin mask (values 0 or 255)
    effective_mask = face_skin_mask.astype(np.float32)

    # Exclude open inner mouth / teeth if distorted
    if mouth_cavity_mask is not None:
        effective_mask = np.clip(effective_mask - mouth_cavity_mask.astype(np.float32), 0.0, 255.0)

    # Exclude foreign occluding items (food, cup, hand, fork)
    if occlusion_mask is not None:
        effective_mask = np.clip(effective_mask - occlusion_mask.astype(np.float32), 0.0, 255.0)

    # Feather edges with Gaussian Blur
    if feather_amount % 2 == 0:
        feather_amount += 1
    soft_mask = cv2.GaussianBlur(effective_mask, (feather_amount, feather_amount), 0) / 255.0
    soft_mask = np.expand_dims(soft_mask, axis=-1)

    # Perform linear alpha blending
    blended = (swapped_crop.astype(np.float32) * soft_mask) + (
        target_crop.astype(np.float32) * (1.0 - soft_mask)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)
```

### 3.3 Dynamic Jaw Alignment Compensation
When 68-point landmarks detect mouth opening (aperture ratio > 0.35), calculate the affine alignment using **only the upper 3 facial points (eyes and nose tip)** rather than anchoring on mouth corners. This prevents jaw stretching while keeping face orientation stable:

```python
def compute_stable_alignment(landmarks_5: np.ndarray) -> np.ndarray:
    # Point 0, 1: Left eye, Right eye | Point 2: Nose tip | Point 3, 4: Mouth corners
    eye_left, eye_right = landmarks_5[0], landmarks_5[1]
    nose_tip = landmarks_5[2]
    mouth_center = (landmarks_5[3] + landmarks_5[4]) / 2.0

    # Compute eye distance and facial height ratio
    eye_dist = np.linalg.norm(eye_left - eye_right)
    mouth_drop = np.linalg.norm(nose_tip - mouth_center)

    # If mouth is wide open, re-weight alignment toward eyes and nose
    if mouth_drop / (eye_dist + 1e-6) > 0.85:
        # Use rigid 3-point estimation (affine partial 2D)
        src_pts = np.float32([eye_left, eye_right, nose_tip])
        # Template canonical reference points for 128x128 crop
        dst_pts = np.float32([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]])
        transform_matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return transform_matrix
    else:
        # Standard 5-point alignment template
        from skimage import transform as trans
        tform = trans.SimilarityTransform()
        dst_5 = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041]
        ], dtype=np.float32)
        tform.estimate(landmarks_5, dst_5)
        return tform.params[0:2, :]
```

---

## Part 4: Verification & Diagnostic Checklist

1. **Confirm Dedicated GPU Compute:**
   Open a terminal and execute:
   ```bash
   nvidia-smi -l 1
   ```
   - Confirm `python` is running under GPU Processes.
   - GPU compute utilization should stay within **75% – 98%** during batch processing.
   - CPU temperatures should normalize (typically dropping from 90°C+ down to 55°C–70°C).

2. **Verify Edge Tracking:**
   - Process a video where the subject moves partially off-screen (left/right border).
   - Verify there is zero frame-to-frame flicker or sudden square boundary clipping.

3. **Verify Mouth & Occlusion Fidelity:**
   - Test on a video with rapid speech, laughing, or food consumption.
   - Ensure the inner teeth cavity and occluding items (utensils, food, fingers) remain crisp without glitchy texture bleeding.
