import os
import cv2
import numpy as np

from core.gpu_config import create_dgpu_session

class CodeFormer:
    def __init__(self, models_dir):
        model_path = os.path.join(models_dir, 'codeformer.onnx')
        if not os.path.exists(model_path):
            raise Exception(f"CodeFormer model not found at {model_path}")
            
        self.model_path = model_path
        # Initialize ONNX Runtime with dedicated NVIDIA GPU acceleration
        self.session = create_dgpu_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        
        # Check for weight/fidelity input
        self.has_weight = len(self.session.get_inputs()) > 1
        if self.has_weight:
            self.weight_name = self.session.get_inputs()[1].name

    def restore(self, face_img, fidelity=0.5):
        """
        Enhances a face image crop using CodeFormer.
        fidelity (0.0 to 1.0): 1.0 means high resemblance, 0.0 means high quality.
        """
        # 1. Preprocessing
        # Convert BGR to RGB
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        # Resize to 512x512
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        # 2. Inference with automatic fallback
        inputs = {self.input_name: img}
        if self.has_weight:
            w = np.array([1.0 - fidelity], dtype=np.double)
            inputs[self.weight_name] = w

        try:
            outputs = self.session.run(None, inputs)
            output = outputs[0][0]
            output = np.transpose(output, (1, 2, 0))
            output = (output * 0.5 + 0.5) * 255.0
            output = np.clip(output, 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Prevent video processing freeze; use high-quality Lanczos4 upscaled crop
            return cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # 4. Smooth fidelity blending to eliminate codebook snapping / jitter
        if fidelity > 0.0:
            base_512 = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            blend_weight = float(np.clip(fidelity * 0.35, 0.0, 0.5))
            output = cv2.addWeighted(output, 1.0 - blend_weight, base_512, blend_weight, 0)

        return output
