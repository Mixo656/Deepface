import os
import cv2
import numpy as np

from core.gpu_config import create_dgpu_session

class Restorer:
    def __init__(self, models_dir):
        model_path = os.path.join(models_dir, 'GFPGANv1.4.onnx')
        if not os.path.exists(model_path):
            raise Exception(f"Restorer model not found at {model_path}")
            
        self.model_path = model_path
        # Initialize ONNX Runtime with dedicated NVIDIA GPU acceleration
        self.session = create_dgpu_session(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def restore(self, face_img, fidelity=0.5):
        """
        Enhances a face image crop using GFPGAN v1.4.
        Returns high-resolution 512x512 BGR face with smooth fidelity blending.
        """
        # 1. Preprocessing
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        # 2. Inference with automatic OOM fallback
        try:
            outputs = self.session.run(None, {self.input_name: img})
            output = outputs[0][0]
            output = np.transpose(output, (1, 2, 0))
            output = (output + 1.0) / 2.0
            output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        # 4. Smooth fidelity blending to maintain temporal consistency
        if fidelity > 0.0:
            base_512 = cv2.resize(face_img, (512, 512), interpolation=cv2.INTER_LINEAR)
            blend_weight = float(np.clip(fidelity * 0.4, 0.0, 0.5))
            output = cv2.addWeighted(output, 1.0 - blend_weight, base_512, blend_weight, 0)

        return output
