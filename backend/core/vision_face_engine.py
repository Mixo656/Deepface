"""
AI Vision Facial Engine (106-Point Dense Landmark & Micro-Movement Tracker)
==========================================================================
Provides:
1. Dense 106-point facial tracking on dedicated NVIDIA GPU (RTX 3050).
2. Micro-movement analysis (blinking, eye aperture, mouth opening, speech phonemes, jawline motion).
3. Rigid cranial 5-landmark alignment to eliminate mouth/jaw speech wobble.
4. Dynamic 33-point jawline anatomical boundary mask generation.
5. Micro-speech oral cavity & teeth detail preservation to eliminate mouth jittering.
6. Temporal One-Euro landmark filtering across frames.
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import Optional, Tuple, Dict, Any

from core.gpu_config import create_dgpu_session
from core.stabilizer import OneEuroFilter


class Dense106VisionTracker:
    """
    High-precision 106-point facial landmark vision model running on dGPU.
    Tracks exact jawline, eyelids, pupils, eyebrows, and inner/outer lip contours.
    """

    def __init__(self, models_dir: str):
        model_path = os.path.join(models_dir, "2d106det.onnx")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"106-point landmark model not found at {model_path}")

        # Initialize ONNX inference session with strict GPU preference (256MB ceiling)
        self.session = create_dgpu_session(model_path, vram_limit_mb=256)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (192, 192)
        self.input_mean = 127.5
        self.input_std = 128.0

        # One-Euro temporal filter across all 106 landmark points
        self.filter = OneEuroFilter(min_cutoff=0.03, beta=0.5)
        self.prev_lmk: Optional[np.ndarray] = None
        self.drop_count = 0
        self.max_drop_frames = 5

    def reset(self):
        """Reset temporal state between video clips."""
        self.filter.reset()
        self.prev_lmk = None
        self.drop_count = 0

    def detect_landmarks(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Detects 106 landmarks for a given face bounding box.
        Returns: (106, 2) float32 coordinates in original frame space.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        center = np.array([(bbox[2] + bbox[0]) / 2.0, (bbox[3] + bbox[1]) / 2.0], dtype=np.float32)
        scale = self.input_size[0] / (max(w, h) * 1.5)

        # Affine transform to 192x192
        M = cv2.getRotationMatrix2D((center[0], center[1]), 0, scale)
        M[0, 2] += (self.input_size[0] / 2.0 - center[0])
        M[1, 2] += (self.input_size[1] / 2.0 - center[1])

        aimg = cv2.warpAffine(img, M, self.input_size, borderMode=cv2.BORDER_REFLECT_101)
        blob = cv2.dnn.blobFromImage(
            aimg,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )

        pred = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        pred = pred.reshape((-1, 2))
        pred += 1.0
        pred *= (self.input_size[0] / 2.0)

        # Invert affine back to frame coordinate space
        IM = cv2.invertAffineTransform(M)
        pts_hom = np.hstack([pred, np.ones((pred.shape[0], 1), dtype=np.float32)])
        landmarks = np.dot(pts_hom, IM.T)

        return landmarks.astype(np.float32)

    def track(self, img: np.ndarray, bbox: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Detects and temporally smooths 106 landmarks across video frames.
        """
        if bbox is None:
            if self.prev_lmk is not None and self.drop_count < self.max_drop_frames:
                self.drop_count += 1
                return self.prev_lmk.copy()
            self.prev_lmk = None
            self.filter.reset()
            return None

        raw_lmk = self.detect_landmarks(img, bbox)
        self.drop_count = 0

        if self.prev_lmk is None:
            self.prev_lmk = raw_lmk.copy()
            self.filter.filter(raw_lmk)
            return raw_lmk

        smoothed = self.filter.filter(raw_lmk)
        self.prev_lmk = smoothed.copy()
        return smoothed


class MicroMovementAnalyzer:
    """
    Analyzes micro-facial movements from 106 landmarks:
    - Eye blink / eyelid aperture (EAR)
    - Mouth cavity opening / speaking phonemes (MAR)
    - Jaw articulation & chin drop
    """

    @staticmethod
    def analyze(lmk106: np.ndarray) -> Dict[str, Any]:
        # Left eye: upper [68, 69], lower [72, 73], corners [66, 70]
        left_eye_h = float(np.linalg.norm(lmk106[68] - lmk106[72]) + np.linalg.norm(lmk106[69] - lmk106[73])) / 2.0
        left_eye_w = float(np.linalg.norm(lmk106[66] - lmk106[70])) + 1e-6
        left_ear = left_eye_h / left_eye_w

        # Right eye: upper [77, 78], lower [81, 82], corners [75, 79]
        right_eye_h = float(np.linalg.norm(lmk106[77] - lmk106[81]) + np.linalg.norm(lmk106[78] - lmk106[82])) / 2.0
        right_eye_w = float(np.linalg.norm(lmk106[75] - lmk106[79])) + 1e-6
        right_ear = right_eye_h / right_eye_w

        avg_ear = float((left_ear + right_ear) / 2.0)

        # Inner mouth aperture (points 97 to 105)
        # Inner upper: 98/99, inner lower: 103/104, corners: 97, 101
        mouth_inner_h = float(
            np.linalg.norm(lmk106[98] - lmk106[104]) + np.linalg.norm(lmk106[99] - lmk106[103])
        ) / 2.0
        mouth_inner_w = float(np.linalg.norm(lmk106[97] - lmk106[101])) + 1e-6
        mar = float(mouth_inner_h / mouth_inner_w)

        # Jaw width (points 0 to 32)
        jaw_width = float(np.linalg.norm(lmk106[0] - lmk106[32]))

        # Chin position relative to eye midpoint
        mid_eyes = (lmk106[66:75].mean(axis=0) + lmk106[75:84].mean(axis=0)) / 2.0
        chin_pt = lmk106[16]
        jaw_drop = float(np.linalg.norm(chin_pt - mid_eyes))

        return {
            "eye_aperture": avg_ear,
            "is_blinking": bool(avg_ear < 0.16),
            "mouth_aperture": mar,
            "is_mouth_open": bool(mar > 0.14),
            "jaw_width": jaw_width,
            "jaw_drop": jaw_drop,
        }


class RigidCranialAligner:
    """
    Computes ultra-stable 5 landmarks anchored to the rigid human skull cranium
    (eyes, nose bridge, and cranial baseline).
    Prevents mouth movement (speaking, laughing, chewing) from wobbling the affine crop.
    """

    @staticmethod
    def align(lmk106: np.ndarray, default_kps: Optional[np.ndarray] = None) -> np.ndarray:
        if lmk106 is None:
            return default_kps

        # Left & right eye centers from canthi and pupils (stable cranium)
        left_eye = lmk106[66:75].mean(axis=0)
        right_eye = lmk106[75:84].mean(axis=0)

        # Upper nose bridge (rigid, immovable skull bone)
        nose_bridge = lmk106[53:57].mean(axis=0)
        nose_tip = lmk106[60]

        # Inter-ocular cranial vector
        eye_vec = right_eye - left_eye
        eye_dist = float(np.linalg.norm(eye_vec))
        if eye_dist < 1e-3:
            return default_kps if default_kps is not None else np.zeros((5, 2), dtype=np.float32)

        # Unit perpendicular down vector
        down_vec = np.array([-eye_vec[1], eye_vec[0]], dtype=np.float32) / eye_dist
        eye_unit = eye_vec / eye_dist

        # Rigid cranial mouth center (Golden ratio distance: 1.15 * eye_dist)
        mid_eyes = (left_eye + right_eye) / 2.0
        rigid_mouth_center = mid_eyes + down_vec * (eye_dist * 1.15)

        # Rigid mouth corners
        rigid_left_mouth = rigid_mouth_center - eye_unit * (eye_dist * 0.35)
        rigid_right_mouth = rigid_mouth_center + eye_unit * (eye_dist * 0.35)

        # Blend with actual outer mouth corners with heavy cranial rigidity weight (75% rigid, 25% actual)
        actual_left_mouth = lmk106[84]
        actual_right_mouth = lmk106[90]

        stable_left_mouth = rigid_left_mouth * 0.75 + actual_left_mouth * 0.25
        stable_right_mouth = rigid_right_mouth * 0.75 + actual_right_mouth * 0.25

        # Assembled stable 5 landmarks: [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
        stable_5kps = np.array([
            left_eye,
            right_eye,
            nose_tip,
            stable_left_mouth,
            stable_right_mouth
        ], dtype=np.float32)

        return stable_5kps


class DenseAnatomicalMaskBuilder:
    """
    Builds sub-pixel accurate facial boundary masks directly from the
    33-point jawline and forehead contour, with intelligent inner mouth
    preservation for speech, teeth, and tongue.
    """

    @staticmethod
    def build_anatomical_mask(
        crop_size: int,
        pts_crop: np.ndarray,
        micro_state: Optional[Dict[str, Any]] = None,
        feather_ratio: float = 0.07
    ) -> np.ndarray:
        """
        Builds full-face anatomical mask in aligned crop space from tracked 106 points.
        The lower perimeter exactly matches the 33-point jawline and chin contour.
        """
        mask = np.zeros((crop_size, crop_size), dtype=np.float32)

        if pts_crop is None or len(pts_crop) < 106:
            # Fallback to normalized contour if landmarks missing
            return DenseAnatomicalMaskBuilder._build_fallback_mask(crop_size)

        # 1. Jawline: 33 points (pts_crop[0:33])
        jaw_pts = pts_crop[0:33]

        # 2. Forehead projection from eyebrows and mid-eyes
        eye_l = pts_crop[66:75].mean(axis=0)
        eye_r = pts_crop[75:84].mean(axis=0)
        mid_eyes = (eye_l + eye_r) / 2.0
        chin = pts_crop[16]
        face_h = float(np.linalg.norm(chin - mid_eyes)) + 1e-4

        # Upward cranial vector
        up_vec = (mid_eyes - chin) / face_h

        brow_r = pts_crop[43:53]
        brow_l = pts_crop[33:43]

        forehead_pts = []
        # From right ear (pt 32) up across right eyebrow
        for pt in brow_r[::-1]:
            forehead_pts.append(pt + up_vec * (face_h * 0.44))
        # Forehead apex
        forehead_pts.append(mid_eyes + up_vec * (face_h * 0.62))
        # Across left eyebrow down to left ear (pt 0)
        for pt in brow_l[::-1]:
            forehead_pts.append(pt + up_vec * (face_h * 0.44))

        full_contour = np.vstack([jaw_pts, np.array(forehead_pts, dtype=np.float32)])
        cv2.fillPoly(mask, [full_contour.astype(np.int32).reshape((-1, 1, 2))], 1.0)

        # Smooth border feathering
        ksize = int(crop_size * feather_ratio) | 1
        ksize = max(ksize, 5)
        sigma = ksize / 3.0
        mask = cv2.GaussianBlur(mask, (ksize, ksize), sigma)

        return mask

    @staticmethod
    def build_mouth_cavity_mask(
        crop_size: int,
        pts_crop: np.ndarray,
        micro_state: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Builds a feathered inner oral cavity mask for preserving real teeth and speech phonemes.
        Returns: (crop_size, crop_size, 1) float32 mask with values [0.0, 1.0].
        """
        if not micro_state.get("is_mouth_open", False):
            return None

        # Inner lip contour: points 97 to 105
        inner_mouth_pts = pts_crop[97:106].astype(np.int32).reshape((-1, 1, 2))

        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        cv2.fillPoly(mask, [inner_mouth_pts], 1.0)

        # Feather the teeth/oral boundary smoothly
        ksize = int(crop_size * 0.04) | 1
        ksize = max(ksize, 3)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)
        return np.expand_dims(mask, axis=-1)

    @staticmethod
    def build_eyelid_mask(
        crop_size: int,
        pts_crop: np.ndarray,
        micro_state: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Builds a feathered eyelid mask for natural eye blinking preservation.
        """
        if not micro_state.get("is_blinking", False):
            return None

        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        left_eye_pts = pts_crop[66:74].astype(np.int32).reshape((-1, 1, 2))
        right_eye_pts = pts_crop[75:83].astype(np.int32).reshape((-1, 1, 2))

        cv2.fillPoly(mask, [left_eye_pts], 1.0)
        cv2.fillPoly(mask, [right_eye_pts], 1.0)

        ksize = int(crop_size * 0.03) | 1
        ksize = max(ksize, 3)
        mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)
        return np.expand_dims(mask, axis=-1)

    @staticmethod
    def _build_fallback_mask(crop_size: int) -> np.ndarray:
        mask = np.zeros((crop_size, crop_size), dtype=np.float32)
        contour_norm = np.array([
            [0.44, 0.10],
            [0.22, 0.22],
            [0.15, 0.50],
            [0.20, 0.78],
            [0.32, 0.92],
            [0.44, 0.97],
            [0.56, 0.92],
            [0.68, 0.78],
            [0.73, 0.50],
            [0.66, 0.22],
        ], dtype=np.float32)
        contour_pts = (contour_norm * crop_size).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [contour_pts], 1.0)
        ksize = int(crop_size * 0.08) | 1
        return cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)
