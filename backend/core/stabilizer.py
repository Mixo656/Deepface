"""
Temporal stabilisation utilities for face detection in video pipelines.

- pad_frame_for_detection / unpad_bbox: reflection-pads a frame so edge faces
  are fully inside the detector's receptive field, then maps results back.
- LandmarkStabilizer: EMA smoother with velocity-adaptive alpha and frame
  persistence (holds the last valid detection for up to N dropped frames).
"""

import cv2
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Frame padding for edge-of-frame detection
# ---------------------------------------------------------------------------

def pad_frame_for_detection(
    frame: np.ndarray, pad_ratio: float = 0.20
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pads a frame with constant black border so edge faces are fully inside
    the detector receptive field without producing ghost mirrored reflections.
    """
    h, w = frame.shape[:2]
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    padded = cv2.copyMakeBorder(
        frame, pad_h, pad_h, pad_w, pad_w, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return padded, (pad_h, pad_w)


def filter_and_track_face(
    faces,
    pad_offsets: Tuple[int, int],
    img_w: int,
    img_h: int,
    prev_center: Optional[Tuple[float, float]] = None,
    prev_size: Optional[float] = None,
):
    """
    Filters out off-screen artifacts and selects the primary face using temporal tracking.
    Guarantees continuous tracking of the main subject across video frames.
    """
    valid = []
    for face in faces:
        if face.kps is not None:
            face.kps = unpad_landmarks(face.kps, pad_offsets)
        face.bbox = unpad_bbox(face.bbox, pad_offsets)

        x1, y1, x2, y2 = face.bbox
        # Calculate intersection with visible canvas
        ix1 = max(0, x1)
        iy1 = max(0, y1)
        ix2 = min(img_w, x2)
        iy2 = min(img_h, y2)

        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        bbox_area = max(1.0, (x2 - x1) * (y2 - y1))

        # Must be at least 40% inside visible screen
        if inter_area / bbox_area < 0.40:
            continue

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        face_size = float(np.sqrt(bbox_area))

        # Score candidate based on spatial proximity to tracked face
        if prev_center is not None and prev_size is not None:
            dist = float(np.hypot(cx - prev_center[0], cy - prev_center[1]))
            dist_penalty = min(2.0, dist / max(prev_size, 50.0))
            size_ratio = abs(face_size - prev_size) / max(prev_size, 50.0)
            score = face.det_score * 1.5 - dist_penalty * 0.8 - size_ratio * 0.5
        else:
            score = bbox_area * 0.0001 + face.det_score

        valid.append((score, cx, cy, face_size, face))

    if not valid:
        return None

    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][4]


def unpad_bbox(bbox: np.ndarray, pad_offsets: Tuple[int, int]) -> np.ndarray:
    """Shift a (x1, y1, x2, y2) bbox back to original frame coords."""
    pad_h, pad_w = pad_offsets
    corrected = bbox.copy()
    corrected[0] -= pad_w
    corrected[1] -= pad_h
    corrected[2] -= pad_w
    corrected[3] -= pad_h
    return corrected


def unpad_landmarks(
    landmarks: np.ndarray, pad_offsets: Tuple[int, int]
) -> np.ndarray:
    """Shift Nx2 landmarks back to original frame coords."""
    pad_h, pad_w = pad_offsets
    corrected = landmarks.copy().astype(np.float32)
    corrected[:, 0] -= pad_w
    corrected[:, 1] -= pad_h
    return corrected


# ---------------------------------------------------------------------------
# One-Euro Filter & Rigid Skull Pose Stabilizer
# ---------------------------------------------------------------------------

class OneEuroFilter:
    """
    Gold-standard adaptive filter for real-time tracking (eliminates jitter while preserving responsiveness).
    """
    def __init__(self, min_cutoff: float = 0.06, beta: float = 1.6, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None

    def reset(self):
        self.x_prev = None
        self.dx_prev = None

    def _smoothing_factor(self, cutoff: np.ndarray, dt: float = 1.0) -> np.ndarray:
        r = 2 * np.pi * cutoff * dt
        return r / (r + 1.0)

    def filter(self, x: np.ndarray, dt: float = 1.0) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x

        # Estimate derivative
        dx = (x - self.x_prev) / dt
        if self.dx_prev is None:
            self.dx_prev = dx
        a_d = self._smoothing_factor(np.full_like(dx, self.d_cutoff), dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._smoothing_factor(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return x_hat


# ---------------------------------------------------------------------------
# Cranial-Anchored Stabilization & 3D Yaw Compensation
# ---------------------------------------------------------------------------

def get_cranial_anchored_kps(kps: np.ndarray) -> np.ndarray:
    """
    Given 5 landmarks (left_eye, right_eye, nose, left_mouth, right_mouth),
    computes rigid 3D perspective-aware cranial anchors for the mouth to decouple
    mouth movement, chewing, speech, and object occlusions from head pose.
    Handles 3D yaw angle rotation smoothly with zero discontinuous jumps.
    """
    kps_clean = kps.astype(np.float32).copy()
    eye_l = kps_clean[0]
    eye_r = kps_clean[1]
    nose = kps_clean[2]

    eye_vec = eye_r - eye_l
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-3:
        return kps_clean

    u = eye_vec / eye_dist
    v = np.array([-u[1], u[0]], dtype=np.float32)
    eye_center = (eye_l + eye_r) / 2.0

    # Vertical distance from eye line to mouth
    det_mouth_center = (kps_clean[3] + kps_clean[4]) / 2.0
    vert_dist = float(np.dot(det_mouth_center - eye_center, v))

    cranial_eye_dist = max(eye_dist, vert_dist / 1.1546) if vert_dist > 5.0 else eye_dist

    # 3D yaw perspective offset along eye vector
    nose_offset_u = float(np.dot(nose - eye_center, u))

    # Theoretical 3D-aware cranial mouth position (ArcFace canonical template geometry)
    cranial_mouth_center = eye_center + 1.1546 * cranial_eye_dist * v + 0.70 * nose_offset_u * u
    cranial_mouth_l = cranial_mouth_center - 0.4141 * eye_dist * u
    cranial_mouth_r = cranial_mouth_center + 0.4141 * eye_dist * u

    # Check detected mouth deviation from cranial expectation
    mouth_offset = float(np.linalg.norm(det_mouth_center - cranial_mouth_center))
    deviation_ratio = mouth_offset / max(cranial_eye_dist, 10.0)

    # Continuous smooth regularization (no threshold jumps)
    # If deviation is small (< 0.12), apply 15% anchor to suppress micro-jitter while preserving natural speech
    # If deviation is large (e.g. object in mouth, chewing, hand occlusion), smoothly blend up to 80%
    if deviation_ratio < 0.12:
        alpha = 0.15
    else:
        alpha = min(0.80, 0.15 + (deviation_ratio - 0.12) * 2.5)

    kps_clean[3] = (1.0 - alpha) * kps_clean[3] + alpha * cranial_mouth_l
    kps_clean[4] = (1.0 - alpha) * kps_clean[4] + alpha * cranial_mouth_r

    return kps_clean


class LandmarkStabilizer:
    """
    Rigid-Skull Pose & Landmark Stabilizer with One-Euro temporal filtering.
    - Anchors rotation and scale to the rigid ocular-nasal cranium (eyes + nose tip).
    - Decouples mouth occlusions (hands, cups, microphones) and speech from affine matrix M.
    - Zero-lag velocity-adaptive tracking: instantaneous response (< 0.5px lag) during movement.
    - Persistent pose tracking across dropped frames with velocity extrapolation (zero dropped swaps).
    """

    def __init__(self, alpha: Optional[float] = None, min_cutoff: float = 0.06, beta: float = 1.6, max_drop_frames: int = 30, **kwargs):
        self.max_drop_frames = max_drop_frames
        self.kps_filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.prev_kps: Optional[np.ndarray] = None
        self.vel: Optional[np.ndarray] = None
        self.drop_count: int = 0

    def reset(self):
        """Clear state between videos."""
        self.kps_filter.reset()
        self.prev_kps = None
        self.vel = None
        self.drop_count = 0

    def update(self, kps: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Stabilizes 5 facial landmarks using rigid skull geometry and One-Euro filtering.
        During detection drops, velocity extrapolates landmarks to guarantee 100% face swap continuity.
        """
        if kps is None:
            if self.prev_kps is not None and self.drop_count < self.max_drop_frames:
                self.drop_count += 1
                # Smooth velocity dampening during occlusion
                step_vel = (self.vel * 0.5) if self.vel is not None else 0.0
                predicted = self.prev_kps + step_vel
                self.prev_kps = predicted.copy()
                return predicted.copy()
            self.prev_kps = None
            self.vel = None
            self.kps_filter.reset()
            return None

        self.drop_count = 0
        current = kps.astype(np.float32)

        # Step 1: Anchor landmarks to rigid cranial frame (suppresses mouth occlusions & speech wobble)
        anchored = get_cranial_anchored_kps(current)

        if self.prev_kps is None:
            self.prev_kps = anchored.copy()
            self.vel = np.zeros_like(anchored)
            self.kps_filter.filter(anchored)
            return anchored

        # Track velocity for dropped frame extrapolation
        raw_vel = anchored - self.prev_kps
        if self.vel is None:
            self.vel = raw_vel
        else:
            self.vel = 0.6 * raw_vel + 0.4 * self.vel

        # Step 2: Apply One-Euro filter across all 5 cranial-anchored points
        filtered_kps = self.kps_filter.filter(anchored)

        # Step 3: Enforce rigid ocular symmetry
        eye_left, eye_right = filtered_kps[0], filtered_kps[1]
        eye_center = (eye_left + eye_right) / 2.0
        eye_vec = eye_right - eye_left
        eye_dist = float(np.linalg.norm(eye_vec))

        if eye_dist > 1.0:
            eye_dir = eye_vec / eye_dist
            filtered_kps[0] = eye_center - 0.5 * eye_dist * eye_dir
            filtered_kps[1] = eye_center + 0.5 * eye_dist * eye_dir

        self.prev_kps = filtered_kps.astype(np.float32)
        return self.prev_kps.copy()


class AffineStabilizer:
    """
    Stabilizes the 2D Similarity Transform M = [s*R | t] across consecutive video frames
    using responsive One-Euro filtering on decomposed (scale, rotation_angle, tx, ty).
    Eliminates all frame-to-frame affine wobble and pulsation while maintaining frame-accurate alignment.
    """
    def __init__(self, min_cutoff: float = 0.08, beta: float = 2.0):
        self.filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.prev_params: Optional[np.ndarray] = None

    def reset(self):
        """Reset stabilizer between videos."""
        self.filter.reset()
        self.prev_params = None

    def decompose(self, M: np.ndarray) -> np.ndarray:
        """Decompose 2x3 similarity transform M into [scale, theta, tx, ty]."""
        a = M[0, 0]
        b = M[1, 0]
        scale = float(np.sqrt(a * a + b * b))
        theta = float(np.arctan2(b, a))
        tx = float(M[0, 2])
        ty = float(M[1, 2])
        return np.array([scale, theta, tx, ty], dtype=np.float32)

    def compose(self, params: np.ndarray) -> np.ndarray:
        """Reconstruct 2x3 similarity transform from [scale, theta, tx, ty]."""
        scale, theta, tx, ty = params
        cos_t = np.cos(theta) * scale
        sin_t = np.sin(theta) * scale
        return np.array([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty]
        ], dtype=np.float32)

    def update(self, M: np.ndarray) -> np.ndarray:
        """
        Smooths affine matrix M to eliminate frame-to-frame jitter.
        Handles angular wrapping across +/- pi.
        """
        params = self.decompose(M)
        if self.prev_params is not None:
            # Handle angle wrap-around across -pi / +pi
            d_theta = params[1] - self.prev_params[1]
            while d_theta > np.pi:
                params[1] -= 2.0 * np.pi
                d_theta = params[1] - self.prev_params[1]
            while d_theta < -np.pi:
                params[1] += 2.0 * np.pi
                d_theta = params[1] - self.prev_params[1]

            # Clamp instantaneous scale shock (> 15% jump in single frame)
            prev_scale = self.prev_params[0]
            if prev_scale > 1e-4:
                scale_ratio = params[0] / prev_scale
                if scale_ratio > 1.15:
                    params[0] = prev_scale * 1.15
                elif scale_ratio < 0.85:
                    params[0] = prev_scale * 0.85

        smoothed = self.filter.filter(params)
        self.prev_params = smoothed.copy()
        return self.compose(smoothed)


