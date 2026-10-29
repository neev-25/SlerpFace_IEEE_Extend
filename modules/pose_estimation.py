"""
modules/pose_estimation.py
==========================
Facial Pose Estimation & Extreme Yaw/Pitch Tolerance.

Calculates head pose (yaw, pitch, roll) from 5 facial landmarks (eyes, nose, mouth corners)
extracted by YuNet, providing adaptive matching tolerance and recapture guidance.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Union
import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    yaw: float          # Negative = left turn, Positive = right turn (degrees)
    pitch: float        # Negative = looking down, Positive = looking up (degrees)
    roll: float         # In-plane tilt (degrees)
    category: str       # "OPTIMAL", "COMPENSATED", "EXTREME"
    is_acceptable: bool # True if OPTIMAL or COMPENSATED
    guidance: str       # Actionable feedback for user / UI
    adaptive_margin: float  # Cosine threshold adjustment for pose compensation

    def to_dict(self) -> Dict[str, Union[float, str, bool]]:
        return asdict(self)


def estimate_head_pose(landmarks: Union[np.ndarray, list]) -> PoseEstimate:
    """
    Estimates 3D head orientation angles from 5 YuNet 2D landmarks:
      0: right eye
      1: left eye
      2: nose tip
      3: right mouth corner
      4: left mouth corner
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (5, 2) or not np.isfinite(pts).all():
        raise ValueError("Landmarks must be a finite array of shape (5, 2).")

    re, le = pts[0], pts[1]
    nose = pts[2]
    rm, lm = pts[3], pts[4]

    # 1. Roll: In-plane eye angle
    dx = le[0] - re[0]
    dy = le[1] - re[1]
    roll = float(np.degrees(np.arctan2(dy, dx)))

    # 2. Yaw: Out-of-plane horizontal rotation
    d_eyes = np.hypot(dx, dy)
    if d_eyes < 1e-4:
        raise ValueError("Invalid landmark geometry: eyes are coincident.")

    d_right = float(np.hypot(nose[0] - re[0], nose[1] - re[1]))
    d_left = float(np.hypot(nose[0] - le[0], nose[1] - le[1]))
    asymmetry = (d_right - d_left) / (d_right + d_left + 1e-6)
    # Calibrate ratio to angular degrees (-90 to +90)
    yaw = float(np.degrees(np.arcsin(np.clip(asymmetry * 1.55, -1.0, 1.0))))

    # 3. Pitch: Vertical tilt
    eye_mid = (re + le) / 2.0
    mouth_mid = (rm + lm) / 2.0
    face_height = float(np.hypot(mouth_mid[0] - eye_mid[0], mouth_mid[1] - eye_mid[1]))
    if face_height < 1e-4:
        raise ValueError("Invalid landmark geometry: eyes and mouth are coincident.")

    nose_vert_offset = float(nose[1] - eye_mid[1])
    ratio = nose_vert_offset / face_height
    # In standard frontal face, nose vertical offset is ~0.42 of eye-to-mouth height
    pitch_diff = ratio - 0.42
    pitch = float(np.clip(pitch_diff * 110.0, -90.0, 90.0))

    # 4. Categorization & Guidance
    abs_yaw = abs(yaw)
    abs_pitch = abs(pitch)

    if abs_yaw <= 20.0 and abs_pitch <= 15.0:
        category = "OPTIMAL"
        is_acceptable = True
        guidance = "Pose optimal (frontal view)."
        adaptive_margin = 0.0
    elif abs_yaw <= 35.0 and abs_pitch <= 25.0:
        category = "COMPENSATED"
        is_acceptable = True
        guidance = f"Moderate pose variation (yaw: {yaw:.1f}°, pitch: {pitch:.1f}°). Compares with adaptive tolerance."
        # Slight threshold adjustment to compensate for facial perspective foreshortening
        adaptive_margin = -0.025
    else:
        category = "EXTREME"
        is_acceptable = False
        guidance = f"Extreme facial pose detected (yaw: {yaw:.1f}°, pitch: {pitch:.1f}°). Please face the camera directly."
        adaptive_margin = -0.05

    return PoseEstimate(
        yaw=round(yaw, 2),
        pitch=round(pitch, 2),
        roll=round(roll, 2),
        category=category,
        is_acceptable=is_acceptable,
        guidance=guidance,
        adaptive_margin=round(adaptive_margin, 4),
    )
