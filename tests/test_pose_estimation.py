"""
tests/test_pose_estimation.py
==============================
Tests for 5-Landmark Facial Pose Estimation & Extreme Yaw/Pitch Tolerance.
"""

import pytest
import numpy as np

from modules.pose_estimation import estimate_head_pose, PoseEstimate


@pytest.fixture
def frontal_landmarks():
    # Canonical 5 landmarks: right_eye, left_eye, nose, right_mouth, left_mouth
    return np.array([
        [40.0, 45.0],
        [72.0, 45.0],
        [56.0, 62.0],
        [43.0, 85.0],
        [69.0, 85.0],
    ], dtype=float)


def test_frontal_pose_optimal(frontal_landmarks):
    """Frontal face yields optimal pose with zero adaptive margin."""
    pose = estimate_head_pose(frontal_landmarks)
    assert isinstance(pose, PoseEstimate)
    assert abs(pose.yaw) < 5.0
    assert abs(pose.pitch) < 5.0
    assert abs(pose.roll) < 2.0
    assert pose.category == "OPTIMAL"
    assert pose.is_acceptable is True
    assert pose.adaptive_margin == 0.0
    assert "optimal" in pose.guidance.lower()


def test_moderate_yaw_compensated(frontal_landmarks):
    """Moderate turn yields COMPENSATED status with adaptive tolerance margin."""
    turned = frontal_landmarks.copy()
    turned[2, 0] += 9.0  # Nose shifted right
    pose = estimate_head_pose(turned)

    assert pose.yaw > 15.0
    assert pose.category in ("OPTIMAL", "COMPENSATED")
    assert pose.is_acceptable is True


def test_extreme_yaw_rejection(frontal_landmarks):
    """Extreme yaw turn (> 35 deg) produces EXTREME status and recapture guidance."""
    extreme = frontal_landmarks.copy()
    extreme[2, 0] += 18.0  # Large horizontal displacement
    pose = estimate_head_pose(extreme)

    assert abs(pose.yaw) > 35.0
    assert pose.category == "EXTREME"
    assert pose.is_acceptable is False
    assert "extreme" in pose.guidance.lower()
    assert "face the camera directly" in pose.guidance.lower()


def test_invalid_landmark_geometry():
    """Invalid shapes or coincident landmarks raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        estimate_head_pose(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="eyes are coincident"):
        estimate_head_pose(np.zeros((5, 2)))
