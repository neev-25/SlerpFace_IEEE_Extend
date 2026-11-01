"""
tests/test_webcam_pipeline.py
==============================
Tests for Physical Webcam Walkthrough, Calibration & Edge Scripts.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.webcam_live_verify import run_live_webcam
from scripts.calibrate_thresholds import calibrate_thresholds
from scripts.test_physical_spoofs import evaluate_physical_spoofs
from scripts.benchmark_edge import benchmark_pipeline


def test_headless_webcam_walkthrough():
    """Confirms live webcam engine runs headless test loop cleanly."""
    res = run_live_webcam(device_index=-1, headless=True, max_frames=3)
    assert res["status"] == "COMPLETED"
    assert res["frames_processed"] == 3
    assert res["authentication_enabled"] is False


def test_headless_webcam_never_opens_a_camera(monkeypatch):
    from modules.face_backend import FaceInputError

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline validation attempted physical camera access")

    def no_face(*args, **kwargs):
        raise FaceInputError("Offline fixture", "NO_FACE")

    monkeypatch.setattr("scripts.webcam_live_verify.cv2.VideoCapture", forbidden)
    monkeypatch.setattr("scripts.webcam_live_verify.extract_pretrained_face", no_face)
    result = run_live_webcam(headless=True, max_frames=1)
    assert result["status"] == "COMPLETED" and result["frames_processed"] == 1
    assert result["authentication_enabled"] is False


def test_webcam_comparison_is_experimental_and_heuristics_are_advisory(monkeypatch):
    import cv2
    import numpy as np

    class Camera:
        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((200, 200, 3), dtype=np.uint8)

        def release(self):
            pass

    face = SimpleNamespace(
        groups=np.ones((8, 16), dtype=np.float32),
        vector=np.ones(128, dtype=np.float32),
        face_box=(10, 10, 100, 100),
        landmarks=np.ones((5, 2), dtype=np.float32),
        aligned_rgb=np.zeros((112, 112, 3), dtype=np.uint8),
    )
    heuristic = SimpleNamespace(
        is_live=False, liveness_score=0.1, badge="UNVERIFIED", attack_type="UNVERIFIED",
    )
    keys = iter((ord("r"), ord("v")))

    monkeypatch.setattr(cv2, "VideoCapture", lambda _device: Camera())
    monkeypatch.setattr(cv2, "cvtColor", lambda frame, _code: frame)
    monkeypatch.setattr(cv2, "rectangle", lambda *args: None)
    monkeypatch.setattr(cv2, "circle", lambda *args: None)
    monkeypatch.setattr(cv2, "putText", lambda *args: None)
    monkeypatch.setattr(cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(cv2, "waitKey", lambda _delay: next(keys))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr("scripts.webcam_live_verify.extract_pretrained_face", lambda _image: face)
    monkeypatch.setattr(
        "scripts.webcam_live_verify.AntiSpoofDetector",
        lambda **_kwargs: SimpleNamespace(evaluate=lambda _image: heuristic),
    )
    monkeypatch.setattr(
        "scripts.webcam_live_verify.estimate_head_pose",
        lambda _landmarks: SimpleNamespace(yaw=0, pitch=0, category="frontal"),
    )

    result = run_live_webcam(headless=False, max_frames=2)
    assert result["authentication_enabled"] is False
    assert result["history"][0]["comparison_label"] == "similar"
    assert result["history"][0]["heuristic_score"] == 0.1
    assert result["history"][0]["authenticated"] is False
    assert "matched" not in result["history"][0]


def test_similarity_calibration_rejects_synthetic_manifest(tmp_path):
    """Calibration cannot synthesize augmented portraits as independent captures."""
    (tmp_path / "sample.png").write_bytes(b"synthetic test material")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "data_kind": "synthetic",
        "source": "unit-test fixture",
        "presentations": [{
            "image": "sample.png", "subject": "synthetic-subject",
            "kind": "bona_fide", "capture_kind": "synthetic",
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="real identity pairs"):
        calibrate_thresholds(manifest)


def test_physical_spoof_benchmark_rejects_synthetic_manifest(tmp_path):
    """Generated sample illustrations cannot count as physical attack trials."""
    (tmp_path / "sample.png").write_bytes(b"synthetic test material")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "data_kind": "synthetic",
        "source": "unit-test fixture",
        "presentations": [{
            "image": "sample.png", "subject": "synthetic-subject",
            "kind": "bona_fide", "capture_kind": "synthetic",
        }],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="real-data presentation manifest"):
        evaluate_physical_spoofs(manifest)


def test_edge_latency_benchmark():
    """Reports local timing observations without asserting deployment performance."""
    report = benchmark_pipeline(num_iterations=5)
    assert "latencies_ms" in report
    assert report["latencies_ms"]["tier1_pad_combined"] > 0.0
    assert report["latencies_ms"]["combined_measured_components"] > 0.0
    assert report["reciprocal_component_rate_per_second"] > 0.0
    assert report["authentication_enabled"] is False
    assert "guarantee" not in report
