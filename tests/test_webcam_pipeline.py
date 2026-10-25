"""
tests/test_webcam_pipeline.py
==============================
Tests for Physical Webcam Walkthrough, Calibration & Edge Scripts.
"""

from pathlib import Path

from scripts.webcam_live_verify import run_live_webcam
from scripts.calibrate_thresholds import compute_pairwise_similarities, generate_calibration_dataset
from scripts.test_physical_spoofs import evaluate_physical_spoofs
from scripts.benchmark_edge import benchmark_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_headless_webcam_walkthrough():
    """Confirms live webcam engine runs headless test loop cleanly."""
    res = run_live_webcam(device_index=-1, headless=True, max_frames=3)
    assert res["status"] == "COMPLETED"
    assert res["frames_processed"] == 3


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


def test_similarity_calibration_dataset():
    """Confirms similarity calibration computes separation and threshold."""
    gen_pairs, imp_pairs = generate_calibration_dataset()
    report = compute_pairwise_similarities(gen_pairs, imp_pairs)

    assert report["num_genuine_comparisons"] > 0
    assert report["num_impostor_comparisons"] > 0
    assert report["genuine_distribution"]["mean"] > report["impostor_distribution"]["mean"]
    assert report["d_prime_separation"] > 2.0
    assert 0.0 <= report["calibrated_threshold"] <= 1.0


def test_physical_spoof_benchmark():
    """Confirms physical spoof benchmark reports ISO/IEC 30107-3 metrics."""
    report = evaluate_physical_spoofs(liveness_thresh=0.60)
    assert report["total_presentations_evaluated"] == 5
    assert report["attack_interception_rate_percent"] == 100.0
    assert report["zero_leakage_protocol_enforced"] is True
    assert report["acer_percent"] == 0.0


def test_edge_latency_benchmark():
    """Confirms edge deployment benchmark validates Tier-1 performance."""
    report = benchmark_pipeline(num_iterations=5)
    assert "latencies_ms" in report
    assert report["latencies_ms"]["tier1_pad_combined"] < 25.0
    assert report["throughput_fps"] > 0.0
