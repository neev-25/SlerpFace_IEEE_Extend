"""Subsystem checks: synthetic samples are not live human faces."""
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.antispoof.frequency_analysis import compute_fft_spectrum, compute_radial_profile
from modules.antispoof.texture_analysis import analyze_texture
from modules.antispoof.detector import AntiSpoofDetector
from verify_with_antispoof import slerp_encrypt


def test_fft_spectrum():
    image = np.random.default_rng(1).integers(0, 255, (112, 112), dtype=np.uint8)
    spectrum = compute_fft_spectrum(image)
    assert spectrum.shape == (112, 112) and np.isfinite(spectrum).all()


def test_radial_profile():
    np.testing.assert_allclose(compute_radial_profile(np.ones((112, 112)), 30), 1)


def test_texture_analysis():
    image = np.random.default_rng(1).integers(50, 200, (112, 112, 3), dtype=np.uint8)
    assert 0 <= analyze_texture(image)["texture_liveness_score"] <= 1


def test_detector_classification():
    detector = AntiSpoofDetector(require_face=False)
    genuine = detector.evaluate(PROJECT_ROOT / "samples/genuine_personA_1.jpg")
    assert genuine.is_live and not genuine.authentication_eligible
    assert "unverified" in genuine.summary_text
    for name in ("spoof_screen_replay.jpg", "spoof_print_attack.jpg"):
        assert not detector.evaluate(PROJECT_ROOT / "samples" / name).is_live


def test_slerp_encryption():
    result = slerp_encrypt(np.random.default_rng(1).normal(size=(49, 16)), seed=123)
    assert result.shape == (49, 16) and np.all((result == 0).sum(axis=1) == 8)


if __name__ == "__main__":
    for check in (test_fft_spectrum, test_radial_profile, test_texture_analysis,
                  test_detector_classification, test_slerp_encryption):
        check()
        print(f"PASS {check.__name__}")
