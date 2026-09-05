"""
tests/test_antispoof.py
========================
Unit and regression tests for the Anti-Spoofing and SlerpFace subsystem.
"""

import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.antispoof.frequency_analysis import (
    compute_fft_spectrum,
    compute_radial_profile,
    detect_moire_patterns,
    analyze_frequency,
)
from modules.antispoof.texture_analysis import (
    compute_lbp,
    compute_lbp_entropy,
    analyze_color_diversity,
    analyze_texture,
)
from modules.antispoof.detector import AntiSpoofDetector
from verify_with_antispoof import slerp_encrypt, cosine_similarity


def test_fft_spectrum():
    img = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
    spectrum = compute_fft_spectrum(img)
    assert spectrum.shape == (112, 112)
    assert np.all(np.isfinite(spectrum))
    print("✅ test_fft_spectrum passed")


def test_radial_profile():
    spectrum = np.ones((112, 112), dtype=np.float64)
    profile = compute_radial_profile(spectrum, num_bins=30)
    assert len(profile) == 30
    assert np.all(profile >= 0)
    print("✅ test_radial_profile passed")


def test_texture_analysis():
    img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    res = analyze_texture(img)
    assert "lbp_entropy" in res
    assert "texture_liveness_score" in res
    assert 0.0 <= res["texture_liveness_score"] <= 1.0
    print("✅ test_texture_analysis passed")


def test_detector_classification():
    detector = AntiSpoofDetector(threshold=0.60)
    
    # 1. Genuine face
    gen_path = os.path.join(PROJECT_ROOT, "samples", "genuine_personA_1.jpg")
    res_gen = detector.evaluate(gen_path)
    assert res_gen.liveness_score >= 0.60, f"Expected >=0.60, got {res_gen.liveness_score}"
    assert res_gen.is_live is True
    print(f"✅ Genuine detection passed (Score: {res_gen.liveness_score:.2f})")

    # 2. Screen replay attack
    screen_path = os.path.join(PROJECT_ROOT, "samples", "spoof_screen_replay.jpg")
    res_scr = detector.evaluate(screen_path)
    assert res_scr.liveness_score < 0.60, f"Expected <0.60, got {res_scr.liveness_score}"
    assert res_scr.is_live is False
    print(f"✅ Screen replay spoof intercepted (Score: {res_scr.liveness_score:.2f}, Verdict: {res_scr.verdict})")

    # 3. Print attack
    print_path = os.path.join(PROJECT_ROOT, "samples", "spoof_print_attack.jpg")
    res_prn = detector.evaluate(print_path)
    assert res_prn.liveness_score < 0.60, f"Expected <0.60, got {res_prn.liveness_score}"
    assert res_prn.is_live is False
    print(f"✅ Print attack spoof intercepted (Score: {res_prn.liveness_score:.2f}, Verdict: {res_prn.verdict})")


def test_slerp_encryption():
    features = np.random.randn(49, 16)
    encrypted = slerp_encrypt(features, alpha=0.9, drop_rate=0.5, seed=123)
    assert encrypted.shape == (49, 16)
    
    # Check that approx 50% are zeroed
    zero_count = np.sum(encrypted == 0.0)
    total = encrypted.size
    zero_ratio = zero_count / total
    assert abs(zero_ratio - 0.5) < 0.05
    print(f"✅ Slerp encryption verified (Zeroed ratio: {zero_ratio:.2%})")


if __name__ == "__main__":
    print("Running subsystem tests...")
    test_fft_spectrum()
    test_radial_profile()
    test_texture_analysis()
    test_detector_classification()
    test_slerp_encryption()
    print("\n🎉 ALL UNIT TESTS PASSED SUCCESSFULLY!")
