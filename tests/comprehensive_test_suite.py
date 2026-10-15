"""Direct-script and pytest-compatible research safety regression suite."""
import ast
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.antispoof.detector import AntiSpoofDetector
from modules.antispoof.frequency_analysis import compute_fft_spectrum, compute_radial_profile
from modules.antispoof.texture_analysis import compute_lbp, compute_lbp_entropy
from verify_with_antispoof import (
    FeatureExtractorUnavailable, create_protection_context, extract_face_features,
    run_pipeline, slerp_encrypt,
)


class ResearchSafetyTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.a = ROOT / "samples/genuine_personA_1.jpg"
        self.b = ROOT / "samples/genuine_personB_1.jpg"
        self.detector = AntiSpoofDetector(require_face=False)

    def test_fft(self):
        spectrum = compute_fft_spectrum(self.rng.normal(size=(112, 112)))
        self.assertEqual(spectrum.shape, (112, 112))
        self.assertTrue(np.isfinite(spectrum).all())

    def test_radial_mean(self):
        np.testing.assert_allclose(compute_radial_profile(np.ones((112, 112)), 30), 1)

    def test_lbp(self):
        lbp = compute_lbp(self.rng.normal(size=(112, 112)))
        self.assertEqual(lbp.shape, (110, 110))
        self.assertGreater(compute_lbp_entropy(lbp), 0)

    def test_per_group_dropout(self):
        result = slerp_encrypt(self.rng.normal(size=(49, 16)))
        self.assertTrue(np.all((result == 0).sum(axis=1) == 8))

    def test_model_missing_no_fallback(self):
        with self.assertRaises(FeatureExtractorUnavailable):
            extract_face_features(self.a, ROOT / "nonexistent-checkpoint.pth")

    def test_positive_demo_not_identity(self):
        decision, report = run_pipeline(self.a, self.a, demo=True)
        self.assertFalse(decision)
        self.assertTrue(report["processed"])
        self.assertIsNone(report["is_match"])

    def test_different_identity_not_authenticated(self):
        decision, report = run_pipeline(self.a, self.b, demo=True)
        self.assertFalse(decision)
        self.assertFalse(report["authenticated"])

    def test_spoof_aborts_before_extraction(self):
        with patch("verify_with_antispoof.extract_face_features") as extract:
            decision, report = run_pipeline(self.a, ROOT / "samples/spoof_screen_replay.jpg", demo=True)
        self.assertFalse(decision)
        self.assertEqual(report["tier1_status"], "REJECTED")
        extract.assert_not_called()

    def test_single_image_is_not_enrollment(self):
        decision, report = run_pipeline(self.a, demo=True)
        self.assertFalse(decision)
        self.assertEqual(report["status"], "DEMO_ONLY")
        self.assertIn("no identity verified or credential enrolled", report["reason"])

    def test_public_json_no_secret_material(self):
        _, report = run_pipeline(self.a, demo=True)
        payload = json.dumps(report, allow_nan=False)
        self.assertNotIn('"key"', payload)
        self.assertNotIn('"keep_mask"', payload)

    def test_black_rejected(self):
        self.assertFalse(self.detector.evaluate(np.zeros((112, 112, 3), dtype=np.uint8)).is_live)

    def test_white_rejected(self):
        self.assertFalse(self.detector.evaluate(np.full((112, 112, 3), 255, dtype=np.uint8)).is_live)

    def test_large_image_bounded(self):
        result = self.detector.evaluate(np.full((1080, 1920, 3), 100, dtype=np.uint8))
        self.assertEqual(result.diagnostics["analysis_width"], 112)

    def test_invalid_small_image(self):
        with self.assertRaises(ValueError):
            self.detector.evaluate(np.zeros((1, 1, 3), dtype=np.uint8))

    def test_dashboard_syntax_only(self):
        ast.parse((ROOT / "app_demo.py").read_text(encoding="utf-8"))

    def test_fresh_contexts(self):
        a, b = create_protection_context((49, 16)), create_protection_context((49, 16))
        self.assertFalse(np.array_equal(a.key, b.key))


def run_all_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ResearchSafetyTests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


if __name__ == "__main__":
    raise SystemExit(0 if run_all_tests() else 1)
