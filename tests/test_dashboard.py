"""Streamlit regressions for empty inputs, truthful state, privacy and reruns.

Run directly with a Python environment containing Streamlit, or via pytest.
"""

import hashlib
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Streamlit is required for dashboard interaction tests")
class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Mock(return_value=(False, {
            "status": "DEMO_ONLY", "processed": True,
            "raw_similarity": 0.99, "encrypted_similarity": 0.99,
            "reason": "Uncalibrated demonstration; identity remains unverified.",
        }))
        self.detector = Mock()
        self.detector.evaluate.return_value = SimpleNamespace(
            is_live=True, liveness_score=0.8,
            diagnostics={"moire_score": 0.1, "high_freq_ratio": 0.2,
                         "lbp_entropy": 7.0, "chrominance_diversity": 9.0},
            to_dict=lambda: {"is_live": True, "liveness_score": 0.8,
                             "diagnostics": {"moire_score": 0.1, "high_freq_ratio": 0.2,
                                             "lbp_entropy": 7.0, "chrominance_diversity": 9.0}},
        )
        self.real_inspection = Mock(return_value={
            "face": {"confidence": 0.99, "quality": {"warnings": []}},
            "heuristic": {"is_live": False, "liveness_score": 0.2,
                          "advisory": True, "diagnostics": {}},
            "presentation": {"status": "UNAVAILABLE"},
        })
        self.pipeline_patch = patch("verify_with_antispoof.run_pipeline", self.pipeline)
        self.detector_patch = patch("modules.antispoof.detector.AntiSpoofDetector", return_value=self.detector)
        self.inspection_patch = patch("verify_with_antispoof.inspect_real_image", self.real_inspection)
        self.backend_patch = patch("modules.face_backend.face_backend_status", return_value={"ready": True})
        self.pipeline_patch.start()
        self.detector_patch.start()
        self.inspection_patch.start()
        self.backend_patch.start()
        self.addCleanup(self.pipeline_patch.stop)
        self.addCleanup(self.detector_patch.stop)
        self.addCleanup(self.inspection_patch.stop)
        self.addCleanup(self.backend_patch.stop)
        self.app = AppTest.from_file(str(PROJECT_ROOT / "app_demo.py"), default_timeout=20).run()
        self.assertEqual(len(self.app.exception), 0)

    def text(self):
        return "\n".join(str(element.value) for kind in
                         [self.app.markdown, self.app.info, self.app.warning, self.app.success,
                          self.app.caption, self.app.error] for element in kind)

    def test_missing_upload_and_camera_disable_operations(self):
        for first, second in [("Upload", "Camera"), ("Camera", "Upload"),
                              ("Preset", "Upload"), ("Camera", "Preset")]:
            with self.subTest(first=first, second=second):
                self.app.radio(key="in1").set_value(first)
                self.app.radio(key="in2").set_value(second).run()
                self.assertTrue(self.app.button(key="pair_run").disabled)
                self.assertEqual(len(self.app.exception), 0)
                self.pipeline.assert_not_called()
        self.app.radio(key="ininspect").set_value("Upload").run()
        self.assertTrue(self.app.button(key="inspect_run").disabled)
        self.app.radio(key="ininspect").set_value("Camera").run()
        self.assertTrue(self.app.button(key="inspect_run").disabled)
        self.assertNotIn("IDENTITY VERIFIED", self.text())

    def test_inspector_requires_explicit_operation_and_reports_actual_result(self):
        self.pipeline.assert_not_called()
        self.assertIn("No result for the current inputs and settings", self.text())
        self.assertEqual(len(self.app.success), 0)
        self.app.button(key="inspect_run").click().run()
        self.pipeline.assert_called_once()
        self.assertIsInstance(self.pipeline.call_args.args[0], np.ndarray)
        self.assertFalse(self.pipeline.call_args.kwargs["demo"])
        self.assertIn("Template demonstration completed", self.text())
        self.assertIn("Identity remains unverified", self.text())
        self.assertNotIn("SlerpFace Encrypted", self.text())
        self.assertEqual(len(self.app.exception), 0)

    def test_real_model_ready_and_default_threshold_are_visible(self):
        self.assertIn("Face backend ready", self.text())
        self.assertIn("No checkpoint entry required", self.text())
        self.assertAlmostEqual(self.app.slider[3].value, 0.363)
        self.assertEqual(self.app.slider[3].label, "Experimental cosine threshold")
        self.assertFalse(self.app.checkbox[0].value)

    def test_result_persists_and_invalidates_when_inputs_or_settings_change(self):
        self.app.button(key="pair_run").click().run()
        self.assertIn("Processing status: DEMO_ONLY", self.text())
        self.app.run()
        self.assertIn("Processing status: DEMO_ONLY", self.text())
        self.pipeline.assert_called_once()
        self.app.selectbox(key="tinspect").set_value(self.app.selectbox(key="tinspect").options[1]).run()
        self.assertIn("Processing status: DEMO_ONLY", self.text())
        self.app.slider[3].set_value(0.4).run()
        self.assertNotIn("Processing status: DEMO_ONLY", self.text())
        self.app.button(key="pair_run").click().run()
        self.assertEqual(self.pipeline.call_count, 2)
        self.app.selectbox(key="t1").set_value(self.app.selectbox(key="t1").options[1]).run()
        self.assertNotIn("Processing status: DEMO_ONLY", self.text())

    def test_real_face_results_show_raw_comparison_and_never_authenticate(self):
        for label in ("similar", "different", "uncertain"):
            with self.subTest(label=label):
                self.pipeline.return_value = (False, {
                    "status": "RESEARCH_COMPARISON", "processed": True,
                    "comparison_label": label, "match_threshold": 0.363,
                    "raw_similarity": 0.9, "encrypted_similarity": 0.99,
                    "img1_face": {"confidence": 0.99, "quality": {"warnings": []}},
                    "img1_presentation": {"status": "MODEL_ESTIMATE", "bona_fide_score": 0.8,
                                           "model_name": "Test presentation model"},
                })
                self.app.button(key="pair_run").click().run()
                self.assertIn(f"Experimental comparison: {label}", self.text())
                self.assertIn("Based on raw face cosine", self.text())
                self.assertIn("It does not determine the experimental comparison result", self.text())
                self.assertIn("Identity remains unverified", self.text())
                self.assertIn("Bona fide model score (uncalibrated)", self.text())
                self.assertEqual(len(self.app.success), 0)
                self.assertEqual(len(self.app.exception), 0)

    def test_camera_capture_has_no_duplicate_selected_input_preview(self):
        capture = io.BytesIO()
        Image.new("RGB", (48, 32), (120, 90, 70)).save(capture, format="PNG")
        capture.seek(0)
        self.app.radio(key="ininspect").set_value("Camera")
        original_image = __import__("streamlit").image
        with patch("streamlit.camera_input", return_value=capture), \
                patch("streamlit.image", wraps=original_image) as renderer:
            self.app.run()
        # Pair presets still have two previews; the camera widget owns its own.
        selected = [call for call in renderer.call_args_list if call.kwargs.get("caption") == "Selected input"]
        self.assertEqual(len(selected), 2)
        self.real_inspection.assert_called_once()
        self.assertFalse(self.app.button(key="inspect_run").disabled)
        self.assertIn("Heuristic assessment: inconclusive (advisory only)", self.text())
        self.assertNotIn("Heuristic gate: rejected", self.text())
        self.assertEqual(len(self.app.exception), 0)

    def test_pair_inputs_stay_in_memory_and_scores_never_authenticate(self):
        original_save = Image.Image.save
        paths = []

        def record_save(image, fp, *args, **kwargs):
            if isinstance(fp, (str, os.PathLike)):
                paths.append(os.fspath(fp))
            return original_save(image, fp, *args, **kwargs)

        existing = {path: hashlib.sha256(path.read_bytes()).digest()
                    for path in [PROJECT_ROOT / "temp_img1.jpg", PROJECT_ROOT / "temp_img2.jpg"]
                    if path.exists()}
        self.app.checkbox[0].set_value(True).run()
        with patch.object(Image.Image, "save", record_save):
            self.app.button(key="pair_run").click().run()
        self.assertEqual(paths, [])
        self.pipeline.assert_called_once()
        self.assertTrue(all(isinstance(image, np.ndarray) for image in self.pipeline.call_args.args))
        self.assertTrue(self.pipeline.call_args.kwargs["demo"])
        for path, digest in existing.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), digest)
        self.assertIn("DEMO_ONLY", self.text())
        self.assertIn("Identity remains unverified", self.text())
        self.assertEqual(len(self.app.success), 0)
        self.assertNotIn("IDENTITY VERIFIED", self.text())
        self.assertEqual(len(self.app.exception), 0)

    def test_valid_upload_uses_decoded_rgb_memory(self):
        upload = io.BytesIO()
        Image.new("RGBA", (48, 32), (120, 90, 70, 255)).save(upload, format="PNG")
        upload.seek(0)
        self.app.radio(key="in1").set_value("Upload")
        with patch("streamlit.file_uploader", return_value=upload):
            self.app.run()
            self.assertFalse(self.app.button(key="pair_run").disabled)
            self.app.button(key="pair_run").click().run()
        self.pipeline.assert_called_once()
        self.assertEqual(self.pipeline.call_args.args[0].shape, (32, 48, 3))
        self.assertEqual(self.pipeline.call_args.args[0].dtype, np.uint8)
        self.assertEqual(len(self.app.exception), 0)

    def test_stable_presets_local_rng_and_cached_inspection(self):
        original = self.app.session_state["sample_inspect"][1].copy()
        calls = self.detector.evaluate.call_count
        random_state = np.random.get_state()
        self.app.slider[1].set_value(0.8).run()
        np.testing.assert_array_equal(original, self.app.session_state["sample_inspect"][1])
        self.assertEqual(self.detector.evaluate.call_count, calls)
        after = np.random.get_state()
        self.assertEqual(random_state[0], after[0])
        np.testing.assert_array_equal(random_state[1], after[1])
        self.assertEqual(random_state[2:], after[2:])
        options = self.app.selectbox(key="tinspect").options
        self.app.selectbox(key="tinspect").set_value(options[1]).run()
        self.app.selectbox(key="tinspect").set_value(options[0]).run()
        np.testing.assert_array_equal(original, self.app.session_state["sample_inspect"][1])
        self.app.slider[0].set_value(0.65).run()
        self.assertEqual(self.detector.evaluate.call_count, calls + 3)

    def test_corrupt_upload_and_processing_errors_are_bounded(self):
        self.app.radio(key="in1").set_value("Upload")
        with patch("streamlit.file_uploader", return_value=io.BytesIO(b"invalid image")):
            self.app.run()
        self.assertTrue(self.app.button(key="pair_run").disabled)
        self.assertIn("Unable to read this image", self.text())
        self.assertEqual(len(self.app.exception), 0)
        self.pipeline.assert_not_called()
        oversized = SimpleNamespace(size=20 * 1024 * 1024 + 1)
        with patch("streamlit.file_uploader", return_value=oversized):
            self.app.run()
        self.assertTrue(self.app.button(key="pair_run").disabled)
        self.assertEqual(len(self.app.exception), 0)
        self.app.radio(key="in1").set_value("Preset").run()
        self.pipeline.side_effect = RuntimeError("sensitive detail that must not enter the page")
        self.app.button(key="pair_run").click().run()
        self.assertIn("Processing unavailable", self.text())
        self.assertNotIn("sensitive detail", self.text())
        self.assertEqual(len(self.app.exception), 0)

    def test_missing_model_is_an_unverified_result(self):
        self.pipeline.return_value = (False, {
            "status": "MODEL_UNAVAILABLE", "processed": False,
            "reason": "Configure the trained model before extraction.",
        })
        self.app.button(key="pair_run").click().run()
        self.assertIn("MODEL_UNAVAILABLE", self.text())
        self.assertIn("No template demonstration completed", self.text())
        self.assertEqual(len(self.app.success), 0)
        self.assertEqual(len(self.app.exception), 0)

    def test_benchmark_labels_constructed_results_without_fabricated_comparison(self):
        dataset = [{"is_spoof": False, "img": np.zeros((16, 16, 3), dtype=np.uint8)},
                   {"is_spoof": True, "img": np.ones((16, 16, 3), dtype=np.uint8)}]
        with patch("evaluate_antispoof.generate_synthetic_benchmark_dataset", return_value=dataset):
            self.app.button(key="benchmark_run").click().run()
        self.assertIn("Evaluated 2 generated patterns", self.text())
        self.assertIn("do not measure physical attacks", self.text())
        self.assertEqual(len(self.app.table), 0)
        self.assertEqual(len(self.app.exception), 0)
        self.app.run()
        self.assertIn("Evaluated 2 generated patterns", self.text())
        self.app.slider[0].set_value(0.65).run()
        self.assertNotIn("Evaluated 2 generated patterns", self.text())


@unittest.skipIf(AppTest is None, "Streamlit is required for dashboard interaction tests")
class DashboardCoreIntegrationTests(unittest.TestCase):
    def test_unavailable_presence_backend_does_not_crash_or_extract(self):
        from modules.face_backend import FaceBackendUnavailable

        app = AppTest.from_file(str(PROJECT_ROOT / "app_demo.py"), default_timeout=30).run()
        with patch("verify_with_antispoof.extract_pretrained_face", side_effect=FaceBackendUnavailable("Face backend unavailable.")), \
                patch("verify_with_antispoof.extract_face_features") as extractor:
            app.button(key="pair_run").click().run()
        extractor.assert_not_called()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 0)
        self.assertTrue(any("UNAVAILABLE" in element.value for element in app.markdown))
        self.assertTrue(any("Identity remains unverified" in element.value for element in app.warning))

    def test_real_default_and_opt_in_demo_remain_unverified(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app_demo.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.checkbox[0].value)
        app.button(key="pair_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 0)
        self.assertTrue(any("Identity remains unverified" in element.value for element in app.warning))
        app.checkbox[0].set_value(True).run()
        app.button(key="pair_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.success), 0)
        self.assertTrue(any("DEMO_ONLY" in element.value for element in app.markdown))
        self.assertTrue(any("Identity remains unverified" in element.value for element in app.info))
        self.assertTrue(any(metric.label == "Transformed descriptor cosine (diagnostic)" for metric in app.metric))
        app.button(key="inspect_run").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Template demonstration completed" in element.value for element in app.info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
