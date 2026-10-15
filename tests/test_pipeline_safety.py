"""Regression tests for audited false authentication and unsafe fallback."""

import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from modules.antispoof.detector import AntiSpoofDetector
from modules.face_backend import FaceBackendUnavailable
from verify_with_antispoof import (
    FeatureExtractorUnavailable, ProtectionContext, cosine_similarity,
    create_protection_context, extract_face_features, main, run_pipeline, slerp_encrypt,
)

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "samples/genuine_personA_1.jpg"
B = ROOT / "samples/genuine_personB_1.jpg"


def test_missing_model_never_selects_statistics(tmp_path):
    with pytest.raises(FeatureExtractorUnavailable, match="checkpoint"):
        extract_face_features(A, tmp_path / "absent.pth")
    groups, vector = extract_face_features(A, demo=True)
    assert groups.shape == (49, 16) and vector.shape == (512,)


def test_demo_never_authenticates_a_different_identity():
    decision, report = run_pipeline(A, B, demo=True)
    assert report["processed"] and report["status"] == "DEMO_ONLY"
    assert decision is False and report["authenticated"] is False and report["is_match"] is None
    assert "not an identity" in report["score_warning"]


def test_configured_checkpoint_without_face_backend_cannot_authenticate(tmp_path):
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"not loaded before policy validation")
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=FaceBackendUnavailable("Missing aligned face backend")):
        with patch("verify_with_antispoof.extract_face_features") as extract:
            decision, report = run_pipeline(A, A, model_path=checkpoint)
    assert not decision and report["status"] == "MODEL_UNAVAILABLE"
    extract.assert_not_called()


@pytest.mark.parametrize("failed_image", [1, 2])
def test_rejected_input_aborts_all_extraction(failed_image):
    screen = ROOT / "samples/spoof_screen_replay.jpg"
    inputs = (screen, A) if failed_image == 1 else (A, screen)
    with patch("verify_with_antispoof.extract_face_features") as extract:
        decision, report = run_pipeline(*inputs, demo=True)
    assert not decision and report["failed_on"] == f"img{failed_image}"
    extract.assert_not_called()


def test_no_temp_input_writes_or_key_reports():
    with patch.object(Image.Image, "save", side_effect=AssertionError("raw input write")):
        decision, report = run_pipeline(A, A, demo=True)
    assert not decision and report["processed"]
    public = json.dumps(report, allow_nan=False)
    assert '"key"' not in public and '"keep_mask"' not in public


@pytest.mark.parametrize("alpha,drop", [(1, .5), (-.1, .5), (np.nan, .5), (.9, 1), (.9, -.1), (.9, np.inf)])
def test_transform_rejects_unsafe_boundaries(alpha, drop):
    with pytest.raises(ValueError):
        slerp_encrypt(np.ones((49, 16)), alpha, drop)


def test_context_is_fresh_explicit_and_does_not_reset_global_rng():
    state = np.random.get_state()
    first, second = create_protection_context((49, 16)), create_protection_context((49, 16))
    assert not np.array_equal(first.key, second.key)
    assert np.all(first.keep_mask.sum(axis=1) == 8)
    assert "array" not in repr(first)
    seeded = create_protection_context((49, 16), seed=42)
    assert np.array_equal(seeded.key, create_protection_context((49, 16), seed=42).key)
    current = np.random.get_state()
    assert state[0] == current[0] and np.array_equal(state[1], current[1]) and state[2:] == current[2:]


def test_slerp_stable_for_parallel_and_antipodal_groups():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(3, 16))
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    keep = np.ones_like(features, dtype=bool)
    for key in (unit, -unit):
        result = slerp_encrypt(features, drop_rate=0, context=ProtectionContext(key, keep))
        assert np.isfinite(result).all()
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(slerp_encrypt(features, alpha=0, drop_rate=0), unit, atol=1e-7)


def test_similarity_invalid_values_and_zero_norm():
    assert cosine_similarity(np.zeros(2), np.ones(2)) == 0
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(2), [np.nan, 1])


@pytest.mark.parametrize("magnitude", [1e308, 1e-308])
def test_extreme_finite_features_keep_normalization(magnitude):
    features = np.full((2, 16), magnitude)
    assert cosine_similarity(features, features) == pytest.approx(1)
    result = slerp_encrypt(features, alpha=0, drop_rate=0)
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1)


def test_complex_features_and_context_are_rejected():
    features = np.ones((2, 16))
    with pytest.raises(ValueError, match="real"):
        slerp_encrypt(features * 1j)
    with pytest.raises(ValueError, match="real"):
        cosine_similarity(features * 1j, features)
    context = ProtectionContext(np.full((2, 16), 1j / 4), np.ones((2, 16), dtype=bool))
    with pytest.raises(ValueError, match="real"):
        slerp_encrypt(features, drop_rate=0, context=context)


def test_weights_only_checkpoint_rejection_is_actionable():
    from verify_with_antispoof import _load_model

    class DummyModel:
        def to(self, device):
            return self

        def load_state_dict(self, state):
            raise AssertionError("Invalid weights must fail before model state loading")

    def reject_weights(*args, **kwargs):
        assert kwargs["weights_only"] is True
        raise pickle.UnpicklingError("unsafe or incompatible checkpoint")

    torch_stub = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                 device=lambda name: name, load=reject_weights)
    with patch.dict(sys.modules, {"torch": torch_stub,
                                 "modules.model": SimpleNamespace(SlerpFace=lambda **kwargs: DummyModel())}):
        with pytest.raises(FeatureExtractorUnavailable, match="Checkpoint"):
            _load_model("rejected-test-weights.pth", 1, 1)


def test_cli_demo_denial_and_errors_have_distinct_exit_status(tmp_path, capsys):
    report_file = tmp_path / "report.json"
    assert main(["--img1", str(A), "--demo", "--save_report", str(report_file)]) == 0
    assert json.loads(report_file.read_text())["authenticated"] is False
    assert main(["--img1", str(ROOT / "samples/spoof_print_attack.jpg"), "--demo"]) == 1
    assert main(["--img1", str(A), "--demo", "--alpha", "1"]) == 2
    assert main(["--img1", str(tmp_path / "missing.jpg")]) == 2
    capsys.readouterr()


def test_no_face_and_multiple_faces_fail_before_spectral_work():
    for boxes in ([], [(0, 0, 20, 20), (30, 30, 50, 50)]):
        with patch("modules.antispoof.detector._detect_faces", return_value=boxes):
            with patch("modules.antispoof.detector.analyze_frequency") as spectral:
                result = AntiSpoofDetector().evaluate(A)
        assert not result.is_live and not result.authentication_eligible
        spectral.assert_not_called()


def test_missing_face_detector_never_falls_back():
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=FaceBackendUnavailable("missing")):
        decision, report = run_pipeline(A, A)
    assert not decision and not report["processed"] and report["status"] == "MODEL_UNAVAILABLE"


def test_spectral_work_is_bounded_for_camera_images():
    image = Image.new("RGB", (1920, 1080), (170, 125, 95))
    with patch("modules.antispoof.detector.analyze_frequency", wraps=__import__("modules.antispoof.frequency_analysis", fromlist=["analyze_frequency"]).analyze_frequency) as spectral:
        result = AntiSpoofDetector(require_face=False).evaluate(image)
    assert spectral.call_args.args[0].shape == (112, 112, 3)
    assert result.authentication_eligible is False


@pytest.mark.parametrize("size", [64, 112, 224, 448])
def test_resized_attack_cannot_authorize_even_when_heuristic_passes(size):
    with Image.open(ROOT / "samples/spoof_print_attack.jpg") as image:
        probe = image.resize((size, size))
    decision, report = run_pipeline(A, probe, demo=True)
    assert not decision and report["authenticated"] is False
