"""Research comparisons must execute trained features without granting access."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest

from modules.face_backend import FaceBackendUnavailable, FaceInputError
from verify_with_antispoof import inspect_real_image, main, run_pipeline


def feature_pair(cosine):
    rng = np.random.default_rng(181)
    first = rng.normal(size=128)
    first /= np.linalg.norm(first)
    orthogonal = rng.normal(size=128)
    orthogonal -= np.dot(first, orthogonal) * first
    orthogonal /= np.linalg.norm(orthogonal)
    second = cosine * first + np.sqrt(1 - cosine ** 2) * orthogonal

    def record(vector):
        return SimpleNamespace(vector=vector, groups=vector.reshape(-1, 16),
                               face_box=(30, 20, 130, 140), landmarks=np.ones((5, 2)) * 60,
                               detection_score=.99, model_name="Test pretrained backend",
                               quality={"warnings": []},
                               aligned_rgb=np.full((112, 112, 3), 90, dtype=np.uint8))

    return record(first), record(second)


@pytest.fixture
def advisory_pad(monkeypatch):
    monkeypatch.setattr("verify_with_antispoof._presentation_estimate", lambda *args: {
        "status": "UNAVAILABLE", "bona_fide_score": None, "reason": "Isolated unit test"})


@pytest.mark.parametrize("score,label,experimental", [(.8, "similar", True), (.1, "different", False), (.36, "uncertain", None)])
def test_real_comparison_labels_use_trained_cosine_without_authentication(advisory_pad, score, label, experimental):
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=feature_pair(score)):
        decision, report = run_pipeline(Image.new("RGB", (640, 480)), Image.new("RGB", (800, 600)))
    assert report["processed"] and report["status"] == "RESEARCH_COMPARISON"
    assert report["raw_similarity"] == pytest.approx(score)
    assert report["comparison_label"] == label and report["experimental_match"] is experimental
    assert not decision and report["authenticated"] is False and report["is_match"] is None
    assert report["img1_res"]["advisory"] is True
    assert report["img1_presentation"]["status"] == "UNAVAILABLE"
    assert "SFace embedding reshaped" in report["transform_source"]
    public = json.dumps(report, allow_nan=False)
    assert '"vector"' not in public and '"key"' not in public and '"keep_mask"' not in public


def test_texture_rejection_no_longer_prevents_real_feature_comparison(advisory_pad):
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=feature_pair(.7)):
        _, report = run_pipeline(np.zeros((480, 640, 3), dtype=np.uint8), np.zeros((480, 640, 3), dtype=np.uint8))
    assert report["img1_res"]["heuristic_passed"] is False
    assert report["img1_res"]["verdict"] == "INCONCLUSIVE"
    assert report["processed"] and report["comparison_label"] == "similar"


def test_single_real_image_completes_transient_trained_operation(advisory_pad):
    with patch("verify_with_antispoof.extract_pretrained_face", return_value=feature_pair(.7)[0]):
        _, report = run_pipeline(Image.new("RGB", (640, 480)))
    assert report["status"] == "RESEARCH_TEMPLATE" and report["processed"]
    assert report["img1_template_shape"] == [8, 16]
    assert "raw_similarity" not in report and report["authenticated"] is False


@pytest.mark.parametrize("code", ["NO_FACE", "MULTIPLE_FACES", "LOW_QUALITY"])
def test_face_failures_do_not_extract_statistical_fallback(advisory_pad, code):
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=FaceInputError("Recapture one clear face", code)):
        with patch("verify_with_antispoof.extract_face_features") as fallback:
            _, report = run_pipeline(Image.new("RGB", (640, 480)))
    assert report["status"] == code and not report["processed"]
    fallback.assert_not_called()


def test_missing_default_models_is_actionable_not_synthetic_fallback():
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=FaceBackendUnavailable("Run scripts/provision_face_models.py")):
        _, report = run_pipeline(Image.new("RGB", (640, 480)))
    assert report["status"] == "MODEL_UNAVAILABLE" and "provision_face_models" in report["reason"]
    assert report["authenticated"] is False


def test_custom_ir50_receives_aligned_crop_and_runs(advisory_pad, tmp_path):
    path = tmp_path / "compatible-test-checkpoint.pth"
    path.write_bytes(b"Not loaded: controlled checkpoint-integration seam")
    faces = feature_pair(.7)
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=faces):
        with patch("verify_with_antispoof.extract_face_features", side_effect=[(f.groups, f.vector) for f in faces]) as extract:
            _, report = run_pipeline(Image.new("RGB", (640, 480)), Image.new("RGB", (640, 480)), model_path=path)
    assert report["processed"] and report["backend"] == "slerpface_ir50"
    assert extract.call_count == 2
    np.testing.assert_array_equal(extract.call_args_list[0].args[0], faces[0].aligned_rgb)
    assert extract.call_args_list[0].kwargs["model_path"] == path


def test_missing_explicit_ir50_is_not_replaced_with_default_model(advisory_pad, tmp_path):
    with patch("verify_with_antispoof.extract_pretrained_face") as extract:
        _, report = run_pipeline(Image.new("RGB", (640, 480)), model_path=tmp_path / "missing.pth")
    extract.assert_not_called()
    assert report["status"] == "MODEL_UNAVAILABLE"


def test_inspector_uses_same_alignment_and_returns_no_embedding(advisory_pad):
    with patch("verify_with_antispoof.extract_pretrained_face", return_value=feature_pair(.7)[0]):
        result = inspect_real_image(Image.new("RGB", (640, 480)))
    assert result["face"]["alignment"] == "five_landmark_112x112"
    assert result["heuristic"]["advisory"] and result["presentation"]["status"] == "UNAVAILABLE"
    assert "vector" not in json.dumps(result)


def test_cli_completed_research_operation_exits_zero_without_authorizing(tmp_path, advisory_pad, capsys):
    image = tmp_path / "capture.png"
    Image.new("RGB", (640, 480)).save(image)
    with patch("verify_with_antispoof.extract_pretrained_face", side_effect=feature_pair(.1)):
        assert main(["--img1", str(image), "--img2", str(image)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["comparison_label"] == "different" and not report["authenticated"]
