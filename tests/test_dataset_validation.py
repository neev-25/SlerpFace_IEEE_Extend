import json

import numpy as np
from PIL import Image
import pytest

from evaluate_dataset import main
from modules.dataset_validation import (
    calibrate_threshold, evaluate_identity_pairs, evaluate_manifest,
    group_cosine, identity_rates, load_manifest,
)


@pytest.fixture
def manifest_file(tmp_path):
    images = {"a1": 34, "a2": 35, "b1": 80, "c1": 120, "c2": 121, "d1": 200}
    for name, color in images.items():
        Image.new("RGB", (112, 112), (color, color // 2, 20)).save(tmp_path / f"{name}.png")
    document = {"schema_version": 1, "data_kind": "synthetic", "source": "Generated unit-test color fields; no real identities",
                "pairs": [
                    {"left": "a1.png", "right": "a2.png", "left_subject": "a", "right_subject": "a", "same_person": True, "split": "calibration"},
                    {"left": "a1.png", "right": "b1.png", "left_subject": "a", "right_subject": "b", "same_person": False, "split": "calibration"},
                    {"left": "c1.png", "right": "c2.png", "left_subject": "c", "right_subject": "c", "same_person": True, "split": "test"},
                    {"left": "c1.png", "right": "d1.png", "left_subject": "c", "right_subject": "d", "same_person": False, "split": "test"},
                ]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def rewrite(path, document):
    path.write_text(json.dumps(document), encoding="utf-8")


def test_calibration_and_heldout_test_are_separate():
    result = calibrate_threshold([.9, .8, .1, -.1], [True, True, False, False], target_fmr=0)
    assert result["threshold"] == .8
    assert result["false_match"]["rate"] == 0
    tested = identity_rates([.7, .95], [True, False], result["threshold"])
    assert tested["false_match"]["rate"] == 1 and tested["false_nonmatch"]["rate"] == 1
    assert result["false_match"]["interval"][1] > 0  # zero observations are not proof of zero risk


def test_equal_scores_can_only_reject_all_at_zero_empirical_fmr():
    result = calibrate_threshold([.99, .99], [True, False], 0)
    assert result["threshold"] > .99
    assert result["false_nonmatch"]["rate"] == 1


@pytest.mark.parametrize("scores,labels", [([np.nan, .1], [True, False]), ([1.1, .1], [True, False]),
                                           ([.9, .1], [1, 0]), ([.9], [True])])
def test_invalid_identity_observations_rejected(scores, labels):
    with pytest.raises(ValueError):
        calibrate_threshold(scores, labels)


def test_manifest_runs_only_as_explicit_synthetic_demo(manifest_file):
    path, _ = manifest_file
    with pytest.raises(ValueError, match="explicit --demo"):
        evaluate_manifest(path)
    result = evaluate_manifest(path, demo=True)
    assert result["authentication_enabled"] is False
    assert result["identity"]["backend"] == "untrained_synthetic_demo"
    assert "a1.png" not in json.dumps(result)
    assert "left_subject" not in json.dumps(result)
    assert result["identity"]["methods"]["slerp_group_cosine"]["test"]["false_match"]["trials"] == 1


def test_subject_overlap_rejected(manifest_file):
    path, document = manifest_file
    document["pairs"][2].update(left_subject="a", right_subject="a")
    rewrite(path, document)
    with pytest.raises(ValueError, match="subjects must be disjoint"):
        load_manifest(path)


@pytest.mark.parametrize("split", [[], {}, ["test"]])
def test_cli_malformed_split_is_controlled_error(manifest_file, tmp_path, capsys, split):
    path, document = manifest_file
    document["pairs"][0]["split"] = split
    rewrite(path, document)
    output = tmp_path / "invalid-result.json"
    assert main(["--manifest", str(path), "--out", str(output), "--demo"]) == 2
    assert "Every identity pair needs split calibration or test" in capsys.readouterr().err
    assert not output.exists()


def test_renamed_image_leakage_rejected(manifest_file):
    path, _ = manifest_file
    (path.parent / "c1.png").write_bytes((path.parent / "a1.png").read_bytes())
    with pytest.raises(ValueError, match="contents overlap|conflicting"):
        load_manifest(path)


def test_copied_image_does_not_count_as_a_distinct_capture(manifest_file):
    path, _ = manifest_file
    (path.parent / "a2.png").write_bytes((path.parent / "a1.png").read_bytes())
    with pytest.raises(ValueError, match="distinct captures"):
        load_manifest(path)


def test_image_cannot_be_assigned_conflicting_subjects(manifest_file):
    path, document = manifest_file
    document["pairs"][1]["left_subject"] = "different-claimed-subject"
    rewrite(path, document)
    with pytest.raises(ValueError, match="conflicting"):
        load_manifest(path)


def test_reversed_pair_copies_do_not_inflate_trials(manifest_file):
    path, document = manifest_file
    (path.parent / "a1-copy.png").write_bytes((path.parent / "a1.png").read_bytes())
    (path.parent / "a2-copy.png").write_bytes((path.parent / "a2.png").read_bytes())
    document["pairs"].append({"left": "a2-copy.png", "right": "a1-copy.png", "left_subject": "a",
                              "right_subject": "a", "same_person": True, "split": "calibration"})
    rewrite(path, document)
    with pytest.raises(ValueError, match="Duplicate or reversed"):
        load_manifest(path)


def test_real_identity_requires_an_explicit_model_for_provenance(manifest_file):
    path, document = manifest_file
    document["data_kind"] = "real"
    rewrite(path, document)
    with pytest.raises(ValueError, match="explicit model"):
        evaluate_manifest(path)


def test_sface_evaluation_uses_provisioned_alignment_and_records_model_hashes(manifest_file, monkeypatch):
    from types import SimpleNamespace
    path, document = manifest_file
    document["data_kind"] = "real"
    rewrite(path, document)
    artifacts = {"sface": {"sha256": "pinned-test-digest", "verified": True}}
    monkeypatch.setattr("modules.dataset_validation.face_backend_status", lambda: {"ready": True, "models": artifacts})

    def extract(image):
        rng = np.random.default_rng(np.asarray(image).sum())
        vector = rng.normal(size=128)
        vector /= np.linalg.norm(vector)
        return SimpleNamespace(vector=vector, groups=vector.reshape(-1, 16))

    monkeypatch.setattr("modules.dataset_validation.extract_pretrained_face", extract)
    result = evaluate_manifest(path, backend="sface")
    assert result["model_artifacts"] == artifacts
    assert result["identity"]["backend"] == "sface_trained_model_research"
    assert result["authentication_enabled"] is False


def test_sface_and_ir50_configuration_cannot_be_mixed(manifest_file):
    path, _ = manifest_file
    with pytest.raises(ValueError, match="--backend ir50"):
        evaluate_manifest(path, backend="sface", model_path="not-used.pth", demo=True)


def test_real_ir50_evaluator_passes_aligned_pixels_to_legacy_extractor(manifest_file, monkeypatch):
    from types import SimpleNamespace
    path, _ = manifest_file
    aligned = np.full((112, 112, 3), 77, dtype=np.uint8)
    seen = []
    monkeypatch.setattr("modules.dataset_validation.extract_pretrained_face", lambda image: SimpleNamespace(aligned_rgb=aligned))

    def extract(image, model_path=None, demo=False):
        seen.append(np.asarray(image).copy())
        assert demo is False and model_path == "controlled-checkpoint"
        return np.ones((8, 16)), np.ones(128)

    monkeypatch.setattr("modules.dataset_validation.extract_face_features", extract)
    result = evaluate_identity_pairs(load_manifest(path), backend="ir50", model_path="controlled-checkpoint")
    assert result["backend"] == "ir50_trained_model_research"
    assert seen and all(np.array_equal(image, aligned) for image in seen)


def test_checkpoint_cannot_change_under_reported_digest(manifest_file, tmp_path, monkeypatch):
    path, document = manifest_file
    document["data_kind"] = "real"
    rewrite(path, document)
    model = tmp_path / "test.pth"
    model.write_bytes(b"original-weights")

    def changed(*args, **kwargs):
        model.write_bytes(b"replaced-weights")
        return {"note": "isolated manifest/checkpoint binding test"}

    monkeypatch.setattr("modules.dataset_validation.evaluate_identity_pairs", changed)
    with pytest.raises(ValueError, match="Checkpoint changed"):
        evaluate_manifest(path, model_path=model)


def test_mislabeled_pair_rejected(manifest_file):
    path, document = manifest_file
    document["pairs"][0]["same_person"] = False
    rewrite(path, document)
    with pytest.raises(ValueError, match="labels disagree"):
        load_manifest(path)


def test_no_self_comparison_or_duplicate_pair(manifest_file):
    path, document = manifest_file
    document["pairs"][0]["right"] = "a1.png"
    rewrite(path, document)
    with pytest.raises(ValueError, match="distinct captures"):
        load_manifest(path)


def test_path_escape_rejected(manifest_file):
    path, document = manifest_file
    document["pairs"][0]["left"] = "../outside.png"
    rewrite(path, document)
    with pytest.raises(ValueError, match="outside"):
        load_manifest(path)


def test_changed_image_cannot_be_scored_with_stale_digest(manifest_file):
    path, _ = manifest_file
    validated = load_manifest(path)
    Image.new("RGB", (112, 112), "black").save(path.parent / "a1.png")
    with pytest.raises(ValueError, match="changed after"):
        evaluate_identity_pairs(validated, demo=True)


def test_real_data_never_uses_statistical_extractor(manifest_file):
    path, document = manifest_file
    document["data_kind"] = "real"
    rewrite(path, document)
    with pytest.raises(ValueError, match="cannot evaluate"):
        evaluate_manifest(path, demo=True)


def test_cli_no_overwrite_and_no_authentication(manifest_file, tmp_path):
    path, _ = manifest_file
    output = tmp_path / "report.json"
    assert main(["--manifest", str(path), "--out", str(output), "--demo"]) == 0
    contents = output.read_text(encoding="utf-8")
    assert json.loads(contents)["authentication_enabled"] is False
    assert main(["--manifest", str(path), "--out", str(output), "--demo"]) == 2
    assert output.read_text(encoding="utf-8") == contents


def test_group_cosine_uses_each_group_not_shared_norm():
    a = np.zeros((2, 16))
    a[0, 0] = 100
    a[1, 1] = 1
    b = a.copy()
    b[1, 1] = -1
    assert group_cosine(a, b) == 0
    with pytest.raises(ValueError):
        group_cosine(np.empty((0, 16)), np.empty((0, 16)))


def test_presentation_rates_keep_face_failures_explicit(tmp_path):
    from types import SimpleNamespace
    from modules.dataset_validation import evaluate_presentations
    Image.new("RGB", (112, 112), "red").save(tmp_path / "capture.png")
    path = tmp_path / "manifest.json"
    rewrite(path, {"schema_version": 1, "data_kind": "real", "source": "Unit test, labels not actual physical evidence",
                   "presentations": [{"image": "capture.png", "subject": "test", "kind": "bona_fide", "capture_kind": "camera"}]})
    detector = SimpleNamespace(evaluate=lambda image: SimpleNamespace(is_live=False, verdict="NO_FACE"))
    result = evaluate_presentations(load_manifest(path), detector=detector)
    assert result["classes"]["bona_fide"]["face_detection_failures"] == 1
    assert result["classes"]["bona_fide"]["rate"] == 1
