"""Contract, preprocessing, integrity, and real pretrained PAD smoke checks."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from modules import presentation_backend as backend
from scripts import provision_pad_model


@pytest.fixture
def image():
    return Image.new("RGB", (160, 160), (255, 64, 8))


def fake_models(monkeypatch, probabilities=(0.1, 0.7, 0.2)):
    monkeypatch.setattr(backend, "_read_verified_model", lambda path, spec: b"test")
    monkeypatch.setattr(backend, "_predict", lambda *args: np.asarray(probabilities))


def test_missing_models_are_unavailable_without_network(image, tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("Inference must not download models or attempt partial inference.")
    monkeypatch.setattr(backend, "_predict", forbidden)
    result = backend.evaluate_presentation(image, (40, 40, 120, 120), tmp_path)
    assert result.status == "UNAVAILABLE"
    assert result.bona_fide_score is None
    assert "provision_pad_model.py" in result.reason
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("box", [
    None, (), (0, 0, 100), (0, 0, 100, 100, 100),
    (-1, 0, 100, 100), (0, -1, 100, 100), (0, 0, 161, 100),
    (0, 0, 100, 161), (100, 40, 40, 100), (40, 40, 40, 100),
    (40, 40, 60, 100), (0, 0, np.nan, 100), (0, 0, np.inf, 100),
])
def test_invalid_boxes_are_inconclusive(image, box):
    result = backend.evaluate_presentation(image, box)
    assert result.status == "INCONCLUSIVE"
    assert result.bona_fide_score is None


def test_invalid_image_is_inconclusive():
    result = backend.evaluate_presentation(np.zeros((2, 2, 3), dtype=np.uint8), (0, 0, 2, 2))
    assert result.status == "INCONCLUSIVE"


def test_advisory_output_uses_bona_fide_class_one(image, monkeypatch):
    fake_models(monkeypatch)
    result = backend.evaluate_presentation(image, (40, 40, 120, 120))
    assert result.status == "MODEL_ESTIMATE"
    assert result.bona_fide_score == pytest.approx(0.7)
    assert result.class_scores == {"attack_class_0": 0.1, "bona_fide": 0.7, "attack_class_2": 0.2}
    assert result.metadata["authentication_enabled"] is False
    assert result.metadata["capture_provenance_verified"] is False
    assert result.metadata["calibrated_for_this_camera"] is False
    assert "live capture and identity are not verified" in result.reason
    assert "label" not in result.to_dict()
    json.dumps(result.to_dict(), allow_nan=False)


def test_crops_use_original_rgb_converted_to_bgr_byte_scale(image, monkeypatch):
    monkeypatch.setattr(backend, "_read_verified_model", lambda path, spec: b"test")
    seen = []
    def predict(spec, payload, crop):
        seen.append((spec.scale, crop.copy()))
        return np.asarray((0.2, 0.3, 0.5))
    monkeypatch.setattr(backend, "_predict", predict)
    result = backend.evaluate_presentation(image, (40, 40, 120, 120))
    assert result.status == "MODEL_ESTIMATE"
    assert [scale for scale, _ in seen] == [2.7, 4.0]
    for _, crop in seen:
        assert crop.shape == (80, 80, 3)
        assert crop.dtype == np.uint8
        np.testing.assert_array_equal(crop[0, 0], (8, 64, 255))


@pytest.mark.parametrize("box", [(0, 0, 80, 80), (80, 80, 160, 160), (0, 80, 80, 160), (80, 0, 160, 80)])
def test_context_crop_remains_valid_near_image_edges(box):
    bgr = np.zeros((160, 160, 3), dtype=np.uint8)
    crop = backend._context_crop(bgr, np.asarray(box), 4)
    assert crop.shape == (80, 80, 3)


@pytest.mark.parametrize("scores", [(np.nan, 0.5, 0.5), (-0.1, 0.5, 0.6), (0.1, 0.2, 0.3)])
def test_invalid_model_probabilities_are_not_reported(image, monkeypatch, scores):
    fake_models(monkeypatch, scores)
    result = backend.evaluate_presentation(image, (40, 40, 120, 120))
    assert result.status == "UNAVAILABLE"
    assert result.bona_fide_score is None
    json.dumps(result.to_dict(), allow_nan=False)


def test_model_payload_is_verified_each_call_even_when_model_is_cached(image, monkeypatch):
    seen = []
    def verified(path, spec):
        seen.append(path)
        if len(seen) > 2:
            raise ValueError("Integrity check failed for replaced checkpoint.")
        return b"test"
    monkeypatch.setattr(backend, "_read_verified_model", verified)
    monkeypatch.setattr(backend, "_predict", lambda *args: np.asarray((0.1, 0.7, 0.2)))
    assert backend.evaluate_presentation(image, (40, 40, 120, 120)).status == "MODEL_ESTIMATE"
    assert backend.evaluate_presentation(image, (40, 40, 120, 120)).status == "UNAVAILABLE"


def test_missing_torch_backend_is_actionable_and_does_not_fall_back(image, monkeypatch):
    fake_models(monkeypatch)
    def missing(*args):
        raise ImportError("No module named torch")
    monkeypatch.setattr(backend, "_predict", missing)
    result = backend.evaluate_presentation(image, (40, 40, 120, 120))
    assert result.status == "UNAVAILABLE"
    assert "torch" in result.reason
    assert result.class_scores == {}


def test_tensor_preprocessing_has_no_unintended_division(monkeypatch):
    torch = pytest.importorskip("torch")
    import threading
    class Model:
        def __call__(self, tensor):
            assert tensor.shape == (1, 3, 80, 80)
            assert tensor.dtype == torch.float32
            assert tensor[0, 2, 0, 0].item() == 255
            return torch.tensor([[0.0, 1.0, 0.0]])
    monkeypatch.setattr(backend, "_load_model", lambda *args: (Model(), threading.Lock()))
    crop = np.full((80, 80, 3), 255, dtype=np.uint8)
    prediction = backend._predict(backend.MODEL_SPECS[0], b"test", crop)
    assert prediction[1] > prediction[0]
    assert prediction.sum() == pytest.approx(1)


def test_cpu_thread_configuration_is_bounded_and_does_not_raise_existing_low_limit():
    class Runtime:
        def __init__(self, threads):
            self.threads = threads
            self.changes = []
        def get_num_threads(self):
            return self.threads
        def set_num_threads(self, value):
            self.threads = value
            self.changes.append(value)
    high = Runtime(32)
    backend._configure_cpu_runtime(high)
    backend._configure_cpu_runtime(high)
    assert high.threads == 2
    assert high.changes == [2]
    low = Runtime(1)
    backend._configure_cpu_runtime(low)
    assert low.threads == 1
    assert low.changes == []


def test_verify_bytes_checks_both_size_and_digest():
    spec = backend.PresentationModelSpec("test.pth", "unused", 1, 3, hashlib.sha256(b"abc").hexdigest())
    assert backend.verify_model_bytes(b"abc", spec) == b"abc"
    for payload in (b"ab", b"abcd", b"abd"):
        with pytest.raises(ValueError, match="Integrity"):
            backend.verify_model_bytes(payload, spec)


def test_architecture_source_is_exact_pinned_copy():
    source = Path(__file__).resolve().parents[1] / "third_party" / "silent_face" / "MiniFASNet.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == backend.ARCHITECTURE_SHA256


def test_status_checks_integrity_without_running_inference(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    seen = []
    def verified(path, spec):
        seen.append(spec.filename)
        return b"test"
    monkeypatch.setattr(backend, "_read_verified_model", verified)
    def forbidden(*args):
        pytest.fail("Readiness must not run a fake image through the model.")
    monkeypatch.setattr(backend, "_predict", forbidden)
    status = backend.presentation_backend_status(tmp_path)
    assert status["available"] is True
    assert status["ready"] is True
    assert status["authentication_enabled"] is False
    assert seen == [spec.filename for spec in backend.MODEL_SPECS]


def test_status_missing_models_is_unavailable(tmp_path):
    status = backend.presentation_backend_status(tmp_path)
    assert status["available"] is False
    assert status["ready"] is False
    assert "unavailable" in status["reason"]


def test_provision_existing_verified_files_is_offline(tmp_path, monkeypatch):
    payload = b"abc"
    spec = backend.PresentationModelSpec("test.pth", "unused", 1, 3, hashlib.sha256(payload).hexdigest())
    (tmp_path / spec.filename).write_bytes(payload)
    monkeypatch.setattr(provision_pad_model, "MODEL_SPECS", (spec,))
    def forbidden(*args, **kwargs):
        pytest.fail("Verified existing models must not be downloaded again.")
    monkeypatch.setattr(provision_pad_model.urllib.request, "urlopen", forbidden)
    assert provision_pad_model.provision(tmp_path)["models"][0]["sha256"] == spec.sha256


def test_provision_does_not_replace_a_corrupt_existing_checkpoint(tmp_path, monkeypatch):
    spec = backend.PresentationModelSpec("test.pth", "unused", 1, 3, hashlib.sha256(b"abc").hexdigest())
    (tmp_path / spec.filename).write_bytes(b"bad")
    monkeypatch.setattr(provision_pad_model, "MODEL_SPECS", (spec,))
    with pytest.raises(ValueError, match="Integrity"):
        provision_pad_model.provision(tmp_path)
    assert (tmp_path / spec.filename).read_bytes() == b"bad"


@pytest.mark.parametrize("payload", [b"abc", b"bad", b"abcd"])
def test_download_is_bounded_verified_and_partial_files_removed(tmp_path, monkeypatch, payload):
    spec = backend.PresentationModelSpec("test.pth", "unused", 1, 3, hashlib.sha256(b"abc").hexdigest())
    monkeypatch.setattr(provision_pad_model, "MODEL_SPECS", (spec,))
    @contextmanager
    def response(*args, **kwargs):
        handle = io.BytesIO(payload)
        handle.url = spec.url
        try:
            yield handle
        finally:
            handle.close()
    monkeypatch.setattr(provision_pad_model.urllib.request, "urlopen", response)
    if payload == b"abc":
        provision_pad_model.provision(tmp_path)
        assert (tmp_path / "test.pth").read_bytes() == payload
    else:
        with pytest.raises(ValueError, match="Integrity"):
            provision_pad_model.provision(tmp_path)
        assert not (tmp_path / "test.pth").exists()
    assert not list(tmp_path.glob("*.download"))


def test_download_rejects_redirect_to_another_source(tmp_path, monkeypatch):
    spec = backend.PresentationModelSpec("test.pth", "unused", 1, 3, hashlib.sha256(b"abc").hexdigest())
    monkeypatch.setattr(provision_pad_model, "MODEL_SPECS", (spec,))
    @contextmanager
    def response(*args, **kwargs):
        handle = io.BytesIO(b"abc")
        handle.url = "https://raw.githubusercontent.com/another/repository/model.pth"
        try:
            yield handle
        finally:
            handle.close()
    monkeypatch.setattr(provision_pad_model.urllib.request, "urlopen", response)
    with pytest.raises(ValueError, match="redirect"):
        provision_pad_model.provision(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_real_official_pretrained_models_load_and_produce_repeatable_estimates():
    if not all((backend.DEFAULT_MODEL_DIR / spec.filename).exists() for spec in backend.MODEL_SPECS):
        pytest.skip("Run scripts/provision_pad_model.py for the optional pretrained-model smoke test.")
    import torch
    before = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        image = Image.new("RGB", (160, 160), (128, 128, 128))
        first = backend.evaluate_presentation(image, (40, 40, 120, 120))
        second = backend.evaluate_presentation(image, (40, 40, 120, 120))
    finally:
        torch.set_num_threads(before)
    assert first.status == "MODEL_ESTIMATE", first.reason
    assert first.bona_fide_score == pytest.approx(0.42098977, abs=1e-5)
    assert first.to_dict() == second.to_dict()
    # The neutral square is an execution fixture, not evidence of PAD accuracy.
    assert first.metadata["authentication_enabled"] is False
