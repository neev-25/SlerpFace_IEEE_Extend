"""Real inference smoke tests and input/integrity failure contracts."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytest

from modules import face_backend as backend
from scripts import provision_face_models as setup


FIXTURE = Path(__file__).parent / "fixtures" / "astronaut.png"
OTHER_FIXTURE = Path(__file__).parent / "fixtures" / "grace_hopper.jpg"


def fake_backend(monkeypatch, *, faces="single", feature=None):
    row = np.array([100, 80, 100, 120, 125, 110, 170, 110, 150, 140, 130, 165, 170, 165, .95], dtype=np.float32)

    class Detector:
        def setInputSize(self, size):
            self.input_size = size

        def detect(self, image):
            if faces == "none":
                return 0, None
            if faces == "multiple":
                return 0, np.stack([row, row])
            if faces == "small":
                adjusted = row.copy()
                adjusted[2:4] = 20
                return 0, adjusted[None, :]
            return 0, row[None, :]

    class Recognizer:
        def alignCrop(self, bgr, face):
            self.source = bgr
            self.face = face
            return np.full((112, 112, 3), (11, 22, 33), dtype=np.uint8)

        def feature(self, aligned):
            return np.arange(1, 129, dtype=np.float32)[None, :] if feature is None else feature

    detector, recognizer = Detector(), Recognizer()
    monkeypatch.setattr(backend, "_get_backend", lambda model_dir=None: (cv2, detector, recognizer))
    return detector, recognizer


@pytest.mark.parametrize("image", [np.zeros((0, 4, 3)), np.full((5, 5, 3), np.nan), np.zeros((5, 5, 2)), np.full((5, 5, 3), 256), b"invalid"])
def test_invalid_pixels_are_rejected_before_loading_models(monkeypatch, image):
    monkeypatch.setattr(backend, "_get_backend", lambda *_: pytest.fail("invalid input reached inference"))
    with pytest.raises(backend.FaceInputError) as error:
        backend.extract_pretrained_face(image)
    assert error.value.code == "INVALID_IMAGE"


@pytest.mark.parametrize("faces,code", [("none", "NO_FACE"), ("multiple", "MULTIPLE_FACES"), ("small", "LOW_QUALITY")])
def test_capture_failures_are_distinguished(monkeypatch, faces, code):
    fake_backend(monkeypatch, faces=faces)
    with pytest.raises(backend.FaceInputError) as error:
        backend.extract_pretrained_face(np.zeros((400, 400, 3), dtype=np.uint8))
    assert error.value.code == code


def test_detection_is_bounded_but_alignment_uses_original_pixels(monkeypatch):
    detector, recognizer = fake_backend(monkeypatch)
    rgb = np.full((1000, 2000, 3), (10, 20, 30), dtype=np.uint8)
    result = backend.extract_pretrained_face(rgb)
    assert detector.input_size == (640, 320)
    assert recognizer.source.shape == rgb.shape
    np.testing.assert_array_equal(recognizer.source[0, 0], [30, 20, 10])
    np.testing.assert_allclose(result.face_box, [312.5, 250, 625, 625])
    np.testing.assert_allclose(result.landmarks[0], [390.625, 343.75])
    assert result.vector.shape == (128,)
    assert result.groups.shape == (8, 16)
    assert np.linalg.norm(result.vector) == pytest.approx(1)
    np.testing.assert_array_equal(result.groups.reshape(-1), result.vector)
    np.testing.assert_array_equal(result.aligned_rgb[0, 0], [33, 22, 11])
    assert result.quality["eye_distance_pixels"] > 10
    assert "array" not in repr(result)


@pytest.mark.parametrize("feature", [np.zeros((1, 128)), np.full((1, 128), np.nan), np.ones((1, 127))])
def test_invalid_network_outputs_never_become_embeddings(monkeypatch, feature):
    fake_backend(monkeypatch, feature=feature)
    with pytest.raises(backend.FaceBackendUnavailable):
        backend.extract_pretrained_face(np.zeros((400, 400, 3), dtype=np.uint8))


def test_missing_models_are_actionable_without_network(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_: pytest.fail("status performed network I/O"))
    status = backend.face_backend_status(tmp_path)
    assert status["ready"] is False
    assert set(status["models"]) == {"yunet", "sface"}
    assert "provision_face_models.py" in status["reason"]
    with pytest.raises(backend.FaceBackendUnavailable, match="provision_face_models"):
        backend.extract_pretrained_face(np.zeros((100, 100, 3), dtype=np.uint8), tmp_path)


def test_integrity_failure_cannot_initialize_models(tmp_path):
    for spec in backend.MODEL_SPECS.values():
        (tmp_path / spec["filename"]).write_bytes(b"not a model")
    status = backend.face_backend_status(tmp_path)
    assert not status["ready"]
    assert all(item["present"] and not item["verified"] for item in status["models"].values())


def test_digest_cache_invalidates_when_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"first")
    monkeypatch.setattr(backend, "MODEL_SPECS", {"sample": {"size": 5}})
    before = backend._verified_digest(backend._fingerprint(path))
    path.write_bytes(b"other")
    after = backend._verified_digest(backend._fingerprint(path))
    assert before == hashlib.sha256(b"first").hexdigest()
    assert after == hashlib.sha256(b"other").hexdigest()
    assert before != after


def require_real_models():
    if not FIXTURE.exists():
        pytest.fail("The provenance-documented public-domain fixture is missing.")
    if not backend.face_backend_status()["ready"]:
        pytest.skip("Run scripts/provision_face_models.py to enable real-model integration checks.")


def test_real_models_produce_stable_aligned_normalized_features():
    require_real_models()
    first = backend.extract_pretrained_face(FIXTURE)
    second = backend.extract_pretrained_face(FIXTURE)
    assert first.vector.shape == (128,)
    assert first.aligned_rgb.shape == (112, 112, 3)
    assert first.detection_score > .75
    assert np.linalg.norm(first.vector) == pytest.approx(1)
    assert np.isfinite(first.vector).all()
    np.testing.assert_allclose(first.vector, second.vector, atol=1e-6)
    assert float(first.vector @ second.vector) > .9999
    assert all(0 <= value <= 512 for value in first.face_box)


def test_real_models_reject_blank_and_multiple_face_images():
    require_real_models()
    with pytest.raises(backend.FaceInputError) as blank:
        backend.extract_pretrained_face(np.zeros((512, 512, 3), dtype=np.uint8))
    assert blank.value.code == "NO_FACE"
    with Image.open(FIXTURE) as source:
        rgb = np.asarray(source.convert("RGB"))
    with pytest.raises(backend.FaceInputError) as multiple:
        backend.extract_pretrained_face(np.concatenate([rgb, rgb], axis=1))
    assert multiple.value.code == "MULTIPLE_FACES"


def test_real_pretrained_pair_smoke_distinguishes_two_public_portraits():
    require_real_models()
    first = backend.extract_pretrained_face(FIXTURE)
    with Image.open(FIXTURE) as source:
        transformed = ImageEnhance.Brightness(source.convert("RGB")).enhance(.93).rotate(
            2, resample=Image.Resampling.BILINEAR,
        )
    same_source = backend.extract_pretrained_face(transformed)
    other = backend.extract_pretrained_face(OTHER_FIXTURE)
    same_cosine = float(first.vector @ same_source.vector)
    different_cosine = float(first.vector @ other.vector)
    # Fixed upstream OpenCV example operating point; not fitted to these images.
    # This is only a functional regression on two known public photographs.
    example_threshold = .363
    assert same_cosine > example_threshold + .1
    assert different_cosine < example_threshold - .1
    assert same_cosine - different_cosine > .5


@pytest.mark.parametrize("path,digest", [
    (FIXTURE, "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"),
    (OTHER_FIXTURE, "a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130"),
])
def test_public_fixture_bytes_match_documented_source(path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


class Response:
    status = 200

    def __init__(self, data, length=None):
        import io
        self.stream = io.BytesIO(data)
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stream.close()

    def read(self, size):
        return self.stream.read(size)


def configure_download(monkeypatch, data, length=None):
    class Opener:
        def open(self, request, timeout):
            return Response(data, length)
    monkeypatch.setattr(setup.urllib.request, "build_opener", lambda *_: Opener())


def test_provision_download_is_verified_atomic_and_repeatable(tmp_path, monkeypatch):
    data = b"trusted model bytes"
    configure_download(monkeypatch, data, len(data))
    target = tmp_path / "model.onnx"
    digest = hashlib.sha256(data).hexdigest()
    result = setup.download_verified("https://media.githubusercontent.com/media/model", target, digest, len(data))
    assert result["state"] == "installed"
    assert target.read_bytes() == data
    assert list(tmp_path.glob("*.part")) == []
    monkeypatch.setattr(setup.urllib.request, "build_opener", lambda *_: pytest.fail("verified model was downloaded again"))
    assert setup.download_verified("https://media.githubusercontent.com/media/model", target, digest, len(data))["state"] == "already_verified"


@pytest.mark.parametrize("data,length", [(b"unexpected", None), (b"too long", 99), (b"oversized bytes", None)])
def test_failed_provision_preserves_existing_file_and_cleans_temporary(tmp_path, monkeypatch, data, length):
    configure_download(monkeypatch, data, length)
    target = tmp_path / "model.onnx"
    target.write_bytes(b"existing")
    with pytest.raises(ValueError):
        setup.download_verified("https://raw.githubusercontent.com/model", target, "0" * 64, 10, replace=True)
    assert target.read_bytes() == b"existing"
    assert len(list(tmp_path.iterdir())) == 1


def test_provision_rejects_untrusted_host_and_existing_unverified_file(tmp_path):
    target = tmp_path / "model.onnx"
    with pytest.raises(ValueError, match="official HTTPS"):
        setup.download_verified("http://example.com/model", target, "0" * 64, 10)
    target.write_bytes(b"existing")
    with pytest.raises(ValueError, match="exists but is unverified"):
        setup.download_verified("https://raw.githubusercontent.com/model", target, "0" * 64, 10)


def test_provision_redirects_cannot_leave_trusted_https_hosts():
    handler = setup._TrustedRedirects()
    with pytest.raises(ValueError, match="outside official"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://example.com/model")
