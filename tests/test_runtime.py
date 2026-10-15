"""Real tensor/backend compatibility and source-provenance regression checks."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from extract_template.data_loader import load_dataset_from_folder


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,expected', [
    ('torchkit/backbone/common.py', '53e2025f253406e1d98b0ad86a764ca9f9753a9afbe31ba49ff418f6778e7dbb'),
    ('License.txt', 'e534cd4a5cd86d462e842be79a306a52a45d01062a2d295367a21ccb4175845b'),
    ('LICENSE.face-evoLVe.txt', 'b5ee08d1d7d7165e5d5b7085b4b328703a52f94c7a9c21c5386bad17bd1f2d29'),
])
def test_pinned_upstream_source_and_notices(name, expected):
    data = (ROOT / 'third_party/tface' / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected


def test_irse_vendor_changes_are_only_documented_import_namespace():
    data = (ROOT / 'third_party/tface/torchkit/backbone/model_irse.py').read_bytes()
    data = data.replace(
        b'# Local namespace-only change; upstream commit and notices are in ../../README.md.\n'
        b'from .common import initialize_weights, Flatten, SEModule',
        b'from torchkit.backbone.common import initialize_weights, Flatten, SEModule',
    )
    assert hashlib.sha256(data).hexdigest() == '40582defaff7812e102942f46e3fb2d8eaf30e3beed9775875aebaba2347fb2f'


def test_dataset_normalization_matches_upstream_endpoints(tmp_path):
    folder = tmp_path / 'faces'
    folder.mkdir()
    Image.new('RGB', (112, 112), 'black').save(folder / 'black.png')
    Image.new('RGB', (112, 112), 'white').save(folder / 'white.png')
    (tmp_path / 'ann.txt').write_text('0 black.png white.png\n')
    images, _ = load_dataset_from_folder(tmp_path, 'faces', 'ann.txt')
    np.testing.assert_array_equal(images[0], np.full((3, 112, 112), -1, dtype=np.float32))
    np.testing.assert_array_equal(images[1], np.ones((3, 112, 112), dtype=np.float32))


@pytest.mark.parametrize('pixel,expected', [(0, -1), (255, 1)])
def test_trained_extraction_normalization_matches_upstream(monkeypatch, tmp_path, pixel, expected):
    torch = pytest.importorskip('torch')
    import verify_with_antispoof as core
    captured = []

    class Capture:
        def __call__(self, tensor, flip):
            captured.append(tensor.clone())
            return torch.ones((1, 512)), torch.ones((1, 16, 7, 7))

    checkpoint = tmp_path / 'capture.pth'
    checkpoint.write_bytes(b'not loaded: isolates preprocessing with real tensors')
    monkeypatch.setattr(core, '_load_model', lambda *args: (torch, Capture(), torch.device('cpu')))
    core.extract_face_features(np.full((112, 112, 3), pixel, dtype=np.uint8), checkpoint)
    torch.testing.assert_close(captured[0], torch.full((1, 3, 112, 112), float(expected)))


def test_real_pinned_model_forward_and_attention():
    torch = pytest.importorskip('torch')
    from modules.model import SlerpFace
    from third_party.tface.torchkit.backbone.model_irse import BasicBlockIR
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model = SlerpFace([112, 112], 18, group_size=16).eval()
        assert isinstance(model.body[0], BasicBlockIR)
        with torch.inference_mode():
            vector, groups = model(torch.zeros((1, 3, 112, 112)), flip=True)
            weights = model.gen_weight_map(groups)
        assert vector.shape == (1, 512)
        assert groups.shape == (1, 16, 7, 7)
        assert weights.shape == (1, 7, 7, 1)
        assert torch.isfinite(vector).all() and torch.isfinite(groups).all()
        torch.testing.assert_close(weights.sum(dim=(1, 2, 3)), torch.ones(1))
        with pytest.raises(ValueError, match='shape'):
            model(torch.zeros((1, 3, 100, 112)))
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize('layers,mode,size', [(152, 'ir', 112), (200, 'ir_se', 224)])
def test_deep_backbone_channels_and_shapes_without_allocating_gigabyte_weights(layers, mode, size):
    torch = pytest.importorskip('torch')
    from modules.model import SlerpFace
    # PyTorch's real meta operators validate the entire convolution/linear
    # shape graph. Real IR-18/50 numerics are tested separately above/below.
    with torch.device('meta'):
        model = SlerpFace([size, size], layers, mode=mode).eval()
        vector, groups = model(torch.zeros((1, 3, size, size)), flip=False)
    assert model.group_cov[0].in_channels == 2048
    assert vector.shape == (1, 512)
    assert groups.shape == (1, 16, size // 16, size // 16)


def test_ir50_generated_checkpoint_roundtrip_is_not_biometric_validation(tmp_path):
    torch = pytest.importorskip('torch')
    from modules.model import SlerpFace
    import verify_with_antispoof as core
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        model = SlerpFace([112, 112], 50, group_size=16).eval()
        checkpoint = tmp_path / 'random-test-only.pth'
        torch.save(model.state_dict(), checkpoint)
        del model
        groups, vector = core.extract_face_features(
            np.zeros((112, 112, 3), dtype=np.uint8), model_path=checkpoint
        )
        assert groups.shape == (49, 16) and vector.shape == (512,)
        assert np.isfinite(groups).all() and np.isfinite(vector).all()
        assert core._load_model.cache_info().currsize >= 1
    finally:
        core._load_model.cache_clear()
        torch.set_num_threads(old_threads)


def test_runtime_report_keeps_assurance_boundary():
    from scripts.check_runtime import inspect_runtime
    report = inspect_runtime()
    assert report['authentication_enabled'] is False
    assert report['checks']['trained_checkpoint']['available'] is False
    assert report['checks']['trained_checkpoint']['status'] in ('missing', 'not_tested')
    assert 'Capability checks only' in report['assurance']


def test_runtime_cli_requires_an_explicit_successful_checkpoint(monkeypatch, capsys):
    import scripts.check_runtime as runtime
    monkeypatch.setattr(runtime, 'inspect_runtime', lambda checkpoint: {
        'demo_runtime_ready': True,
        'checks': {'trained_checkpoint': {'available': False}},
    })
    assert runtime.main([]) == 0
    assert runtime.main(['--require-model']) == 2
    assert 'trained_checkpoint' in capsys.readouterr().out


def test_runtime_cli_can_require_actual_prototype_and_presentation_capabilities(monkeypatch, capsys):
    import scripts.check_runtime as runtime
    state = {'demo_runtime_ready': True, 'prototype_runtime_ready': False,
             'checks': {'presentation_model': {'available': False}}}
    monkeypatch.setattr(runtime, 'inspect_runtime', lambda checkpoint: state)
    assert runtime.main(['--require-prototype']) == 2
    state['prototype_runtime_ready'] = True
    assert runtime.main(['--require-prototype']) == 0
    assert runtime.main(['--require-prototype', '--require-presentation']) == 2
    state['checks']['presentation_model']['available'] = True
    assert runtime.main(['--require-prototype', '--require-presentation']) == 0
    capsys.readouterr()


def test_runtime_reports_broken_face_assets_without_traceback(monkeypatch):
    cv2 = pytest.importorskip('cv2')
    from scripts.check_runtime import inspect_runtime
    def broken(*args):
        raise cv2.error('simulated invalid classifier asset')
    monkeypatch.setattr(cv2, 'CascadeClassifier', broken)
    report = inspect_runtime()
    assert report['checks']['face_detection']['available'] is False
    assert report['demo_runtime_ready'] is False


def test_runtime_reports_optional_extractor_dependency_errors(monkeypatch, tmp_path):
    pytest.importorskip('torch')
    from scripts.check_runtime import inspect_runtime
    import verify_with_antispoof as core
    def unavailable(*args, **kwargs):
        raise ImportError('simulated optional dependency failure')
    monkeypatch.setattr(core, 'extract_face_features', unavailable)
    checkpoint = tmp_path / 'unused.pth'
    checkpoint.write_bytes(b'not loaded')
    report = inspect_runtime(checkpoint)
    assert report['checks']['trained_checkpoint']['status'] == 'unavailable_or_incompatible'
