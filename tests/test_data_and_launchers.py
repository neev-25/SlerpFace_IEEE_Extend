"""Regression checks for dataset input, model boundaries, and supported launchers."""
import ast
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import pytest

from extract_template.data_loader import load_dataset_from_folder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    folder = tmp_path / 'faces'
    folder.mkdir()
    Image.new('RGB', (12, 14), (20, 40, 60)).save(folder / 'a.png')
    Image.new('RGB', (12, 14), (80, 100, 120)).save(folder / 'b.png')
    return tmp_path


@pytest.mark.parametrize('prefix', ['', 'faces/'])
def test_dataset_resolves_named_folder_and_preserves_pairs(dataset, prefix):
    (dataset / 'ann.txt').write_text(
        f'1 {prefix}a.png {prefix}b.png\n0 {prefix}b.png {prefix}a.png\n',
        encoding='utf-8',
    )
    images, labels = load_dataset_from_folder(dataset, 'faces', 'ann.txt')
    assert images.shape == (4, 3, 112, 112)
    assert images.dtype == np.float32
    assert labels.tolist() == [True, False]
    np.testing.assert_array_equal(images[0], images[3])
    np.testing.assert_array_equal(images[1], images[2])


@pytest.mark.parametrize('row,message', [
    ('1 a.png', 'expected'),
    ('2 a.png b.png', 'label'),
    ('true a.png b.png', 'label'),
    ('1 ../outside.png b.png', 'inside'),
    ('1 C:/outside.png b.png', 'inside'),
])
def test_dataset_reports_invalid_annotation_line(dataset, row, message):
    (dataset / 'ann.txt').write_text('\n' + row, encoding='utf-8')
    with pytest.raises(ValueError, match=f'ann.txt:2:.*{message}'):
        load_dataset_from_folder(dataset, 'faces', 'ann.txt')


def test_dataset_missing_image_is_actionable(dataset):
    (dataset / 'ann.txt').write_text('1 missing.png b.png', encoding='utf-8')
    with pytest.raises(FileNotFoundError, match='ann.txt:1: image not found'):
        load_dataset_from_folder(dataset, 'faces', 'ann.txt')


def test_empty_dataset_has_consistent_shapes(dataset):
    (dataset / 'ann.txt').write_text('\n\n', encoding='utf-8')
    images, labels = load_dataset_from_folder(dataset, 'faces', 'ann.txt')
    assert images.shape == (0, 3, 112, 112)
    assert images.dtype == np.float32
    assert labels.shape == (0,)
    assert labels.dtype == np.bool_


def run_python(*args, cwd=PROJECT_ROOT):
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run([sys.executable, '-B', *map(str, args)], cwd=cwd,
                          env=env, capture_output=True, text=True, timeout=30)


def test_extraction_help_does_not_require_model_dependencies():
    result = run_python('-m', 'extract_template.verification', '--help')
    assert result.returncode == 0, result.stderr
    assert '--model_path' in result.stdout


@pytest.mark.parametrize('extra', [['--batch_size', '0'], ['--gpu_ids', '-1'], []])
def test_extraction_errors_do_not_create_outputs(tmp_path, extra):
    output = tmp_path / 'output'
    result = run_python('-m', 'extract_template.verification',
                        '--model_path', tmp_path / 'missing.pth',
                        '--data_root', tmp_path, '--output_dir', output, *extra)
    assert result.returncode == 2
    assert 'Traceback' not in result.stderr
    assert not output.exists()


def test_extraction_missing_backend_does_not_write_outputs(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'unused: dependency validation comes before model loading')
    (tmp_path / 'val').mkdir()
    output = tmp_path / 'output'
    # Exercise the missing-backend contract even in a fully provisioned runtime.
    # Only this child process blocks Torch import; installed packages stay intact.
    bootstrap = (
        "import builtins,runpy,sys\n"
        "original_import=builtins.__import__\n"
        "def guarded_import(name,*args,**kwargs):\n"
        "    if name=='torch' or name.startswith('torch.'):\n"
        "        raise ModuleNotFoundError('test: PyTorch unavailable',name='torch')\n"
        "    return original_import(name,*args,**kwargs)\n"
        "builtins.__import__=guarded_import\n"
        "sys.argv[0]='extract_template.verification'\n"
        "runpy.run_module('extract_template.verification',run_name='__main__')\n"
    )
    result = run_python('-c', bootstrap,
                        '--model_path', checkpoint, '--data_root', tmp_path,
                        '--output_dir', output)
    assert result.returncode == 2
    assert 'requires PyTorch' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize('argv', [None, ['--demo', '--img1', 'sample.png']])
def test_legacy_image_entry_forwards_arguments_and_status(monkeypatch, argv):
    forwarded = []
    def verify_main(arguments):
        forwarded.append(arguments)
        return 1
    monkeypatch.setitem(sys.modules, 'verify_with_antispoof', types.SimpleNamespace(main=verify_main))
    spec = importlib.util.spec_from_file_location('legacy_image_entry', PROJECT_ROOT / 'test_my_image.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(argv) == 1
    assert forwarded == [argv]


def test_legacy_image_entry_import_has_no_model_side_effects():
    result = run_python('-c',
                        "import runpy,sys; runpy.run_path('test_my_image.py', run_name='legacy_import'); "
                        "assert 'torch' not in sys.modules; assert 'verify_with_antispoof' not in sys.modules")
    assert result.returncode == 0, result.stderr


POWERSHELL = shutil.which('pwsh') or shutil.which('powershell')


def run_launcher(script, *args, cwd, env=None):
    return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-File',
                           str(PROJECT_ROOT / script), *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=30)


@pytest.mark.skipif(POWERSHELL is None, reason='PowerShell unavailable')
def test_legacy_pipeline_fails_before_side_effects(tmp_path):
    result = run_launcher('run_pipeline.ps1', cwd=tmp_path)
    assert result.returncode == 2
    assert 'not included' in result.stderr
    assert 'extract_template.verification --help' in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(POWERSHELL is None, reason='PowerShell unavailable')
def test_launcher_invalid_mode_is_operational_error(tmp_path):
    result = run_launcher('run_guardface.ps1', '-Mode', 'invalid', cwd=tmp_path)
    assert result.returncode == 2
    assert 'Available modes' in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(POWERSHELL is None or os.name != 'nt', reason='Windows PowerShell stub check')
@pytest.mark.parametrize('mode', ['cli', 'demo', 'eval', 'test'])
@pytest.mark.parametrize('exit_code', [0, 1, 2])
def test_launcher_resolves_root_and_propagates_native_status(tmp_path, mode, exit_code):
    stub = tmp_path / 'fake-python.cmd'
    log = tmp_path / 'calls.txt'
    stub.write_text('@echo off\necho %CD%>> "%GUARDFACE_TEST_LOG%"\n'
                    'echo %*>> "%GUARDFACE_TEST_LOG%"\n'
                    'exit /b %GUARDFACE_TEST_EXIT%\n', encoding='utf-8')
    env = os.environ.copy()
    env.update(GUARDFACE_TEST_LOG=str(log), GUARDFACE_TEST_EXIT=str(exit_code))
    result = run_launcher('run_guardface.ps1', '-Mode', mode, '-PythonExe', stub,
                          cwd=tmp_path, env=env)
    assert result.returncode == exit_code, result.stderr
    calls = log.read_text(encoding='utf-8')
    assert str(PROJECT_ROOT).lower() in calls.lower()
    assert str(tmp_path).lower() not in calls.lower()
    if mode == 'cli':
        assert '--demo' in calls
        assert calls.count('verify_with_antispoof.py') == (1 if exit_code == 2 else 2)


def tensor_function(path, name, class_name=None):
    """Exercise actual numerical methods without requiring external torchkit."""
    torch = pytest.importorskip('torch', reason='PyTorch is not installed; tensor runtime remains unvalidated')
    tree = ast.parse((PROJECT_ROOT / path).read_text(encoding='utf-8-sig'))
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'torch': torch, 'F': torch.nn.functional, 'math': math}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return torch, namespace[name]


def test_zero_embedding_normalization_is_finite():
    torch, normalize = tensor_function('extract_template/utils.py', 'l2_norm')
    result = normalize(torch.tensor([[0., 0.], [3., 4.]]))
    assert torch.isfinite(result).all()
    assert torch.equal(result[0], torch.zeros(2))
    torch.testing.assert_close(result[1], torch.tensor([.6, .8]))


def test_constant_attention_is_uniform_and_does_not_mutate_input():
    torch, normalize = tensor_function('modules/model.py', 'divByNorm', 'XCosAttention')
    original = torch.ones(2, 1, 7, 7)
    result = normalize(None, original)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full_like(original, 1 / 49))
    torch.testing.assert_close(original, torch.ones_like(original))
    torch.testing.assert_close(result.sum(dim=(2, 3)), torch.ones(2, 1))


def test_attention_softmax_validates_temperature_without_mutation():
    torch, softmax = tensor_function('modules/model.py', 'softmax', 'XCosAttention')
    original = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    clone = original.clone()
    result = softmax(None, original, T=2)
    torch.testing.assert_close(original, clone)
    torch.testing.assert_close(result.sum(), torch.tensor(1.))
    for invalid in (0, -1, float('nan'), float('inf')):
        with pytest.raises(ValueError, match='temperature'):
            softmax(None, original, T=invalid)
