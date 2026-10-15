"""Report concrete runtime capabilities without claiming biometric assurance."""

import argparse
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def inspect_runtime(checkpoint=None):
    """Check imports, an executable face classifier, and optional trained loading."""
    checks = {}
    for package, module in (
        ('numpy', 'numpy'), ('scipy', 'scipy'), ('Pillow', 'PIL.Image'),
        ('streamlit', 'streamlit'), ('matplotlib', 'matplotlib'),
        ('opencv-python', 'cv2'),
    ):
        try:
            importlib.import_module(module)
            checks[package] = {'available': True, 'version': version(package)}
        except (ImportError, OSError, PackageNotFoundError) as exc:
            checks[package] = {'available': False, 'error_type': type(exc).__name__}

    try:
        import cv2
        import numpy as np
        classifier = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml')
        )
        if classifier.empty():
            raise RuntimeError('Face classifier data could not be loaded')
        faces = classifier.detectMultiScale(np.zeros((112, 112), dtype=np.uint8))
        checks['face_detection'] = {'available': True, 'blank_faces': len(faces)}
    except Exception as exc:
        # OpenCV may raise its own extension exception for unreadable assets.
        # This diagnostic must report an unavailable capability, not crash.
        checks['face_detection'] = {'available': False, 'error_type': type(exc).__name__}

    try:
        import torch
        from modules.model import SlerpFace
        del SlerpFace
        checks['model_backend'] = {
            'available': bool(torch.isfinite(torch.ones(1) + 1).all()),
            'torch_version': torch.__version__,
            'source': 'pinned Tencent/TFace subset in third_party/tface',
        }
    except (ImportError, OSError, RuntimeError) as exc:
        checks['model_backend'] = {'available': False, 'error_type': type(exc).__name__}

    model_path = Path(checkpoint) if checkpoint else PROJECT_ROOT / 'ckpt/Backbone_Epoch_24_checkpoint.pth'
    checks['trained_checkpoint'] = {
        'available': False,
        'status': 'not_tested' if model_path.is_file() else 'missing',
    }
    # Loading is opt-in because model weights require independent provenance and
    # licensing review. A forward pass checks compatibility, never recognition.
    if checkpoint and checks['model_backend']['available']:
        try:
            from verify_with_antispoof import extract_face_features
            import numpy as np
            groups, vector = extract_face_features(
                np.zeros((112, 112, 3), dtype=np.uint8), model_path=model_path
            )
            valid = (groups.shape == (49, 16) and vector.shape == (512,) and
                     np.isfinite(groups).all() and np.isfinite(vector).all())
            checks['trained_checkpoint'] = {
                'available': bool(valid),
                'status': 'compatible_forward' if valid else 'invalid_outputs',
            }
        except (ImportError, OSError, RuntimeError):
            checks['trained_checkpoint'] = {'available': False, 'status': 'unavailable_or_incompatible'}

    required = ('numpy', 'scipy', 'Pillow', 'streamlit', 'matplotlib', 'opencv-python', 'face_detection')
    try:
        from modules.face_backend import face_backend_status
        status = face_backend_status()
        checks['pretrained_face'] = {'available': status['ready'], 'reason': status['reason'],
                                    'model_name': status['model_name'], 'models': status['models']}
    except (ImportError, OSError, RuntimeError) as exc:
        checks['pretrained_face'] = {'available': False, 'error_type': type(exc).__name__}
    try:
        from modules.presentation_backend import presentation_backend_status
        checks['presentation_model'] = presentation_backend_status()
    except (ImportError, OSError, RuntimeError):
        checks['presentation_model'] = {'available': False, 'reason': 'Optional learned presentation backend unavailable'}
    return {
        'python': sys.version.split()[0],
        'demo_runtime_ready': all(checks[name]['available'] for name in required),
        'prototype_runtime_ready': all(checks[name]['available'] for name in ('numpy', 'Pillow', 'streamlit', 'opencv-python', 'pretrained_face')),
        'authentication_enabled': False,
        'assurance': 'Capability checks only; no accuracy, liveness, privacy or licensing validation.',
        'checks': checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, help='Explicitly load a provenance-reviewed checkpoint and test one forward pass')
    parser.add_argument('--require-model', action='store_true', help='Fail unless --checkpoint completed a compatible trained forward pass')
    parser.add_argument('--require-prototype', action='store_true', help='Fail unless the local pretrained face prototype is ready')
    parser.add_argument('--require-presentation', action='store_true', help='Also require the optional learned presentation models')
    args = parser.parse_args(argv)
    report = inspect_runtime(args.checkpoint)
    print(json.dumps(report, indent=2))
    ready = report['demo_runtime_ready']
    if args.require_prototype:
        ready = report.get('prototype_runtime_ready', False)
    if args.require_presentation:
        presentation = report['checks'].get('presentation_model', {})
        ready = ready and presentation.get('available', presentation.get('ready', False))
    if args.require_model:
        ready = ready and report['checks']['trained_checkpoint']['available']
    return 0 if ready else 2


if __name__ == '__main__':
    raise SystemExit(main())
