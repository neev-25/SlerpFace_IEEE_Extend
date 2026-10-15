"""Optional pretrained single-frame presentation estimates; never authentication.

Only the pinned official MiniVision checkpoints are accepted. Inference never
downloads anything. ``face_box`` uses original-image (x1, y1, x2, y2) coordinates,
with the upper bounds exclusive, matching ``FaceFeatures.face_box``.
"""

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import threading

import numpy as np

from modules.image_input import load_rgb_image

SOURCE_COMMIT = "b6d5f04ad78778917853b25c778acef6d5626d15"
SOURCE_BASE = (
    "https://raw.githubusercontent.com/minivision-ai/"
    f"Silent-Face-Anti-Spoofing/{SOURCE_COMMIT}"
)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "ckpt" / "presentation"
MODEL_NAME = "MiniVision MiniFASNetV2 + MiniFASNetV1SE (static RGB)"
ARCHITECTURE_SHA256 = "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7"
CPU_INTRAOP_THREAD_LIMIT = 2
_CPU_RUNTIME_LOCK = threading.Lock()


@dataclass(frozen=True)
class PresentationModelSpec:
    filename: str
    architecture: str
    scale: float
    size: int
    sha256: str

    @property
    def url(self):
        return f"{SOURCE_BASE}/resources/anti_spoof_models/{self.filename}"


MODEL_SPECS = (
    PresentationModelSpec(
        "2.7_80x80_MiniFASNetV2.pth", "MiniFASNetV2", 2.7, 1_849_453,
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    ),
    PresentationModelSpec(
        "4_0_0_80x80_MiniFASNetV1SE.pth", "MiniFASNetV1SE", 4.0, 1_856_130,
        "84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990",
    ),
)


@dataclass(frozen=True)
class PresentationEstimate:
    status: str
    bona_fide_score: float | None = None
    model_name: str = MODEL_NAME
    reason: str = ""
    class_scores: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        """Public JSON-compatible metadata; contains no image or face features."""
        return asdict(self)


def verify_model_bytes(data, spec):
    if len(data) != spec.size or hashlib.sha256(data).hexdigest() != spec.sha256:
        raise ValueError(f"Integrity check failed for {spec.filename}.")
    return data


def _read_verified_model(path, spec):
    # Check and consume the same file descriptor. Bound reads before torch.load.
    with path.open("rb") as stream:
        data = stream.read(spec.size + 1)
    return verify_model_bytes(data, spec)


def _verify_architecture():
    source = Path(__file__).resolve().parents[1] / "third_party" / "silent_face" / "MiniFASNet.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != ARCHITECTURE_SHA256:
        raise ValueError("Pinned MiniFASNet architecture integrity check failed.")


def _configure_cpu_runtime(torch):
    # Tiny depthwise 80x80 networks perform poorly when each operation recruits
    # many CPU threads. This is deliberately process-wide: changing/restoring
    # PyTorch's thread pool around concurrent requests would create races.
    with _CPU_RUNTIME_LOCK:
        if torch.get_num_threads() > CPU_INTRAOP_THREAD_LIMIT:
            torch.set_num_threads(CPU_INTRAOP_THREAD_LIMIT)


def presentation_backend_status(model_dir=None):
    """Inspect installed model integrity and imports without fabricated inference."""
    status = {
        "available": False,
        "ready": False,
        "model_name": MODEL_NAME,
        "source_commit": SOURCE_COMMIT,
        "checkpoint_sha256": [spec.sha256 for spec in MODEL_SPECS],
        "authentication_enabled": False,
    }
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    try:
        for spec in MODEL_SPECS:
            _read_verified_model(directory / spec.filename, spec)
        _verify_architecture()
        import torch
        from third_party.silent_face import MiniFASNet

        if not callable(torch.inference_mode) or not callable(MiniFASNet.MiniFASNetV2):
            raise ValueError("Required PyTorch/model APIs are unavailable.")
    except (OSError, ValueError, ImportError, AttributeError) as exc:
        status["reason"] = f"Optional presentation backend unavailable: {exc}"
        return status
    status.update(available=True, ready=True, reason="Both pinned checkpoints and CPU inference imports are available.")
    return status


def _validate_box(face_box, image_size):
    try:
        box = np.asarray(face_box, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Face box must contain four finite coordinates.") from exc
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("Face box must contain four finite coordinates.")
    x1, y1, x2, y2 = box
    width, height = image_size
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise ValueError("Face box must be inside the original image.")
    if min(x2 - x1, y2 - y1) < 32:
        raise ValueError("Face is too small for the presentation estimate (minimum 32 pixels).")
    return box


def _context_crop(bgr, box, scale):
    """Original MiniVision context crop/boundary behavior, resized to 80x80.

    Adapted from pinned src/generate_patches.py; Apache-2.0, Minivision 2020.
    Input uses exclusive xyxy bounds; upstream uses xywh and inclusive crop ends.
    """
    import cv2

    src_h, src_w = bgr.shape[:2]
    x, y, x2, y2 = box
    box_w, box_h = x2 - x, y2 - y
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_width, new_height = box_w * scale, box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left, top = center_x - new_width / 2, center_y - new_height / 2
    right, bottom = center_x + new_width / 2, center_y + new_height / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    crop = bgr[int(top):int(bottom) + 1, int(left):int(right) + 1]
    if crop.size == 0:
        raise ValueError("Face context crop is empty.")
    return cv2.resize(crop, (80, 80), interpolation=cv2.INTER_LINEAR)


@lru_cache(maxsize=2)
def _load_model(spec, payload):
    """Cache by verified immutable bytes, so replacements cannot bypass hashes."""
    import torch

    _configure_cpu_runtime(torch)

    _verify_architecture()
    from third_party.silent_face import MiniFASNet

    constructor = getattr(MiniFASNet, spec.architecture)
    model = constructor(conv6_kernel=(5, 5))
    # Old upstream loader used unrestricted pickle. This adapter never does.
    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Presentation checkpoint must contain a tensor state dictionary.")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval(), threading.Lock()


def _predict(spec, payload, crop):
    import torch

    _configure_cpu_runtime(torch)
    model, lock = _load_model(spec, payload)
    # Upstream uses OpenCV BGR bytes, transposes to CHW float32 without /255.
    tensor = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))).float().unsqueeze(0)
    with lock, torch.inference_mode():
        logits = model(tensor)
        if tuple(logits.shape) != (1, 3) or not torch.isfinite(logits).all():
            raise ValueError("Presentation model returned invalid class logits.")
        return torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float64)


def evaluate_presentation(image, face_box, model_dir=None):
    """Return an advisory learned estimate, without a pass/fail threshold.

    The face box must come from a separate detector; this function neither finds
    faces nor proves live capture. Missing/incompatible models never silently
    fall back to heuristic or synthetic results.
    """
    try:
        rgb = load_rgb_image(image)
        box = _validate_box(face_box, rgb.size)
    except (ValueError, OSError, TypeError) as exc:
        return PresentationEstimate("INCONCLUSIVE", reason=str(exc))
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    try:
        # Verify both models before using either; no one-model fallback.
        payloads = [_read_verified_model(directory / spec.filename, spec) for spec in MODEL_SPECS]
        bgr = np.asarray(rgb)[:, :, ::-1]
        predictions = [
            _predict(spec, payload, _context_crop(bgr, box, spec.scale))
            for spec, payload in zip(MODEL_SPECS, payloads, strict=True)
        ]
        scores = np.mean(predictions, axis=0)
        if scores.shape != (3,) or not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("Presentation model returned invalid probabilities.")
        if not np.isclose(scores.sum(), 1.0, atol=1e-5):
            raise ValueError("Presentation probabilities do not sum to one.")
    except FileNotFoundError:
        return PresentationEstimate(
            "UNAVAILABLE", reason="Optional presentation models are missing. Run scripts/provision_pad_model.py.",
        )
    except (ImportError, OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        return PresentationEstimate("UNAVAILABLE", reason=f"Presentation backend unavailable: {exc}")
    return PresentationEstimate(
        "MODEL_ESTIMATE",
        bona_fide_score=float(scores[1]),
        reason="Single-frame model estimate only; live capture and identity are not verified.",
        class_scores={
            "attack_class_0": float(scores[0]),
            "bona_fide": float(scores[1]),
            "attack_class_2": float(scores[2]),
        },
        metadata={
            "source_commit": SOURCE_COMMIT,
            "fusion": "equal mean of two softmax vectors",
            "input": "BGR float32, byte scale [0,255], 80x80 context crops",
            "context_scales": [spec.scale for spec in MODEL_SPECS],
            "checkpoint_sha256": [spec.sha256 for spec in MODEL_SPECS],
            "cpu_intraop_thread_limit": CPU_INTRAOP_THREAD_LIMIT,
            "capture_provenance_verified": False,
            "calibrated_for_this_camera": False,
            "authentication_enabled": False,
        },
    )
