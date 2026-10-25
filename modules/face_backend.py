"""Offline CPU face detection, alignment and pretrained SFace embeddings.

This is an experimental recognition backend, not a liveness or authentication
decision. Model bytes are pinned to OpenCV Zoo Git LFS SHA-256 identifiers.
"""

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
from pathlib import Path
import threading

import numpy as np

from modules.image_input import load_rgb_image
from modules.pose_estimation import estimate_head_pose

MODEL_COMMIT = "47534e27c9851bb1128ccc0102f1145e27f23f98"
MODEL_NAME = "OpenCV SFace (2021dec) + YuNet (2023mar)"
MODEL_SPECS = {
    "yunet": {
        "filename": "face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "size": 232589,
        "directory": "face_detection_yunet",
        "license": "MIT",
    },
    "sface": {
        "filename": "face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "size": 38696353,
        "directory": "face_recognition_sface",
        "license": "Apache-2.0",
    },
}
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "opencv"
_LOCAL = threading.local()
MAX_DETECTION_SIDE = 640


class FaceBackendUnavailable(RuntimeError):
    """Trusted local model bytes or the required OpenCV API are unavailable."""


class FaceInputError(ValueError):
    """A recoverable capture problem with a public machine-readable code."""

    def __init__(self, message, code="INVALID_IMAGE"):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FaceFeatures:
    vector: np.ndarray = field(repr=False)
    groups: np.ndarray = field(repr=False)
    face_box: tuple
    landmarks: np.ndarray = field(repr=False)
    detection_score: float
    model_name: str
    aligned_rgb: np.ndarray = field(repr=False)
    quality: dict = field(default_factory=dict)


def _fingerprint(path):
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)


@lru_cache(maxsize=16)
def _verified_digest(fingerprint):
    path, size, *_ = fingerprint
    # A fixed expected size bounds reads even when a model path is user supplied.
    if size not in {spec["size"] for spec in MODEL_SPECS.values()}:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if _fingerprint(Path(path)) != fingerprint:
        return None
    return digest.hexdigest()


def _opencv():
    try:
        import cv2
    except ImportError as exc:
        raise FaceBackendUnavailable("Install the pinned project runtime with opencv-python.") from exc
    if not all(hasattr(cv2, name) for name in ("FaceDetectorYN", "FaceRecognizerSF", "dnn")):
        raise FaceBackendUnavailable("This OpenCV build lacks YuNet/SFace. Use the project's .venv runtime.")
    return cv2


def face_backend_status(model_dir=None):
    """Return public local readiness; never downloads or initializes a network."""
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    directory = directory.resolve()
    result = {
        "ready": False, "backend": "opencv_sface", "device": "CPU",
        "model_name": MODEL_NAME, "model_dir": str(directory), "models": {}, "reason": "",
    }
    try:
        cv = _opencv()
        result["opencv_version"] = cv.__version__
    except FaceBackendUnavailable as exc:
        result["reason"] = str(exc)
        return result
    try:
        for name, spec in MODEL_SPECS.items():
            path = directory / spec["filename"]
            item = {"filename": spec["filename"], "sha256": spec["sha256"],
                    "present": path.is_file(), "verified": False}
            if item["present"]:
                item["verified"] = _verified_digest(_fingerprint(path)) == spec["sha256"]
            result["models"][name] = item
    except OSError:
        result["reason"] = "Cannot read local face model files. Check file permissions."
        return result
    result["ready"] = all(item["verified"] for item in result["models"].values())
    result["reason"] = "Verified pretrained face models are ready locally." if result["ready"] else (
        "Face models are missing or failed integrity checks. Run .venv\\Scripts\\python.exe "
        "-B scripts/provision_face_models.py."
    )
    return result


def _get_backend(model_dir=None):
    cv = _opencv()
    status = face_backend_status(model_dir)
    if not status["ready"]:
        raise FaceBackendUnavailable(status["reason"])
    directory = Path(status["model_dir"])
    paths = [directory / spec["filename"] for spec in MODEL_SPECS.values()]
    key = tuple(_fingerprint(path) for path in paths)
    cached = getattr(_LOCAL, "backend", None)
    if cached is not None and cached[0] == key:
        return cv, cached[1], cached[2]
    try:
        detector = cv.FaceDetectorYN.create(
            str(paths[0]), "", (320, 320), 0.75, 0.3, 5000,
            cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU,
        )
        recognizer = cv.FaceRecognizerSF.create(
            str(paths[1]), "", cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU,
        )
        if tuple(_fingerprint(path) for path in paths) != key:
            raise FaceBackendUnavailable("Face models changed while loading; retry after provisioning completes.")
    except cv.error as exc:
        raise FaceBackendUnavailable("OpenCV could not load the verified models. Use the pinned project runtime.") from exc
    # One pair per thread: OpenCV nets have mutable input/scratch state.
    _LOCAL.backend = (key, detector, recognizer)
    return cv, detector, recognizer


def extract_pretrained_face(image, model_dir=None):
    """Extract exactly one aligned real face; no network or synthetic fallback.

    ``face_box`` is (x1, y1, x2, y2) in original image pixel coordinates.
    The 128-vector's eight 16-value groups are an experimental vector grouping,
    distinct from the official SlerpFace network's learned spatial grouping.
    """
    try:
        rgb = np.asarray(load_rgb_image(image)).copy()
    except (ValueError, OSError, TypeError) as exc:
        raise FaceInputError(str(exc), "INVALID_IMAGE") from exc
    cv, detector, recognizer = _get_backend(model_dir)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    scale = min(1.0, MAX_DETECTION_SIDE / max(width, height))
    detection_image = bgr if scale == 1 else cv.resize(
        bgr, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv.INTER_AREA,
    )
    actual_x = detection_image.shape[1] / width
    actual_y = detection_image.shape[0] / height
    try:
        detector.setInputSize((detection_image.shape[1], detection_image.shape[0]))
        _, faces = detector.detect(detection_image)
    except cv.error as exc:
        raise FaceBackendUnavailable("Face detection failed in OpenCV; check the pinned runtime and model files.") from exc
    if faces is None or len(faces) == 0:
        raise FaceInputError("No face found. Face the camera with even lighting and move closer.", "NO_FACE")
    if len(faces) != 1:
        raise FaceInputError("Multiple faces found. Keep exactly one person in each image.", "MULTIPLE_FACES")
    row = np.asarray(faces[0], dtype=np.float32).copy()
    if row.shape != (15,) or not np.isfinite(row).all():
        raise FaceBackendUnavailable("The face detector returned invalid landmarks.")
    row[[0, 2, 4, 6, 8, 10, 12]] /= actual_x
    row[[1, 3, 5, 7, 9, 11, 13]] /= actual_y
    landmarks = row[4:14].reshape(5, 2).copy()
    eye_distance = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    if min(row[2], row[3]) < 40 or eye_distance < 10:
        raise FaceInputError("The face is too small for a useful comparison. Move closer to the camera.", "LOW_QUALITY")
    x, y, face_width, face_height = map(float, row[:4])
    box = (max(0.0, x), max(0.0, y), min(float(width), x + face_width), min(float(height), y + face_height))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise FaceInputError("The detected face is outside the image. Retake the capture.", "LOW_QUALITY")
    try:
        aligned_bgr = recognizer.alignCrop(bgr, row)
        vector = np.asarray(recognizer.feature(aligned_bgr), dtype=np.float64).reshape(-1).copy()
    except cv.error as exc:
        raise FaceInputError("Face alignment failed. Retake a front-facing capture.", "LOW_QUALITY") from exc
    if vector.shape != (128,) or not np.isfinite(vector).all():
        raise FaceBackendUnavailable("The recognizer returned invalid face features.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise FaceBackendUnavailable("The recognizer returned empty face features.")
    vector /= norm
    aligned_rgb = cv.cvtColor(aligned_bgr, cv.COLOR_BGR2RGB).copy()
    if aligned_rgb.shape != (112, 112, 3):
        raise FaceBackendUnavailable("The recognizer returned an unexpected aligned image size.")
    pose = estimate_head_pose(landmarks)
    warnings = [] if pose.is_acceptable else [pose.guidance]
    return FaceFeatures(
        vector=vector, groups=vector.reshape(-1, 16).copy(), face_box=box,
        landmarks=landmarks, detection_score=float(row[14]), model_name=MODEL_NAME,
        aligned_rgb=aligned_rgb,
        quality={"face_width_pixels": face_width, "face_height_pixels": face_height,
                 "eye_distance_pixels": eye_distance, "detection_score": float(row[14]),
                 "pose": pose.to_dict(), "warnings": warnings},
    )
