"""Bounded static-image PAD heuristics, not a validated liveness detector."""

from dataclasses import asdict, dataclass
import threading

import numpy as np
from PIL import Image

from modules.image_input import load_rgb_image
from .frequency_analysis import analyze_frequency
from .texture_analysis import analyze_texture

_thread_state = threading.local()
ANALYSIS_SIZE = (112, 112)


class FaceDetectionUnavailable(RuntimeError):
    """The face-presence dependency is unavailable; never silently bypass it."""


def _detect_faces(image):
    try:
        import cv2
    except ImportError as exc:
        raise FaceDetectionUnavailable("Face presence checks need opencv-python; synthetic mode must be explicitly selected.") from exc
    if not all(hasattr(cv2, name) for name in ("CascadeClassifier", "data", "cvtColor", "COLOR_RGB2GRAY")):
        raise FaceDetectionUnavailable("OpenCV face-presence support is incomplete; install a compatible full opencv-python build.")
    if not hasattr(cv2.data, "haarcascades"):
        raise FaceDetectionUnavailable("OpenCV's face-presence cascade data is missing.")
    if not hasattr(_thread_state, "cascade"):
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            raise FaceDetectionUnavailable("OpenCV's face-presence cascade is missing.")
        _thread_state.cascade = cascade
    preview = image.copy()
    preview.thumbnail((640, 640), Image.Resampling.BILINEAR)
    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    boxes = _thread_state.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    sx, sy = image.width / preview.width, image.height / preview.height
    return [(int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy)) for x, y, w, h in boxes]


@dataclass
class AntiSpoofResult:
    # Compatibility field: only means a heuristic pass, never verified liveness.
    is_live: bool
    liveness_score: float
    verdict: str
    risk_level: str
    confidence: float
    diagnostics: dict
    summary_text: str
    authentication_eligible: bool = False

    def to_dict(self):
        result = asdict(self)
        result["heuristic_passed"] = self.is_live
        result["assurance"] = "Unvalidated static-image heuristic; not proof of liveness."
        return result


class AntiSpoofDetector:
    def __init__(self, threshold=0.60, freq_weight=0.50, texture_weight=0.50,
                 *, require_face=True):
        values = (threshold, freq_weight, texture_weight)
        if not all(np.isfinite(v) for v in values) or not 0 < threshold < 1:
            raise ValueError("Heuristic threshold must be finite and strictly between 0 and 1.")
        if min(freq_weight, texture_weight) < 0 or not np.isclose(freq_weight + texture_weight, 1):
            raise ValueError("Frequency and texture weights must be nonnegative and sum to one.")
        self.threshold = float(threshold)
        self.freq_weight, self.texture_weight = float(freq_weight), float(texture_weight)
        self.require_face = require_face

    def _prepare_image(self, img_input):
        return np.asarray(load_rgb_image(img_input, size=ANALYSIS_SIZE), dtype=np.uint8)

    def evaluate(self, img_input):
        image = load_rgb_image(img_input)
        diagnostics = {"liveness_threshold": self.threshold, "analysis_width": 112,
                       "analysis_height": 112, "face_check_enabled": self.require_face,
                       "freq_liveness_score": 0.0, "texture_liveness_score": 0.0,
                       "moire_score": 0.0, "high_freq_ratio": 0.0, "lbp_entropy": 0.0,
                       "chrominance_diversity": 0.0, "glare_ratio": 0.0}
        if self.require_face:
            boxes = _detect_faces(image)
            diagnostics["face_count"] = len(boxes)
            if len(boxes) != 1:
                return AntiSpoofResult(False, 0.0, "NO_FACE" if not boxes else "MULTIPLE_FACES",
                                       "REJECTED", 0.0, diagnostics,
                                       "Exactly one detectable face is required; extraction has not run.")
            image = image.crop(boxes[0])
        rgb = np.asarray(image.resize(ANALYSIS_SIZE, Image.Resampling.BILINEAR))
        freq, texture = analyze_frequency(rgb), analyze_texture(rgb)
        base = self.freq_weight * freq["freq_liveness_score"] + self.texture_weight * texture["texture_liveness_score"]
        penalty = 0.0
        if freq["moire_score"] > 0.70 and texture["glare_ratio"] > 0.02:
            penalty += 0.35
        if texture["lbp_entropy"] < 6.2 and texture["chrominance_diversity"] < 7:
            penalty += 0.30
        score = float(np.clip(base - penalty, 0, 1))
        passed = score >= self.threshold
        verdict = "HEURISTIC_PASS" if passed else "HEURISTIC_REJECT"
        for key in diagnostics:
            if key in freq:
                diagnostics[key] = float(freq[key])
            elif key in texture:
                diagnostics[key] = float(texture[key])
        return AntiSpoofResult(passed, score, verdict, "UNVERIFIED" if passed else "REJECTED",
                               0.0, diagnostics,
                               f"Static-image heuristic score {score:.3f}; liveness and identity remain unverified.")
