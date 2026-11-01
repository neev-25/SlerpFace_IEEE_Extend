"""
modules/antispoof/__init__.py
==============================
Presentation Attack Detection (Anti-Spoofing) Subsystem for GuardFace.

New modules use graceful imports so optional dependencies (e.g. opencv-python)
do not break the package for systems where they are not installed.
"""

# ── Core (always available, no optional deps) ─────────────────────────────────
from .detector import AntiSpoofDetector, AntiSpoofResult
from .frequency_analysis import analyze_frequency, compute_fft_spectrum
from .texture_analysis import analyze_texture
from .chrominance_analysis import analyze_chrominance
from .glare_detection import analyze_specular_glare, compute_glare_mask
from .visualizer import generate_pipeline_visuals
from .metrics import evaluate_pad_system, PADMetrics, compute_apcer, compute_bpcer, compute_acer
from .deep_pad import DeepPAD, DeepPADResult

# ── Optional: face_alignment requires opencv-python-headless ─────────────────
try:
    from .face_alignment import preprocess_for_pad, detect_face_roi, draw_face_bbox
    _FACE_ALIGNMENT_AVAILABLE = True
except ImportError:
    _FACE_ALIGNMENT_AVAILABLE = False
    # Provide graceful no-op stubs so imports in app_demo.py don't crash
    def preprocess_for_pad(img_rgb, target_size=112):
        """Stub: install opencv-python-headless for face alignment."""
        from PIL import Image
        import numpy as np
        pil = Image.fromarray(img_rgb).resize((target_size, target_size))
        return {"face_crop": np.array(pil, dtype="uint8"), "face_detected": False, "bbox": None, "original_size": img_rgb.shape[:2]}

    def detect_face_roi(img_rgb, target_size=112):
        from PIL import Image
        import numpy as np
        pil = Image.fromarray(img_rgb).resize((target_size, target_size))
        return np.array(pil, dtype="uint8"), None

    def draw_face_bbox(img_rgb, bbox, color=(0, 220, 80), thickness=3):
        return img_rgb.copy()


__all__ = [
    # Core detector
    "AntiSpoofDetector",
    "AntiSpoofResult",
    # Frequency domain (Section 3.1)
    "analyze_frequency",
    "compute_fft_spectrum",
    # Texture domain (Section 3.2)
    "analyze_texture",
    # Chrominance (Section 3.3)
    "analyze_chrominance",
    # Glare detection (Section 3.4)
    "analyze_specular_glare",
    "compute_glare_mask",
    # Face alignment (Section 2) — stubs if opencv not installed
    "preprocess_for_pad",
    "detect_face_roi",
    "draw_face_bbox",
    # Visualization
    "generate_pipeline_visuals",
    # Metrics (ISO/IEC 30107-3)
    "evaluate_pad_system",
    "PADMetrics",
    "compute_apcer",
    "compute_bpcer",
    "compute_acer",
    # Deep Learning PAD (Section 4)
    "DeepPAD",
    "DeepPADResult",
]
