"""
modules/antispoof/__init__.py
==============================
Presentation Attack Detection (Anti-Spoofing) Subsystem for SlerpFace.
"""

from .detector import AntiSpoofDetector, AntiSpoofResult
from .frequency_analysis import analyze_frequency, compute_fft_spectrum
from .texture_analysis import analyze_texture

__all__ = [
    "AntiSpoofDetector",
    "AntiSpoofResult",
    "analyze_frequency",
    "compute_fft_spectrum",
    "analyze_texture",
]
