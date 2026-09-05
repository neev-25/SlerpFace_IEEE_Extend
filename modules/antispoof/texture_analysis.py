"""
modules/antispoof/texture_analysis.py
======================================
Texture and Color Space Analysis for Face Presentation Attack Detection (PAD).

Physical Principle:
- Real human skin exhibits non-uniform micro-texture (pores, fine wrinkles, dermal translucency)
  and rich chrominance distribution in the YCrCb skin-color subspace.
- Paper prints suffer from halftone dot patterns, reduced chromatic variance, and paper fiber reflectance.
- Electronic screens exhibit flat glass specular reflection, unnatural RGB color gamut clipping,
  and LED backlight glare.
"""

import numpy as np


def compute_lbp(img_gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Computes standard 8-neighbor Local Binary Pattern (LBP) texture map.
    Fast vectorized implementation using NumPy slices.
    """
    h, w = img_gray.shape
    lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
    center = img_gray[radius:-radius, radius:-radius]

    # Offsets for 8 neighbors around center
    offsets = [
        (-radius, -radius, 0),
        (-radius, 0, 1),
        (-radius, radius, 2),
        (0, radius, 3),
        (radius, radius, 4),
        (radius, 0, 5),
        (radius, -radius, 6),
        (0, -radius, 7),
    ]

    for dy, dx, bit in offsets:
        neighbor = img_gray[radius + dy : h - radius + dy, radius + dx : w - radius + dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << bit)

    return lbp


def compute_lbp_entropy(lbp_map: np.ndarray) -> float:
    """
    Calculates the Shannon entropy of the LBP histogram.
    Natural human skin exhibits rich, balanced micro-texture entropy.
    Overly smooth masks or flat screens yield lower texture entropy.
    """
    hist, _ = np.histogram(lbp_map.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)


def rgb_to_ycrcb(img_rgb: np.ndarray) -> np.ndarray:
    """Converts normalized or uint8 RGB image to YCrCb color space."""
    img = img_rgb.astype(np.float32)
    # ITU-R BT.601 conversion matrix
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cr = (img[:, :, 0] - y) * 0.713 + 128.0
    cb = (img[:, :, 2] - y) * 0.564 + 128.0
    return np.stack([y, cr, cb], axis=2)


def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """Converts RGB image (0..255) to HSV color space."""
    img = img_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    v = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    delta = v - min_val

    # Saturation
    s = np.zeros_like(v)
    non_zero = v > 1e-6
    s[non_zero] = delta[non_zero] / v[non_zero]

    # Hue
    h = np.zeros_like(v)
    mask_r = (v == r) & (delta > 1e-6)
    mask_g = (v == g) & (delta > 1e-6)
    mask_b = (v == b) & (delta > 1e-6)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)
    h = h / 360.0

    return np.stack([h, s, v], axis=2)


def analyze_color_diversity(img_rgb: np.ndarray) -> dict:
    """
    Analyzes chrominance variance across YCrCb and HSV color spaces.
    Real skin has distinct Cb-Cr dispersion. Fake displays/prints show
    restricted or unnaturally biased chrominance distributions.
    """
    ycrcb = rgb_to_ycrcb(img_rgb)
    cr_std = float(np.std(ycrcb[:, :, 1]))
    cb_std = float(np.std(ycrcb[:, :, 2]))

    hsv = rgb_to_hsv(img_rgb)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))

    # Composite color diversity
    color_score = float((cr_std + cb_std) / 2.0)
    return {
        "cr_std": cr_std,
        "cb_std": cb_std,
        "sat_std": sat_std,
        "val_std": val_std,
        "chrominance_diversity": color_score,
    }


def detect_specular_hotspots(img_rgb: np.ndarray) -> float:
    """
    Detects planar screen glare and plastic reflection hotspots.
    Screens often produce intense, localized saturated patches with zero texture.
    Returns a glare_ratio in [0, 1].
    """
    gray = np.mean(img_rgb, axis=2)
    threshold = 245.0  # Near pure white glare
    saturated_pixels = np.sum(gray >= threshold)
    total_pixels = gray.size
    glare_ratio = float(saturated_pixels / (total_pixels + 1e-8))
    return glare_ratio


def analyze_texture(img_rgb: np.ndarray) -> dict:
    """
    Full texture and color-space analysis on an RGB face crop.
    
    Returns:
      {
        'lbp_entropy': float,
        'chrominance_diversity': float,
        'glare_ratio': float,
        'texture_liveness_score': float  # 0.0 (likely spoof) to 1.0 (likely genuine)
      }
    """
    gray = np.mean(img_rgb, axis=2).astype(np.uint8)
    lbp_map = compute_lbp(gray, radius=1)
    lbp_entropy = compute_lbp_entropy(lbp_map)

    color_metrics = analyze_color_diversity(img_rgb)
    glare_ratio = detect_specular_hotspots(img_rgb)

    # Scoring logic:
    # Genuine face: LBP entropy typically 6.8 - 7.9.
    # Flat prints/screen: LBP entropy < 6.0 or unnaturally clipped.
    # Chrominance std in Cr/Cb typically 8.0 - 24.0 for real skin.
    penalty = 0.0

    # LBP entropy calibration
    if lbp_entropy < 4.5:
        penalty += (4.5 - lbp_entropy) * 0.8
    elif lbp_entropy > 7.2:
        penalty += (lbp_entropy - 7.2) * 0.6

    # Chrominance diversity (washed out print has low diversity < 1.35)
    if color_metrics["chrominance_diversity"] < 1.35:
        penalty += (1.35 - color_metrics["chrominance_diversity"]) * 0.8

    # Glare hotspot penalty (glass reflection from phone/laptop screen)
    if glare_ratio > 0.015:
        penalty += glare_ratio * 15.0

    texture_liveness_score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    return {
        "lbp_entropy": lbp_entropy,
        "chrominance_diversity": color_metrics["chrominance_diversity"],
        "glare_ratio": glare_ratio,
        "texture_liveness_score": texture_liveness_score,
        **color_metrics,
    }
