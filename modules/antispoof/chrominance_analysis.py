"""
modules/antispoof/chrominance_analysis.py
==========================================
Standalone Section 3.3 — Chrominance Analysis (YCrCb / HSV)
Physical Principle: Real skin has rich Cr/Cb variance. Prints are desaturated.
Key formula: sigma_Cr = sqrt( (1/N) * sum( (Cr_i - mu_Cr)^2 ) )
"""
import numpy as np


def rgb_to_ycrcb(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to YCrCb using ITU-R BT.601."""
    img = img_rgb.astype(np.float32)
    y  =  0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cr = (img[:, :, 0] - y) * 0.713 + 128.0
    cb = (img[:, :, 2] - y) * 0.564 + 128.0
    return np.stack([y, cr, cb], axis=2)


def rgb_to_hsv(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB [0,255] to HSV [0,1]."""
    img = img_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    v     = np.max(img, axis=2)
    s_min = np.min(img, axis=2)
    delta = v - s_min
    s = np.where(v > 1e-6, delta / (v + 1e-8), 0.0)
    h = np.zeros_like(v)
    mr = (v == r) & (delta > 1e-6)
    mg = (v == g) & (delta > 1e-6)
    mb = (v == b) & (delta > 1e-6)
    h[mr] = 60.0 * (((g[mr] - b[mr]) / (delta[mr] + 1e-8)) % 6)
    h[mg] = 60.0 * (((b[mg] - r[mg]) / (delta[mg] + 1e-8)) + 2)
    h[mb] = 60.0 * (((r[mb] - g[mb]) / (delta[mb] + 1e-8)) + 4)
    return np.stack([np.clip(h / 360.0, 0.0, 1.0), s, v], axis=2)


def skin_tone_coverage(img_rgb: np.ndarray) -> float:
    """Fraction of pixels within YCrCb skin-tone ellipse (Kovac et al.)."""
    ycrcb = rgb_to_ycrcb(img_rgb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return float(np.sum(mask) / (mask.size + 1e-8))


def analyze_chrominance(img_rgb: np.ndarray) -> dict:
    """
    Full standalone chrominance analysis (Section 3.3).
    Returns sigma_Cr, skin_tone_fraction, chrominance_liveness_score, etc.
    """
    ycrcb   = rgb_to_ycrcb(img_rgb)
    cr_std  = float(np.std(ycrcb[:, :, 1]))
    cb_std  = float(np.std(ycrcb[:, :, 2]))
    hsv     = rgb_to_hsv(img_rgb)
    skin_frac = skin_tone_coverage(img_rgb)
    diversity = (cr_std + cb_std) / 2.0

    penalty = 0.0
    if diversity < 4.0:
        penalty += (4.0 - diversity) * 0.12
    if skin_frac < 0.15:
        penalty += (0.15 - skin_frac) * 2.5
    if float(np.std(hsv[:, :, 0])) > 0.28 and float(np.std(hsv[:, :, 1])) < 0.05:
        penalty += 0.25

    return {
        "y_mean":   float(np.mean(ycrcb[:, :, 0])),
        "y_std":    float(np.std(ycrcb[:, :, 0])),
        "cr_mean":  float(np.mean(ycrcb[:, :, 1])),
        "cr_std":   cr_std,
        "cb_mean":  float(np.mean(ycrcb[:, :, 2])),
        "cb_std":   cb_std,
        "cr_cb_ratio": float(cr_std / (cb_std + 1e-8)),
        "hue_std":  float(np.std(hsv[:, :, 0])),
        "sat_mean": float(np.mean(hsv[:, :, 1])),
        "sat_std":  float(np.std(hsv[:, :, 1])),
        "val_mean": float(np.mean(hsv[:, :, 2])),
        "val_std":  float(np.std(hsv[:, :, 2])),
        "skin_tone_fraction":        skin_frac,
        "chrominance_diversity":     diversity,
        "chrominance_liveness_score": float(np.clip(1.0 - penalty, 0.0, 1.0)),
    }
