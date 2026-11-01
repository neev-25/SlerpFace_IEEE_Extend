"""
modules/antispoof/glare_detection.py
======================================
Standalone Section 3.4 — Specular Glare / Hotspot Detection
B(x, y) = 1 if I(x, y) > T, else 0
"""
import numpy as np
from scipy.ndimage import label as nd_label


def compute_glare_mask(img_rgb: np.ndarray, threshold: int = 245) -> np.ndarray:
    """Binary specular glare mask B(x,y)."""
    luminance = np.mean(img_rgb.astype(np.float32), axis=2)
    return (luminance >= threshold).astype(np.uint8)


def analyze_glare_regions(glare_mask: np.ndarray) -> dict:
    """Connected-component analysis on glare mask."""
    total_pixels = glare_mask.size
    glare_pixels  = int(np.sum(glare_mask))
    glare_ratio   = float(glare_pixels / (total_pixels + 1e-8))
    labeled, num_features = nd_label(glare_mask)
    max_area = 0
    if num_features > 0:
        areas    = [int(np.sum(labeled == i)) for i in range(1, num_features + 1)]
        max_area = max(areas)
    compactness = float(max_area / (glare_pixels + 1e-8)) if glare_pixels > 0 else 0.0
    return {
        "glare_ratio":         glare_ratio,
        "num_hotspots":        int(num_features),
        "max_hotspot_area":    max_area,
        "hotspot_compactness": compactness,
    }


def compute_channel_saturation(img_rgb: np.ndarray, threshold: int = 250) -> dict:
    """Per-channel saturation ratios."""
    n = float(img_rgb[:, :, 0].size)
    r_sat = float(np.sum(img_rgb[:, :, 0] >= threshold)) / n
    g_sat = float(np.sum(img_rgb[:, :, 1] >= threshold)) / n
    b_sat = float(np.sum(img_rgb[:, :, 2] >= threshold)) / n
    any_sat = float(np.sum(
        (img_rgb[:, :, 0] >= threshold) |
        (img_rgb[:, :, 1] >= threshold) |
        (img_rgb[:, :, 2] >= threshold)
    )) / n
    return {"r_sat_ratio": r_sat, "g_sat_ratio": g_sat, "b_sat_ratio": b_sat, "any_channel_sat_ratio": any_sat}


def analyze_specular_glare(img_rgb: np.ndarray, threshold: int = 245) -> dict:
    """Full standalone glare analysis (Section 3.4)."""
    glare_mask  = compute_glare_mask(img_rgb, threshold)
    glare_stats = analyze_glare_regions(glare_mask)
    ch_stats    = compute_channel_saturation(img_rgb)
    penalty = 0.0
    if glare_stats["glare_ratio"] > 0.015:
        penalty += glare_stats["glare_ratio"] * 14.0
    if glare_stats["hotspot_compactness"] > 0.65 and glare_stats["glare_ratio"] > 0.005:
        penalty += (glare_stats["hotspot_compactness"] - 0.65) * 0.8
    b_bias = ch_stats["b_sat_ratio"] - ch_stats["r_sat_ratio"]
    if b_bias > 0.01:
        penalty += b_bias * 3.0
    return {
        **glare_stats,
        **ch_stats,
        "glare_liveness_score": float(np.clip(1.0 - penalty, 0.0, 1.0)),
        "glare_mask": glare_mask,
    }
