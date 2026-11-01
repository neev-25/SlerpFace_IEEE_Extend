"""
modules/antispoof/visualizer.py
=================================
PAD Pipeline Stage Visualizer — generates PIL-compatible images for each
analysis stage (Sections 3.1–3.4) for Streamlit's st.image().
"""
import numpy as np
from PIL import Image
from .frequency_analysis import compute_fft_spectrum
from .texture_analysis import compute_lbp
from .chrominance_analysis import rgb_to_ycrcb
from .glare_detection import compute_glare_mask


def _apply_hot_colormap(gray_u8: np.ndarray) -> np.ndarray:
    v = gray_u8.astype(np.float32) / 255.0
    r = np.clip(v * 2.5, 0.0, 1.0)
    g = np.clip((v - 0.4) * 2.5, 0.0, 1.0)
    b = np.clip((v - 0.8) * 5.0, 0.0, 1.0)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


def _apply_cool_colormap(gray_u8: np.ndarray) -> np.ndarray:
    v = gray_u8.astype(np.float32) / 255.0
    return (np.stack([v, 1.0 - v, np.ones_like(v)], axis=2) * 255).astype(np.uint8)


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _resize(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    return np.array(Image.fromarray(img_rgb).resize((size, size), Image.NEAREST), dtype=np.uint8)


def gen_fft_spectrum_image(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Section 3.1 — 2D FFT log-power spectrum (hot colormap)."""
    gray     = (0.2989*img_rgb[:,:,0] + 0.5870*img_rgb[:,:,1] + 0.1140*img_rgb[:,:,2]).astype(np.float64)
    spectrum = compute_fft_spectrum(gray)
    return _resize(_apply_hot_colormap(_normalize_u8(spectrum)), size)


def gen_lbp_map_image(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Section 3.2 — LBP texture map (cool colormap)."""
    gray = np.mean(img_rgb, axis=2).astype(np.uint8)
    lbp  = compute_lbp(gray, radius=1)
    return _resize(_apply_cool_colormap(_normalize_u8(lbp.astype(np.float32))), size)


def gen_chrominance_heatmap(img_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Section 3.3 — Cr channel heatmap."""
    ycrcb = rgb_to_ycrcb(img_rgb)
    cr    = np.clip(ycrcb[:,:,1], 0, 255).astype(np.uint8)
    return _resize(_apply_hot_colormap(cr), size)


def gen_glare_overlay(img_rgb: np.ndarray, size: int = 224, threshold: int = 245) -> np.ndarray:
    """Section 3.4 — Glare region overlay (red on original)."""
    mask    = compute_glare_mask(img_rgb, threshold)
    overlay = img_rgb.copy()
    overlay[mask == 1, 0] = 255
    overlay[mask == 1, 1] = 30
    overlay[mask == 1, 2] = 30
    return _resize(overlay, size)


def generate_pipeline_visuals(img_rgb: np.ndarray, size: int = 224) -> dict:
    """Return dict of all 5 pipeline stage images as H×W×3 uint8 arrays."""
    return {
        "original":    _resize(img_rgb, size),
        "fft_spectrum": gen_fft_spectrum_image(img_rgb, size),
        "lbp_map":     gen_lbp_map_image(img_rgb, size),
        "chrominance": gen_chrominance_heatmap(img_rgb, size),
        "glare_mask":  gen_glare_overlay(img_rgb, size),
    }
