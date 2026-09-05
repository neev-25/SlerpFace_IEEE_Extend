"""
modules/antispoof/frequency_analysis.py
========================================
Fourier Frequency Analysis for Face Presentation Attack Detection (PAD).

Physical Principle:
- Genuine human faces have smooth 3D curved surfaces producing continuous, natural
  spatial frequency decay.
- Digital screens (phone/laptop/tablet) produce strong periodic high-frequency spikes
  due to discrete pixel matrices, refresh artifacts, and moiré interference.
- Printed photos (paper/cardboard) suffer from halftone printing dot grids or high-frequency
  attenuation (blurring of fine epidermal micro-details).
"""

import numpy as np
from scipy.ndimage import uniform_filter


def compute_fft_spectrum(img_gray: np.ndarray) -> np.ndarray:
    """
    Computes the 2D Fast Fourier Transform and returns the centered
    logarithmic power magnitude spectrum.
    """
    h, w = img_gray.shape
    # Apply Hann window to reduce boundary discontinuity leakage
    hann_2d = np.outer(np.hanning(h), np.hanning(w))
    windowed = (img_gray.astype(np.float64) - np.mean(img_gray)) * hann_2d

    f_transform = np.fft.fft2(windowed)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    power_spectrum = np.log1p(magnitude)
    return power_spectrum


def compute_radial_profile(power_spectrum: np.ndarray, num_bins: int = 50) -> np.ndarray:
    """
    Calculates the azimuthally averaged radial profile of the power spectrum
    (energy distribution from center low frequencies to perimeter high frequencies).
    """
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = min(cy, cx)

    # Bin the radial distances
    bin_edges = np.linspace(0, max_r, num_bins + 1)
    radial_profile = np.zeros(num_bins, dtype=np.float64)

    for i in range(num_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_profile[i] = np.mean(power_spectrum[mask])
        else:
            radial_profile[i] = 0.0

    return radial_profile


def detect_moire_patterns(power_spectrum: np.ndarray) -> float:
    """
    Detects periodic screen moiré patterns by finding sharp, isolated spectral peaks
    in the mid-to-high frequency band of the Fourier domain.
    Returns a moire_score (higher = more likely digital screen).
    """
    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2

    # Mask out the DC component (central low frequencies)
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    low_freq_mask = dist_from_center < (min(h, w) * 0.12)
    high_freq_bound = dist_from_center > (min(h, w) * 0.48)

    spectrum_roi = power_spectrum.copy()
    spectrum_roi[low_freq_mask] = 0
    spectrum_roi[high_freq_bound] = 0

    if np.max(spectrum_roi) <= 0:
        return 0.0

    # Local average background subtraction
    local_bg = uniform_filter(spectrum_roi, size=9)
    peak_prominence = spectrum_roi - local_bg
    peak_prominence = np.clip(peak_prominence, 0, None)

    # Calculate the ratio of top 1% peak energy to median energy
    threshold = np.percentile(peak_prominence[peak_prominence > 0], 98) if np.any(peak_prominence > 0) else 0
    strong_peaks = peak_prominence[peak_prominence > threshold]

    if len(strong_peaks) == 0:
        return 0.0

    peak_energy = float(np.mean(strong_peaks))
    # Calibrate sigmoid around 2.50: genuine (<2.2) gives <0.25, print/screen (>2.6) gives >0.60
    moire_score = float(1.0 / (1.0 + np.exp(-3.5 * (peak_energy - 2.50))))
    return moire_score


def analyze_frequency(img_rgb: np.ndarray) -> dict:
    """
    Full Fourier frequency analysis on an RGB face crop.
    
    Returns:
      {
        'high_freq_ratio': float,
        'radial_decay_slope': float,
        'moire_score': float,
        'freq_liveness_score': float  # 0.0 (likely spoof) to 1.0 (likely genuine)
      }
    """
    # Convert to grayscale
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        img_gray = (0.2989 * img_rgb[:, :, 0] +
                    0.5870 * img_rgb[:, :, 1] +
                    0.1140 * img_rgb[:, :, 2])
    else:
        img_gray = img_rgb.astype(np.float64)

    power_spectrum = compute_fft_spectrum(img_gray)
    radial_prof = compute_radial_profile(power_spectrum, num_bins=40)

    # 1. High frequency energy ratio
    total_energy = np.sum(radial_prof) + 1e-8
    low_band = np.sum(radial_prof[:10])
    mid_band = np.sum(radial_prof[10:28])
    high_band = np.sum(radial_prof[28:])

    high_freq_ratio = float(high_band / total_energy)
    mid_freq_ratio = float(mid_band / total_energy)

    # 2. Radial decay slope
    x = np.arange(len(radial_prof))
    if len(radial_prof) > 1:
        slope, _ = np.polyfit(x, radial_prof, 1)
    else:
        slope = 0.0

    # 3. Moiré interference detection
    moire_score = detect_moire_patterns(power_spectrum)

    # 4. Composite Frequency Liveness Score
    # Genuine face: low moire_score (< 0.35) and natural high_freq_ratio (0.15 - 0.28)
    penalty = 0.0
    if moire_score > 0.45:
        penalty += (moire_score - 0.45) * 1.6

    if high_freq_ratio < 0.12:  # blurred photo print
        penalty += (0.12 - high_freq_ratio) * 3.0
    elif high_freq_ratio > 0.38:  # harsh digital noise
        penalty += (high_freq_ratio - 0.38) * 2.5

    freq_liveness_score = float(np.clip(1.0 - penalty, 0.0, 1.0))

    return {
        "high_freq_ratio": high_freq_ratio,
        "mid_freq_ratio": mid_freq_ratio,
        "decay_slope": float(slope),
        "moire_score": moire_score,
        "freq_liveness_score": freq_liveness_score
    }
