"""
modules/antispoof/detector.py
==============================
Main Presentation Attack Detection (PAD) Detector.
Fuses Frequency Domain (FFT/Moiré) and Spatial Texture/Color Domain (LBP/YCrCb)
cues into a robust, real-time liveness verdict.
"""

from dataclasses import dataclass
from typing import Dict, Any, Union
import numpy as np
from PIL import Image

from .frequency_analysis import analyze_frequency
from .texture_analysis import analyze_texture


@dataclass
class AntiSpoofResult:
    """Detailed evaluation result from the AntiSpoofDetector."""
    is_live: bool
    liveness_score: float  # [0.0, 1.0], higher = more confident real human
    verdict: str           # "GENUINE (Live Face)", "SCREEN_REPLAY", "PRINT_ATTACK"
    risk_level: str        # "LOW", "MEDIUM", "CRITICAL"
    confidence: float      # [0.0, 1.0]
    diagnostics: Dict[str, Any]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_live": self.is_live,
            "liveness_score": round(self.liveness_score, 4),
            "verdict": self.verdict,
            "risk_level": self.risk_level,
            "confidence": round(self.confidence, 4),
            "summary_text": self.summary_text,
            "diagnostics": {
                k: round(v, 4) if isinstance(v, (float, np.floating)) else v
                for k, v in self.diagnostics.items()
            },
        }


class AntiSpoofDetector:
    """
    Two-tier Presentation Attack Detector:
    1. Frequency analysis: Screen pixel grids, moiré interference, high-frequency energy.
    2. Texture/Color analysis: LBP entropy, skin chrominance variance, specular reflection.
    """

    def __init__(
        self,
        threshold: float = 0.60,
        freq_weight: float = 0.50,
        texture_weight: float = 0.50,
    ):
        self.threshold = threshold
        self.freq_weight = freq_weight
        self.texture_weight = texture_weight

    def _prepare_image(self, img_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Converts filepath, PIL Image, or array to RGB NumPy uint8 array."""
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert("RGB")
            return np.array(pil_img, dtype=np.uint8)
        elif isinstance(img_input, Image.Image):
            return np.array(img_input.convert("RGB"), dtype=np.uint8)
        elif isinstance(img_input, np.ndarray):
            if img_input.dtype != np.uint8:
                if img_input.max() <= 1.0:
                    img_input = (img_input * 255).astype(np.uint8)
                else:
                    img_input = img_input.astype(np.uint8)
            if img_input.ndim == 2:
                img_input = np.stack([img_input] * 3, axis=2)
            elif img_input.ndim == 3 and img_input.shape[2] == 4:
                img_input = img_input[:, :, :3]
            return img_input
        else:
            raise ValueError(f"Unsupported image input type: {type(img_input)}")

    def evaluate(self, img_input: Union[str, np.ndarray, Image.Image]) -> AntiSpoofResult:
        """
        Evaluates an image for presentation attack indicators.
        Returns an AntiSpoofResult dataclass.
        """
        img_rgb = self._prepare_image(img_input)

        # 1. Extract frequency domain metrics
        freq_res = analyze_frequency(img_rgb)

        # 2. Extract spatial texture and color metrics
        tex_res = analyze_texture(img_rgb)

        # 3. Fuse scores
        base_score = (
            self.freq_weight * freq_res["freq_liveness_score"] +
            self.texture_weight * tex_res["texture_liveness_score"]
        )

        # Additional specific penalty heuristics
        penalty = 0.0
        # Critical screen trigger: strong periodic moire + specular glare
        if freq_res["moire_score"] > 0.70 and tex_res["glare_ratio"] > 0.02:
            penalty += 0.35

        # Critical print trigger: low texture entropy + restricted chrominance
        if tex_res["lbp_entropy"] < 6.2 and tex_res["chrominance_diversity"] < 7.0:
            penalty += 0.30

        final_liveness = float(np.clip(base_score - penalty, 0.0, 1.0))
        is_live = final_liveness >= self.threshold

        # Determine verdict and attack type
        if is_live:
            verdict = "GENUINE (Live Human Face)"
            risk_level = "LOW"
            summary = (
                f"Passed liveness check (Score: {final_liveness:.2f} >= {self.threshold:.2f}). "
                f"Natural 3D skin texture and regular frequency decay detected."
            )
        else:
            risk_level = "CRITICAL"
            # Diagnose primary spoof mechanism
            if freq_res["moire_score"] > 0.60 or tex_res["glare_ratio"] > 0.03:
                verdict = "SCREEN_REPLAY (Digital Monitor / Phone Screen)"
                summary = (
                    f"Spoof detected! Screen pixel grid interference / moiré pattern found "
                    f"(Moiré Score: {freq_res['moire_score']:.2f}, Glare: {tex_res['glare_ratio'] * 100:.1f}%)."
                )
            elif tex_res["lbp_entropy"] < 6.4 or freq_res["high_freq_ratio"] < 0.09:
                verdict = "PRINT_ATTACK (Printed Photo / Paper Replay)"
                summary = (
                    f"Spoof detected! Paper printing texture / low chrominance diversity found "
                    f"(LBP Entropy: {tex_res['lbp_entropy']:.2f}, High-Freq: {freq_res['high_freq_ratio']:.2f})."
                )
            else:
                verdict = "UNKNOWN_SPOOF (Synthetic or Mask Attack)"
                summary = (
                    f"Spoof detected! Abnormal facial micro-texture profile (Score: {final_liveness:.2f})."
                )

        confidence = float(abs(final_liveness - self.threshold) / max(self.threshold, 1.0 - self.threshold))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        diagnostics = {
            "liveness_threshold": self.threshold,
            "freq_liveness_score": freq_res["freq_liveness_score"],
            "texture_liveness_score": tex_res["texture_liveness_score"],
            "moire_score": freq_res["moire_score"],
            "high_freq_ratio": freq_res["high_freq_ratio"],
            "lbp_entropy": tex_res["lbp_entropy"],
            "chrominance_diversity": tex_res["chrominance_diversity"],
            "glare_ratio": tex_res["glare_ratio"],
        }

        return AntiSpoofResult(
            is_live=is_live,
            liveness_score=final_liveness,
            verdict=verdict,
            risk_level=risk_level,
            confidence=confidence,
            diagnostics=diagnostics,
            summary_text=summary,
        )
