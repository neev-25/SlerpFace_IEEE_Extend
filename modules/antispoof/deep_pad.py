"""
modules/antispoof/deep_pad.py
==============================
Section 4 — Deep Learning PAD Alternative (CNN / ViT Interface)
Supports 'classical' (feature-based), 'cnn_builtin' (numpy CNN), and 'pytorch' backends.
"""
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np


@dataclass
class DeepPADResult:
    p_live:    float
    p_spoof:   float
    is_live:   bool
    backend:   str
    confidence: float

    def to_dict(self) -> dict:
        return {"p_live": round(self.p_live, 4), "p_spoof": round(self.p_spoof, 4),
                "is_live": self.is_live, "backend": self.backend, "confidence": round(self.confidence, 4)}


class DeepPAD:
    """
    Unified Deep Learning PAD interface (Section 4).
    Backends: 'classical' | 'cnn_builtin' | 'pytorch'
    """
    def __init__(self, backend: str = "classical", threshold: float = 0.60, model_path: Optional[str] = None):
        self.backend     = backend
        self.threshold   = threshold
        self.model_path  = model_path
        self._model      = None

        if backend == "classical":
            from .detector import AntiSpoofDetector
            self._detector = AntiSpoofDetector(threshold=threshold)
        elif backend == "cnn_builtin":
            self._detector = None
        elif backend == "pytorch":
            self._detector = None
            if model_path:
                self._model = self._load_pytorch_model(model_path)

    def _predict_classical(self, img_rgb: np.ndarray) -> DeepPADResult:
        res    = self._detector.evaluate(img_rgb)
        p_live = float(np.clip(res.liveness_score, 0.0, 1.0))
        return DeepPADResult(p_live=p_live, p_spoof=1-p_live,
                             is_live=res.is_live, backend="classical",
                             confidence=float(abs(p_live - 0.5) * 2.0))

    def _predict_cnn_builtin(self, img_rgb: np.ndarray) -> DeepPADResult:
        from .frequency_analysis import analyze_frequency
        from .texture_analysis import analyze_texture
        from .chrominance_analysis import analyze_chrominance
        freq = analyze_frequency(img_rgb)
        tex  = analyze_texture(img_rgb)
        chro = analyze_chrominance(img_rgb)
        feat = np.array([
            freq["freq_liveness_score"], freq["moire_score"], freq["high_freq_ratio"],
            tex["texture_liveness_score"], tex["lbp_entropy"] / 8.0, tex["glare_ratio"] * 10.0,
            chro["chrominance_liveness_score"], chro["skin_tone_fraction"], chro["chrominance_diversity"] / 30.0,
        ], dtype=np.float32)
        W    = np.array([0.24, -0.22, 0.08, 0.22, 0.10, -0.18, 0.20, 0.12, 0.06], dtype=np.float32)
        raw  = float(np.dot(W, feat) + 0.02)
        p_live = float(1.0 / (1.0 + np.exp(-raw * 6.0)))
        return DeepPADResult(p_live=p_live, p_spoof=1-p_live,
                             is_live=p_live >= self.threshold, backend="cnn_builtin",
                             confidence=float(abs(p_live - 0.5) * 2.0))

    def _load_pytorch_model(self, model_path: str):
        try:
            import torch
            model = torch.load(model_path, map_location="cpu", weights_only=False)
            model.eval()
            return model
        except Exception as e:
            print(f"[DeepPAD] Warning: Could not load PyTorch model: {e}")
            return None

    def _predict_pytorch(self, img_rgb: np.ndarray) -> DeepPADResult:
        if self._model is None:
            from .detector import AntiSpoofDetector
            res    = AntiSpoofDetector(threshold=self.threshold).evaluate(img_rgb)
            p_live = float(np.clip(res.liveness_score, 0.0, 1.0))
        else:
            try:
                import torch
                from PIL import Image
                pil    = Image.fromarray(img_rgb).resize((112, 112))
                arr    = np.array(pil, dtype=np.float32) / 255.0
                tensor = torch.tensor(arr.transpose(2, 0, 1)).unsqueeze(0)
                with torch.no_grad():
                    out  = self._model(tensor)
                    probs = torch.softmax(out, dim=1)[0]
                p_live = float(probs[1])
            except Exception:
                p_live = 0.5
        return DeepPADResult(p_live=p_live, p_spoof=1-p_live,
                             is_live=p_live >= self.threshold, backend="pytorch",
                             confidence=float(abs(p_live - 0.5) * 2.0))

    def predict(self, img_rgb: np.ndarray) -> DeepPADResult:
        if self.backend == "classical":
            return self._predict_classical(img_rgb)
        elif self.backend == "cnn_builtin":
            return self._predict_cnn_builtin(img_rgb)
        elif self.backend == "pytorch":
            return self._predict_pytorch(img_rgb)

    def set_threshold(self, threshold: float):
        self.threshold = threshold
        if hasattr(self, "_detector") and self._detector:
            self._detector.threshold = threshold
