"""
modules/antispoof/metrics.py
==============================
ISO/IEC 30107-3 PAD Metrics: APCER / BPCER / ACER / DET curve / EER
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class PADMetrics:
    apcer: float
    bpcer: float
    acer:  float
    apcer_print: float
    apcer_screen: float
    total_bona_fide: int
    total_attacks: int
    false_accepted: int
    false_rejected: int

    def as_percent(self) -> dict:
        return {
            "APCER":        f"{self.apcer  * 100:.2f}%",
            "BPCER":        f"{self.bpcer  * 100:.2f}%",
            "ACER":         f"{self.acer   * 100:.2f}%",
            "APCER_Print":  f"{self.apcer_print  * 100:.2f}%",
            "APCER_Screen": f"{self.apcer_screen * 100:.2f}%",
        }


def compute_apcer(spoof_scores: List[float], threshold: float) -> float:
    if not spoof_scores:
        return 0.0
    return float(sum(1 for s in spoof_scores if s >= threshold)) / len(spoof_scores)


def compute_bpcer(bona_fide_scores: List[float], threshold: float) -> float:
    if not bona_fide_scores:
        return 0.0
    return float(sum(1 for s in bona_fide_scores if s < threshold)) / len(bona_fide_scores)


def compute_acer(apcer: float, bpcer: float) -> float:
    return (apcer + bpcer) / 2.0


def compute_det_curve(
    bona_fide_scores: List[float],
    spoof_scores: List[float],
    num_thresholds: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_scores = bona_fide_scores + spoof_scores
    thresholds = np.linspace(float(np.min(all_scores)), float(np.max(all_scores)), num_thresholds)
    fprs = np.array([compute_apcer(spoof_scores, t) for t in thresholds])
    fnrs = np.array([compute_bpcer(bona_fide_scores, t) for t in thresholds])
    return thresholds, fprs, fnrs


def find_eer(fprs: np.ndarray, fnrs: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    idx      = int(np.argmin(np.abs(fprs - fnrs)))
    eer_rate = float((fprs[idx] + fnrs[idx]) / 2.0)
    return eer_rate, float(thresholds[idx])


def evaluate_pad_system(
    bona_fide_scores: List[float],
    print_scores: List[float],
    screen_scores: List[float],
    threshold: float = 0.60,
) -> PADMetrics:
    all_spoof = print_scores + screen_scores
    return PADMetrics(
        apcer         = compute_apcer(all_spoof, threshold),
        bpcer         = compute_bpcer(bona_fide_scores, threshold),
        acer          = compute_acer(compute_apcer(all_spoof, threshold), compute_bpcer(bona_fide_scores, threshold)),
        apcer_print   = compute_apcer(print_scores, threshold),
        apcer_screen  = compute_apcer(screen_scores, threshold),
        total_bona_fide = len(bona_fide_scores),
        total_attacks   = len(all_spoof),
        false_accepted  = sum(1 for s in all_spoof if s >= threshold),
        false_rejected  = sum(1 for s in bona_fide_scores if s < threshold),
    )


def format_metrics_table(metrics: PADMetrics, system_name: str = "GuardFace (Proposed)") -> str:
    lines = [
        "| System | Front-End | Back-End | BPCER (%) | APCER (%) | ACER (%) | Security |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        "| SlerpFace (AAAI-25) | None | Slerp α=0.9 | 0.00 | 100.00 | 50.00 | DB only |",
        f"| **{system_name}** | **Dual-Domain PAD** | **Slerp α=0.9** "
        f"| **{metrics.bpcer*100:.2f}** | **{metrics.apcer*100:.2f}** "
        f"| **{metrics.acer*100:.2f}** | **End-to-End** |",
    ]
    return "\n".join(lines)
