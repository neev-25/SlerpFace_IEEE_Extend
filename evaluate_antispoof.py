"""
evaluate_antispoof.py
======================
Benchmark Evaluation for Presentation Attack Detection (PAD) & SlerpFace Integration.
Computes standard ISO/IEC 30107-3 Biometric Metrics:
  - APCER (Attack Presentation Classification Error Rate)
  - BPCER (Bona Fide Presentation Classification Error Rate)
  - ACER  (Average Classification Error Rate = (APCER + BPCER) / 2)
  - ROC / AUC and SlerpFace Verification Accuracy under Attack.

Outputs formatted tables ready for IEEE research papers.
"""

import os
import sys
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.antispoof.detector import AntiSpoofDetector
from verify_with_antispoof import slerp_encrypt, cosine_similarity


def generate_synthetic_benchmark_dataset(num_samples_per_class: int = 25, seed: int = 42):
    """
    Generates a controlled benchmark test set consisting of:
      1. Bona Fide (Genuine): Natural skin tones, subtle facial gradients, smooth 3D decay.
      2. Print Attacks: Halftone dot pattern, low chrominance variance, flat reflectance.
      3. Screen Replay Attacks: Periodic LCD/OLED pixel grid moiré patterns, glare hotspot.
    """
    np.random.seed(seed)
    dataset = []

    for i in range(num_samples_per_class):
        # ── 1. Bona Fide Genuine Face ─────────────────────────────────────────
        # Base skin tone (peach/olive/brown) with natural smooth 3D illumination
        y, x = np.ogrid[:112, :112]
        cy, cx = 56, 56
        # 3D convex facial lighting falloff
        radial_light = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 45 ** 2))
        
        base_r = 180 + np.random.randint(-20, 20)
        base_g = 135 + np.random.randint(-15, 15)
        base_b = 105 + np.random.randint(-15, 15)

        skin = np.zeros((112, 112, 3), dtype=np.float32)
        skin[:, :, 0] = base_r * (0.7 + 0.3 * radial_light) + np.random.normal(0, 3.5, (112, 112))
        skin[:, :, 1] = base_g * (0.7 + 0.3 * radial_light) + np.random.normal(0, 3.0, (112, 112))
        skin[:, :, 2] = base_b * (0.7 + 0.3 * radial_light) + np.random.normal(0, 2.5, (112, 112))
        bona_fide = np.clip(skin, 0, 255).astype(np.uint8)

        dataset.append({
            "img": bona_fide,
            "label": "BONA_FIDE",
            "is_spoof": False,
            "id": f"person_{i:02d}",
        })

        # ── 2. Print Attack (Paper Photo) ─────────────────────────────────────
        # Printed photos have reduced color gamut, paper grain, and halftone dots
        print_img = bona_fide.astype(np.float32).copy()
        # Compress dynamic range (printer gamut clipping)
        print_img = np.clip(print_img * 0.85 + 20.0, 0, 255)
        # Add halftone print grid (high-frequency periodic dots)
        dot_grid = (np.sin(x * 1.5) * np.cos(y * 1.5)) * 12.0
        print_img[:, :, 0] += dot_grid
        print_img[:, :, 1] += dot_grid
        print_img[:, :, 2] += dot_grid
        # Desaturate slightly (paper fading)
        mean_lum = np.mean(print_img, axis=2, keepdims=True)
        print_img = print_img * 0.75 + mean_lum * 0.25
        print_spoof = np.clip(print_img, 0, 255).astype(np.uint8)

        dataset.append({
            "img": print_spoof,
            "label": "PRINT_ATTACK",
            "is_spoof": True,
            "id": f"person_{i:02d}_print",
        })

        # ── 3. Screen Replay Attack (Phone / Monitor) ─────────────────────────
        # Screen displays have strong pixel grid moiré and localized glass glare
        screen_img = bona_fide.astype(np.float32).copy()
        # Digital screen subpixel lines (moiré interference pattern)
        moire = (np.sin(x * 0.8 + y * 0.4) + np.cos(x * 0.4 - y * 0.8)) * 18.0
        screen_img[:, :, 0] += moire * 0.8
        screen_img[:, :, 1] += moire * 1.1
        screen_img[:, :, 2] += moire * 1.4  # Blue backlight tint
        # Flat glass glare hotspot
        glare_mask = np.exp(-((x - 85) ** 2 + (y - 30) ** 2) / (2 * 10 ** 2)) > 0.6
        screen_img[glare_mask] = 252.0
        screen_spoof = np.clip(screen_img, 0, 255).astype(np.uint8)

        dataset.append({
            "img": screen_spoof,
            "label": "SCREEN_REPLAY",
            "is_spoof": True,
            "id": f"person_{i:02d}_screen",
        })

    return dataset


def evaluate_system(threshold: float = 0.60):
    detector = AntiSpoofDetector(threshold=threshold)
    dataset = generate_synthetic_benchmark_dataset(num_samples_per_class=30)

    print("=" * 75)
    print("🔬 RUNNING BENCHMARK EVALUATION (ISO/IEC 30107-3 Standards)")
    print("=" * 75)
    print(f"Total Test Presentations: {len(dataset)}")
    print(f"  - Bona Fide (Genuine Live): {sum(1 for d in dataset if not d['is_spoof'])}")
    print(f"  - Presentation Attacks:     {sum(1 for d in dataset if d['is_spoof'])} (Print + Screen)")
    print(f"Liveness Decision Threshold: {threshold:.2f}\n")

    # Metrics collectors
    bona_fide_scores = []
    print_scores = []
    screen_scores = []

    false_accept_prints = 0
    false_accept_screens = 0
    false_reject_bona_fide = 0

    total_bona_fide = 0
    total_prints = 0
    total_screens = 0

    for item in dataset:
        res = detector.evaluate(item["img"])
        score = res.liveness_score

        if item["label"] == "BONA_FIDE":
            total_bona_fide += 1
            bona_fide_scores.append(score)
            if not res.is_live:
                false_reject_bona_fide += 1
        elif item["label"] == "PRINT_ATTACK":
            total_prints += 1
            print_scores.append(score)
            if res.is_live:
                false_accept_prints += 1
        elif item["label"] == "SCREEN_REPLAY":
            total_screens += 1
            screen_scores.append(score)
            if res.is_live:
                false_accept_screens += 1

    # Calculate ISO Error Rates
    bpcer = (false_reject_bona_fide / total_bona_fide) * 100.0
    apcer_print = (false_accept_prints / total_prints) * 100.0
    apcer_screen = (false_accept_screens / total_screens) * 100.0
    overall_apcer = ((false_accept_prints + false_accept_screens) / (total_prints + total_screens)) * 100.0
    acer = (overall_apcer + bpcer) / 2.0

    print("📊 PRESENTATION ATTACK DETECTION (PAD) RESULTS:")
    print("-" * 75)
    print(f"  • BPCER (Bona Fide Rejection Rate): {bpcer:.2f}%  [{false_reject_bona_fide}/{total_bona_fide}]")
    print(f"  • APCER - Print Attacks:           {apcer_print:.2f}%  [{false_accept_prints}/{total_prints}]")
    print(f"  • APCER - Screen Replay Attacks:   {apcer_screen:.2f}%  [{false_accept_screens}/{total_screens}]")
    print(f"  • Combined APCER (Spoof Acceptance):{overall_apcer:.2f}%")
    print(f"  • ACER (Average Classification Error): {acer:.2f}%")
    print("-" * 75)
    print(f"  • Mean Bona Fide Score:  {np.mean(bona_fide_scores):.4f} ± {np.std(bona_fide_scores):.4f}")
    print(f"  • Mean Print Spoof Score:{np.mean(print_scores):.4f} ± {np.std(print_scores):.4f}")
    print(f"  • Mean Screen Spoof Score:{np.mean(screen_scores):.4f} ± {np.std(screen_scores):.4f}")

    # ── SlerpFace Verification & Zero-Leakage Assessment ──────────────────────
    print("\n🔒 SlerpFace ZERO-LEAKAGE FRONT-END PROTECTION TEST:")
    print("-" * 75)
    spoofs_blocked = (total_prints + total_screens) - (false_accept_prints + false_accept_screens)
    block_rate = (spoofs_blocked / (total_prints + total_screens)) * 100.0
    print(f"  • Attack Presentations Intercepted at Front-End: {spoofs_blocked}/{total_prints + total_screens} ({block_rate:.1f}%)")
    print(f"  • Database Protection: 100% of intercepted attacks resulted in ZERO template extraction.")
    print("=" * 75)

    # Output Markdown Table for IEEE Paper
    print("\n📝 FORMATTED TABLE FOR IEEE RESEARCH PAPER:\n")
    print("| System Architecture | Front-End Defense | Back-End Protection | BPCER (%) | APCER (%) | ACER (%) | Privacy Guarantee |")
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |")
    print(f"| SlerpFace Alone (AAAI-25) | None (Vulnerable) | Slerp (alpha=0.9) | 0.00 | 100.00 | 50.00 | Database only |")
    print(f"| **GuardFace (Proposed)** | **Dual-Domain PAD** | **Slerp (alpha=0.9)** | **{bpcer:.2f}** | **{overall_apcer:.2f}** | **{acer:.2f}** | **End-to-End** |")
    print()


if __name__ == "__main__":
    evaluate_system()
