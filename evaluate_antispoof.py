"""
evaluate_antispoof.py
======================
Evaluate image heuristics on constructed synthetic illustrations only.
This is a reproducible software demo, not a real presentation-attack or identity
benchmark, standards evaluation, privacy measurement, or comparison with a paper.
"""

import argparse
import numpy as np

from modules.antispoof.detector import AntiSpoofDetector


def generate_synthetic_benchmark_dataset(num_samples_per_class: int = 25, seed: int = 42):
    """
    Generates a controlled benchmark test set consisting of:
      1. Bona Fide (Genuine): Natural skin tones, subtle facial gradients, smooth 3D decay.
      2. Print Attacks: Halftone dot pattern, low chrominance variance, flat reflectance.
      3. Screen Replay Attacks: Periodic LCD/OLED pixel grid moiré patterns, glare hotspot.
    """
    if (not isinstance(num_samples_per_class, int) or isinstance(num_samples_per_class, bool)
            or num_samples_per_class <= 0):
        raise ValueError("num_samples_per_class must be a positive integer")
    rng = np.random.default_rng(seed)
    dataset = []

    for i in range(num_samples_per_class):
        # ── 1. Bona Fide Genuine Face ─────────────────────────────────────────
        # Base skin tone (peach/olive/brown) with natural smooth 3D illumination
        y, x = np.ogrid[:112, :112]
        cy, cx = 56, 56
        # 3D convex facial lighting falloff
        radial_light = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 45 ** 2))
        
        base_r = 180 + rng.integers(-20, 20)
        base_g = 135 + rng.integers(-15, 15)
        base_b = 105 + rng.integers(-15, 15)

        skin = np.zeros((112, 112, 3), dtype=np.float32)
        skin[:, :, 0] = base_r * (0.7 + 0.3 * radial_light) + rng.normal(0, 3.5, (112, 112))
        skin[:, :, 1] = base_g * (0.7 + 0.3 * radial_light) + rng.normal(0, 3.0, (112, 112))
        skin[:, :, 2] = base_b * (0.7 + 0.3 * radial_light) + rng.normal(0, 2.5, (112, 112))
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


def evaluate_system(threshold: float = 0.60, *, num_samples_per_class: int = 30,
                    seed: int = 42) -> dict:
    """Report only synthetic class disagreements measured by this routine.

    In particular, no feature extraction, matching, database, or privacy test is
    run here. Every image is generated from the same simple drawing process.
    """
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be finite and strictly between 0 and 1")
    detector = AntiSpoofDetector(threshold=threshold, require_face=False)
    dataset = generate_synthetic_benchmark_dataset(num_samples_per_class, seed)
    groups = {
        "BONA_FIDE": {"name": "Synthetic smooth baseline", "scores": [], "disagreements": 0},
        "PRINT_ATTACK": {"name": "Synthetic print pattern", "scores": [], "disagreements": 0},
        "SCREEN_REPLAY": {"name": "Synthetic screen pattern", "scores": [], "disagreements": 0},
    }
    for item in dataset:
        result = detector.evaluate(item["img"])
        group = groups[item["label"]]
        group["scores"].append(float(result.liveness_score))
        expected_pass = not item["is_spoof"]
        group["disagreements"] += int(result.is_live != expected_pass)

    print("SYNTHETIC DEMO: STATIC IMAGE HEURISTIC EVALUATION")
    print(f"Seed: {seed}; generated images: {len(dataset)}; heuristic threshold: {threshold:.2f}")
    print("Face-presence check disabled for drawings. No real presentations or identities tested.")
    print("| Generated class | Samples | Class disagreements | Rate (%) | Mean heuristic score |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    results = {}
    for label, group in groups.items():
        count = len(group["scores"])
        rate = group["disagreements"] / count * 100
        mean = float(np.mean(group["scores"]))
        results[label] = {"count": count, "disagreements": group["disagreements"],
                          "disagreement_rate_percent": rate, "mean_heuristic_score": mean}
        print(f"| {group['name']} | {count} | {group['disagreements']} | {rate:.2f} | {mean:.4f} |")
    print("These constructed cases do not measure real PAD accuracy, identity verification, "
          "privacy, standards compliance, or comparative performance.")
    return {"synthetic_only": True, "authentication_validated": False,
            "threshold": threshold, "seed": seed, "classes": results}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--samples-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        evaluate_system(args.threshold, num_samples_per_class=args.samples_per_class, seed=args.seed)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
