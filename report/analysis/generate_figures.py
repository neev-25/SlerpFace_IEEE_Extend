"""Reproduce the report's descriptive analysis from the project's generated PAD set.

Run from the repository root with: python report/analysis/generate_figures.py
The script reads the project code and writes inside report/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

REPORT = Path(__file__).resolve().parents[1]
PROJECT = REPORT.parent
FIGS = REPORT / "figs"
FIGS.mkdir(exist_ok=True)

# Keep the source checkout free of cache files from this analysis.
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

from evaluate_antispoof import generate_synthetic_benchmark_dataset  # noqa: E402
from modules.antispoof.detector import AntiSpoofDetector  # noqa: E402
from modules.antispoof.frequency_analysis import compute_fft_spectrum  # noqa: E402
from modules.antispoof.texture_analysis import compute_lbp  # noqa: E402
from verify_with_antispoof import slerp_encrypt  # noqa: E402


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGS / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rate(rows: list[dict], threshold: float, target: str) -> float:
    subset = [r for r in rows if (r["label"] == "BONA_FIDE") == (target == "bpcer")]
    if target == "bpcer":
        return sum(r["score"] < threshold for r in subset) / len(subset)
    return sum(r["score"] >= threshold for r in subset) / len(subset)


def main() -> None:
    dataset = generate_synthetic_benchmark_dataset(num_samples_per_class=30, seed=42)
    detector = AntiSpoofDetector(threshold=0.60)
    rows: list[dict] = []
    for item in dataset:
        result = detector.evaluate(item["img"])
        rows.append(
            {
                "id": item["id"],
                "label": item["label"],
                "score": result.liveness_score,
                "frequency_score": result.diagnostics["freq_liveness_score"],
                "texture_score": result.diagnostics["texture_liveness_score"],
                "moire_score": result.diagnostics["moire_score"],
                "lbp_entropy": result.diagnostics["lbp_entropy"],
                "chrominance_diversity": result.diagnostics["chrominance_diversity"],
                "glare_ratio": result.diagnostics["glare_ratio"],
            }
        )

    with (REPORT / "analysis" / "pad_scores.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    labels = ["BONA_FIDE", "PRINT_ATTACK", "SCREEN_REPLAY"]
    names = ["Bona fide", "Simulated print", "Simulated screen"]
    colors = ["#176B8A", "#C36A22", "#2A8D60"]
    grouped = [[r for r in rows if r["label"] == label] for label in labels]

    # The first three samples share a generated source face.
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.18), constrained_layout=True)
    for ax, item, name in zip(axes, dataset[:3], names):
        ax.imshow(item["img"])
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    _save(fig, "synthetic_triplet.png")

    # Explain the PAD diagnostics using only a generated image. No camera or
    # identity-bearing image is needed to reproduce this illustration.
    sample = dataset[0]["img"]
    gray = (0.2989 * sample[:, :, 0] + 0.5870 * sample[:, :, 1] +
            0.1140 * sample[:, :, 2])
    spectrum = compute_fft_spectrum(gray)
    lbp = compute_lbp(gray)
    result = detector.evaluate(sample)
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.15), constrained_layout=True)
    for ax, data, title, cmap in zip(
        axes[:3],
        [sample, spectrum, lbp],
        ["Generated input", "Log FFT spectrum", "LBP texture map"],
        [None, "magma", "gray"],
    ):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    axes[3].barh(
        ["Frequency", "Texture", "Fused"],
        [result.diagnostics["freq_liveness_score"],
         result.diagnostics["texture_liveness_score"], result.liveness_score],
        color=["#176B8A", "#2A8D60", "#9B2445"],
    )
    axes[3].set_xlim(0, 1)
    axes[3].axvline(0.60, ls="--", lw=1, color="black")
    axes[3].set_title("PAD scores", fontsize=9)
    axes[3].tick_params(labelsize=7)
    _save(fig, "pad_diagnostics.png")

    # Apply the actual project transform to an anonymous, synthetic descriptor.
    # The same seed gives the same key and mask for the two visualized calls.
    descriptor = np.random.default_rng(42).normal(size=(49, 16))
    rotated = slerp_encrypt(descriptor, alpha=0.90, drop_rate=0.0, seed=42)
    protected = slerp_encrypt(descriptor, alpha=0.90, drop_rate=0.50, seed=42)
    mask = protected != 0
    fig, axes = plt.subplots(1, 4, figsize=(9.0, 2.2), constrained_layout=True)
    for ax, data, title, cmap in zip(
        axes,
        [descriptor, rotated, mask, protected],
        ["Synthetic descriptor", "After SLERP", "Retained mask", "After 50% dropout"],
        ["coolwarm", "coolwarm", "Greens", "coolwarm"],
    ):
        ax.imshow(data, aspect="auto", cmap=cmap,
                  vmin=None if data.dtype == bool else -2,
                  vmax=None if data.dtype == bool else 2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Feature dimension", fontsize=7)
        ax.set_ylabel("Group", fontsize=7)
        ax.tick_params(labelsize=7)
    _save(fig, "template_transform.png")

    fig, (ax, cm_ax) = plt.subplots(1, 2, figsize=(7.05, 2.7), gridspec_kw={"width_ratios": [1.5, 1]})
    rng = np.random.default_rng(42)
    for i, (class_rows, color) in enumerate(zip(grouped, colors), 1):
        scores = [r["score"] for r in class_rows]
        ax.scatter(i + rng.uniform(-0.13, 0.13, size=len(scores)), scores, s=13, alpha=0.72, color=color)
        ax.hlines(np.median(scores), i - 0.23, i + 0.23, color="black", lw=1.5)
    ax.axhline(0.60, color="#9B2445", ls="--", lw=1.2, label="Decision threshold (0.60)")
    ax.set_xticks([1, 2, 3], names, fontsize=8)
    ax.set_ylim(-0.05, 1.06)
    ax.set_ylabel("Final liveness score", fontsize=9)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.set_title("Score distribution (30 per class)", fontsize=9)

    bona = grouped[0]
    attacks = grouped[1] + grouped[2]
    cm = np.array(
        [
            [sum(r["score"] >= 0.60 for r in bona), sum(r["score"] < 0.60 for r in bona)],
            [sum(r["score"] >= 0.60 for r in attacks), sum(r["score"] < 0.60 for r in attacks)],
        ]
    )
    cm_ax.imshow(cm, cmap="Blues", vmin=0, vmax=60)
    cm_ax.set_xticks([0, 1], ["Accepted", "Rejected"], fontsize=8)
    cm_ax.set_yticks([0, 1], ["Bona fide", "Attack"], fontsize=8)
    cm_ax.set_title("Binary decisions at 0.60", fontsize=9)
    for i in range(2):
        for j in range(2):
            cm_ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                       color="white" if cm[i, j] > 30 else "black")
    fig.tight_layout()
    _save(fig, "pad_score_and_confusion.png")

    thresholds = np.linspace(0.0, 1.0, 201)
    bpcer = np.array([_rate(rows, float(t), "bpcer") for t in thresholds])
    apcer = np.array([_rate(rows, float(t), "apcer") for t in thresholds])
    fig, ax = plt.subplots(figsize=(6.8, 2.7))
    ax.step(thresholds, bpcer, where="post", color=colors[0], lw=1.6, label="BPCER (30 bona fide)")
    ax.step(thresholds, apcer, where="post", color="#A24037", lw=1.6, label="APCER (60 attacks)")
    ax.axvline(0.60, color="black", ls="--", lw=1.0, label="Reported threshold (0.60)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Liveness decision threshold", fontsize=9)
    ax.set_ylabel("Observed error rate", fontsize=9)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, frameon=False, loc="upper center")
    fig.tight_layout()
    _save(fig, "threshold_sensitivity.png")

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.55), sharey=True)
    for ax, key, title in zip(
        axes,
        ["frequency_score", "texture_score", "score"],
        ["Frequency branch", "Texture branch", "Final fused score"],
    ):
        data = [[r[key] for r in g] for g in grouped]
        box = ax.boxplot(data, patch_artist=True, widths=0.56, showfliers=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
        ax.set_xticks([1, 2, 3], ["Bona", "Print", "Screen"], fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(-0.04, 1.05)
    axes[0].set_ylabel("Heuristic score", fontsize=9)
    axes[2].axhline(0.60, ls="--", lw=1, color="#9B2445")
    fig.tight_layout()
    _save(fig, "component_scores.png")

    summary = {
        "n_total": len(rows),
        "n_each_class": {label: len(group) for label, group in zip(labels, grouped)},
        "class_statistics": {
            label: {
                "mean": float(np.mean([r["score"] for r in group])),
                "std_population": float(np.std([r["score"] for r in group])),
                "min": float(min(r["score"] for r in group)),
                "max": float(max(r["score"] for r in group)),
                "median": float(np.median([r["score"] for r in group])),
            }
            for label, group in zip(labels, grouped)
        },
        "confusion_bona_attack_rows_accept_reject_cols": cm.tolist(),
        "threshold_sweep": {
            f"{threshold:.2f}": {
                "bpcer": _rate(rows, threshold, "bpcer"),
                "apcer": _rate(rows, threshold, "apcer"),
                "acer": (_rate(rows, threshold, "bpcer") + _rate(rows, threshold, "apcer")) / 2,
            }
            for threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
        },
        "component_means": {
            label: {
                key: float(np.mean([r[key] for r in group]))
                for key in ["frequency_score", "texture_score", "score"]
            }
            for label, group in zip(labels, grouped)
        },
    }
    (REPORT / "analysis" / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
