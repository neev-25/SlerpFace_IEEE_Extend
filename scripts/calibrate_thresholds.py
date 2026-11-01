"""Calibrate research thresholds from an explicitly supplied real-data manifest."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.dataset_validation import evaluate_manifest, load_manifest  # noqa: E402
from modules.antispoof.detector import FaceDetectionUnavailable  # noqa: E402
from modules.face_backend import FaceBackendUnavailable  # noqa: E402
from verify_with_antispoof import FeatureExtractorUnavailable  # noqa: E402


def calibrate_thresholds(manifest_path: Path, *, backend: str = "sface",
                         model_path: Path | None = None, target_fmr: float = 0.01) -> dict:
    """Run the shared calibration/test evaluator on supplied independent captures."""
    manifest = load_manifest(manifest_path)
    if manifest["data_kind"] != "real" or not manifest["pairs"]:
        raise ValueError("Threshold calibration requires real identity pairs from a supplied manifest.")
    report = evaluate_manifest(
        manifest_path,
        backend=backend,
        model_path=model_path,
        target_fmr=target_fmr,
    )
    return {
        "scope": "identity_threshold_research",
        "authentication_enabled": False,
        "data_kind": report["data_kind"],
        "manifest_sha256": report["manifest_sha256"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "model_artifacts": report["model_artifacts"],
        "backend": report["identity"]["backend"],
        "methods": report["identity"]["methods"],
        "warning": (
            "Thresholds are research estimates from supplied labels and data. "
            "They do not authorize identity or establish population error rates."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Real-data manifest with disjoint calibration/test subjects and captures.")
    parser.add_argument("--backend", choices=("sface", "ir50"), default="sface")
    parser.add_argument("--model", type=Path, help="Required compatible checkpoint for --backend ir50.")
    parser.add_argument("--target-fmr", type=float, default=0.01)
    parser.add_argument("--save_report", type=Path, required=True,
                        help="New JSON report path; existing files are not overwritten.")
    args = parser.parse_args(argv)
    try:
        if args.save_report.exists():
            raise ValueError("Report already exists; choose a fresh path.")
        report = calibrate_thresholds(
            args.manifest,
            backend=args.backend,
            model_path=args.model,
            target_fmr=args.target_fmr,
        )
        with args.save_report.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, FaceDetectionUnavailable,
            FaceBackendUnavailable, FeatureExtractorUnavailable) as exc:
        print(f"Threshold calibration unavailable: {exc}", file=sys.stderr)
        return 2
    print("Research report written. Authentication remains disabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
