"""Evaluate supplied physical PAD captures; never substitute generated samples."""

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


def evaluate_physical_spoofs(manifest_path: Path, liveness_thresh: float = 0.60) -> dict:
    """Evaluate declared camera/physical presentations from a supplied manifest."""
    manifest = load_manifest(manifest_path)
    if manifest["data_kind"] != "real" or not manifest["presentations"]:
        raise ValueError("Physical PAD evaluation requires a real-data presentation manifest.")

    kinds = {item["kind"] for item in manifest["presentations"]}
    if "bona_fide" not in kinds or not kinds.intersection({"print", "screen", "mask", "other_attack"}):
        raise ValueError("Include bona fide camera captures and at least one attack class.")
    if any(item["kind"] == "bona_fide" and item["capture_kind"] != "camera"
           for item in manifest["presentations"]):
        raise ValueError("Bona fide presentations must be declared as camera captures.")
    if any(item["kind"] != "bona_fide" and item["capture_kind"] != "physical_attack"
           for item in manifest["presentations"]):
        raise ValueError("Attack presentations must be declared as physical_attack captures.")

    report = evaluate_manifest(
        manifest_path,
        backend="sface",
        threshold=liveness_thresh,
    )
    return {
        "scope": "declared_physical_presentation_research",
        "authentication_enabled": False,
        "presentation": report["presentation"],
        "warning": (
            "Capture types and labels are operator-supplied and not independently verified. "
            "Metrics do not establish deployment readiness or certification."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Real-data manifest with consented camera and physical-attack captures.")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--save_report", type=Path, required=True,
                        help="New JSON report path; existing files are not overwritten.")
    args = parser.parse_args(argv)
    try:
        if args.save_report.exists():
            raise ValueError("Report already exists; choose a fresh path.")
        report = evaluate_physical_spoofs(args.manifest, args.threshold)
        with args.save_report.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, FaceDetectionUnavailable,
            FaceBackendUnavailable, FeatureExtractorUnavailable) as exc:
        print(f"Physical PAD evaluation unavailable: {exc}", file=sys.stderr)
        return 2
    print("Research report written. No physical capture provenance or authentication policy was certified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
