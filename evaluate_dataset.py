"""Evaluate a declared dataset offline; no authentication policy is enabled."""

import argparse
import json
from pathlib import Path
import sys

from modules.dataset_validation import evaluate_manifest
from modules.face_backend import FaceBackendUnavailable
from modules.antispoof.detector import FaceDetectionUnavailable
from verify_with_antispoof import FeatureExtractorUnavailable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--backend", choices=("sface", "ir50"), default="sface",
                        help="Identity extractor: provisioned pretrained SFace (default), or an explicit IR-50 checkpoint.")
    parser.add_argument("--demo", action="store_true", help="Permit untrained evaluation of explicitly synthetic fixtures only.")
    parser.add_argument("--threshold", type=float, default=0.6, help="Static-image heuristic threshold.")
    parser.add_argument("--target-fmr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--drop", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise ValueError("Output already exists; choose a fresh report path.")
        report = evaluate_manifest(args.manifest, model_path=args.model, demo=args.demo,
                                   threshold=args.threshold, target_fmr=args.target_fmr,
                                   alpha=args.alpha, drop_rate=args.drop, seed=args.seed, backend=args.backend)
        with args.out.open("x", encoding="utf-8") as destination:
            destination.write(json.dumps(report, indent=2, allow_nan=False))
    except (ValueError, OSError, FaceDetectionUnavailable, FeatureExtractorUnavailable, FaceBackendUnavailable) as exc:
        print(f"Offline evaluation unavailable: {exc}", file=sys.stderr)
        return 2
    print("Offline report written. No authentication decision or deployment policy was enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
