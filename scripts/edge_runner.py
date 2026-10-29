"""
scripts/edge_runner.py
======================
Lightweight, Headless Edge Deployment Runner for GuardFace.

Designed for IoT/Edge devices (Raspberry Pi, Jetson, edge gateways) without GUI dependencies.
Accepts image paths or camera feed, evaluates Tier-1 PAD with Zero-Leakage, applies Keyless PIN
binding, and outputs structured JSON results.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verify_with_antispoof import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="GuardFace Lightweight Edge Runner")
    parser.add_argument("--img1", required=True, help="Path to primary face image")
    parser.add_argument("--img2", default=None, help="Optional path to probe comparison image")
    parser.add_argument("--pin", type=str, default=None, help="Memorized PIN for Keyless PIN Binding")
    parser.add_argument("--liveness_thresh", type=float, default=0.60, help="PAD threshold")
    parser.add_argument("--zero_leakage", action="store_true", default=True, help="Enforce Zero-Leakage abort")
    parser.add_argument("--out", type=Path, default=None, help="Output JSON path")
    args = parser.parse_args()

    try:
        _, report = run_pipeline(
            args.img1,
            args.img2,
            liveness_thresh=args.liveness_thresh,
            pin=args.pin,
            zero_leakage=args.zero_leakage,
        )
        exit_code = 0 if report.get("processed") else 1
    except Exception as exc:
        report = {"status": "EDGE_EXECUTION_ERROR", "error": str(exc), "processed": False}
        exit_code = 2

    formatted_json = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(formatted_json, encoding="utf-8")
    print(formatted_json)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
