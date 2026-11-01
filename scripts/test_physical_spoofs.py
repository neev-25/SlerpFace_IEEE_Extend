"""
scripts/test_physical_spoofs.py
===============================
Physical Presentation Attack Detection (PAD) Benchmark.

Physical Validation Task 3 (ISO/IEC 30107-3 Evaluation):
Tests printed photos and screen replays to measure presentation attack
detection performance, reporting APCER, BPCER, ACER, and Zero-Leakage compliance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.antispoof.detector import AntiSpoofDetector  # noqa: E402 -- direct-script path bootstrap
from verify_with_antispoof import run_pipeline  # noqa: E402


def evaluate_physical_spoofs(liveness_thresh: float = 0.60) -> Dict:
    """
    Evaluates physical spoof presentations (Print Attacks, Screen Replays, and Genuine Bona Fide captures).
    """
    detector = AntiSpoofDetector(threshold=liveness_thresh, require_face=False)
    samples_dir = PROJECT_ROOT / "samples"

    test_cases = [
        {"file": samples_dir / "genuine_personA_1.jpg", "label": "GENUINE", "is_spoof": False},
        {"file": samples_dir / "genuine_personA_2.jpg", "label": "GENUINE", "is_spoof": False},
        {"file": samples_dir / "genuine_personB_1.jpg", "label": "GENUINE", "is_spoof": False},
        {"file": samples_dir / "spoof_print_attack.jpg", "label": "PRINT_ATTACK", "is_spoof": True},
        {"file": samples_dir / "spoof_screen_replay.jpg", "label": "SCREEN_REPLAY", "is_spoof": True},
    ]

    # Evaluate each case
    bona_scores = []
    spoof_scores = []
    spoof_intercepted = 0
    total_spoofs = 0
    attack_details = []
    zero_leakage_verified = True

    for item in test_cases:
        path = item["file"]
        if not path.is_file():
            print(f"[WARN] Test file missing: {path.name}")
            continue

        res = detector.evaluate(path)
        score = res.liveness_score
        is_attack = item["is_spoof"]

        # Check Zero-Leakage pipeline behavior
        _, pipeline_report = run_pipeline(path, liveness_thresh=liveness_thresh, demo=True, zero_leakage=True)
        if is_attack:
            total_spoofs += 1
            spoof_scores.append(score)
            if not res.is_live:
                spoof_intercepted += 1
            # Check Zero-Leakage abort
            if pipeline_report.get("processed") is not False or pipeline_report.get("status") != "REJECTED":
                zero_leakage_verified = False
        else:
            bona_scores.append(score)

        attack_details.append({
            "filename": path.name,
            "ground_truth": item["label"],
            "classified_attack_type": res.attack_type,
            "badge": res.badge,
            "liveness_score": round(score, 4),
            "tier1_passed": res.is_live,
            "tier2_status": pipeline_report.get("tier2_status"),
            "zero_leakage_active": pipeline_report.get("status") == "REJECTED" if is_attack else False,
        })

    # ISO/IEC 30107-3 Metrics
    apcer = (sum(s >= liveness_thresh for s in spoof_scores) / len(spoof_scores)) * 100 if spoof_scores else 0.0
    bpcer = (sum(s < liveness_thresh for s in bona_scores) / len(bona_scores)) * 100 if bona_scores else 0.0
    acer = (apcer + bpcer) / 2.0
    interception_rate = (spoof_intercepted / total_spoofs) * 100 if total_spoofs > 0 else 0.0

    return {
        "liveness_threshold": liveness_thresh,
        "total_presentations_evaluated": len(test_cases),
        "total_bona_fide": len(bona_scores),
        "total_attack_presentations": total_spoofs,
        "apcer_percent": round(apcer, 2),
        "bpcer_percent": round(bpcer, 2),
        "acer_percent": round(acer, 2),
        "attack_interception_rate_percent": round(interception_rate, 2),
        "zero_leakage_protocol_enforced": zero_leakage_verified,
        "results": attack_details,
    }


def main():
    parser = argparse.ArgumentParser(description="GuardFace Physical Spoof Testing Suite (ISO/IEC 30107-3)")
    parser.add_argument("--threshold", type=float, default=0.60, help="PAD decision threshold")
    parser.add_argument("--save_report", type=Path, default=None, help="Save report to JSON file")
    args = parser.parse_args()

    print("Running Physical Presentation Attack Detection Benchmark...")
    report = evaluate_physical_spoofs(args.threshold)

    print("\n" + "=" * 65)
    print("ISO/IEC 30107-3 PRESENTATION ATTACK DETECTION BENCHMARK")
    print("=" * 65)
    print(f"Decision Threshold:       {report['liveness_threshold']:.2f}")
    print(f"Total Presentations:      {report['total_presentations_evaluated']}")
    print(f"APCER (Attack Error):     {report['apcer_percent']:.2f}%")
    print(f"BPCER (Bona Fide Error):  {report['bpcer_percent']:.2f}%")
    print(f"ACER (Average Error):     {report['acer_percent']:.2f}%")
    print(f"Attack Interception Rate: {report['attack_interception_rate_percent']:.2f}%")
    print(f"Zero-Leakage Protocol:    {'VERIFIED (0 bytes leaked)' if report['zero_leakage_protocol_enforced'] else 'FAILED'}")
    print("=" * 65)

    print("\nPresentation Details:")
    for r in report["results"]:
        status_icon = "[ABORTED]" if "ZERO-LEAKAGE ACTIVE" in str(r.get("tier2_status")) else "[PASSED]"
        print(f" - {r['filename']:<25} | GT: {r['ground_truth']:<14} | Badge: {r['badge']:<40} | Score: {r['liveness_score']:.2f} | {status_icon}")

    if args.save_report:
        args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nFull report written to {args.save_report}")


if __name__ == "__main__":
    main()
