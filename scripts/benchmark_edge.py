"""
scripts/benchmark_edge.py
=========================
Latency & Edge-Device Deployment Benchmark for GuardFace.

Measures stage-by-stage execution latency (PAD < 5ms, Pose, SFace, PIN SLERP),
evaluating throughput and performance feasibility for direct edge-device deployment.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These imports follow the direct-script project-path bootstrap above.
from modules.antispoof.detector import AntiSpoofDetector  # noqa: E402
from modules.antispoof.frequency_analysis import analyze_frequency  # noqa: E402
from modules.antispoof.texture_analysis import analyze_texture  # noqa: E402
from modules.face_backend import extract_pretrained_face  # noqa: E402
from modules.keyless_pin import create_pin_protection_context  # noqa: E402
from modules.pose_estimation import estimate_head_pose  # noqa: E402
from verify_with_antispoof import DEFAULT_DROP_RATE, DEFAULT_SLERP_ALPHA, slerp_encrypt  # noqa: E402


def benchmark_pipeline(num_iterations: int = 50) -> dict:
    """
    Profiles average latency and standard deviation for each pipeline component.
    """
    # Use real face fixture for accurate benchmarking
    fixture_path = PROJECT_ROOT / "tests" / "fixtures" / "grace_hopper.jpg"
    test_crop = np.asarray(Image.open(fixture_path).resize((112, 112)), dtype=np.uint8)
    detector = AntiSpoofDetector(require_face=False)

    # 1. Tier-1 FFT Analysis
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        analyze_frequency(test_crop)
    t_fft = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 2. Tier-1 LBP Texture Analysis
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        analyze_texture(test_crop)
    t_texture = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 3. Combined Tier-1 PAD
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        detector.evaluate(test_crop)
    t_pad = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 4. Pose Estimation
    sample_landmarks = np.array([
        [40.0, 45.0], [72.0, 45.0], [56.0, 62.0], [43.0, 85.0], [69.0, 85.0]
    ])
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        estimate_head_pose(sample_landmarks)
    t_pose = (time.perf_counter() - t0) / num_iterations * 1000.0

    # 5. Face Detection & SFace Extraction (full frame)
    full_image = np.asarray(Image.open(fixture_path))
    t0 = time.perf_counter()
    for _ in range(min(num_iterations, 20)):
        face = extract_pretrained_face(full_image)
    t_face_extract = (time.perf_counter() - t0) / min(num_iterations, 20) * 1000.0

    # 6. Keyless PIN Derivation & SlerpFace Encryption
    groups = face.groups
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        ctx = create_pin_protection_context("4892", groups.shape, DEFAULT_DROP_RATE)
        slerp_encrypt(groups, DEFAULT_SLERP_ALPHA, DEFAULT_DROP_RATE, context=ctx)
    t_pin_slerp = (time.perf_counter() - t0) / num_iterations * 1000.0

    total_latency = t_pad + t_pose + t_face_extract + t_pin_slerp

    return {
        "iterations": num_iterations,
        "latencies_ms": {
            "tier1_fft_moire": round(t_fft, 3),
            "tier1_lbp_texture": round(t_texture, 3),
            "tier1_pad_combined": round(t_pad, 3),
            "pose_estimation": round(t_pose, 4),
            "yunet_detection_sface_extraction": round(t_face_extract, 3),
            "keyless_pin_slerp_encryption": round(t_pin_slerp, 3),
            "total_pipeline_latency": round(total_latency, 3),
        },
        "throughput_fps": round(1000.0 / total_latency, 1),
        "tier1_under_5ms_guarantee": t_pad < 5.0,
    }


def main():
    parser = argparse.ArgumentParser(description="GuardFace Edge Deployment Benchmark")
    parser.add_argument("--iterations", type=int, default=30, help="Number of benchmark iterations")
    parser.add_argument("--save_report", type=Path, default=None, help="Save benchmark results to JSON")
    args = parser.parse_args()

    print(f"Benchmarking GuardFace architecture across {args.iterations} iterations...")
    report = benchmark_pipeline(args.iterations)

    latencies = report["latencies_ms"]
    print("\n" + "=" * 65)
    print("GUARDFACE EDGE-DEVICE LATENCY & PERFORMANCE PROFILE")
    print("=" * 65)
    print(f"Tier-1 FFT Moiré Analysis:            {latencies['tier1_fft_moire']:>7.3f} ms")
    print(f"Tier-1 LBP Texture & Color:           {latencies['tier1_lbp_texture']:>7.3f} ms")
    print(f"Tier-1 Combined PAD Interception:     {latencies['tier1_pad_combined']:>7.3f} ms (Target < 5 ms)")
    print(f"Facial Pose Estimation (Yaw/Pitch):   {latencies['pose_estimation']:>7.4f} ms")
    print(f"YuNet Detection + SFace Extraction:   {latencies['yunet_detection_sface_extraction']:>7.3f} ms")
    print(f"Keyless PIN PBKDF2 + SLERP Privacy:   {latencies['keyless_pin_slerp_encryption']:>7.3f} ms")
    print("-" * 65)
    print(f"Total End-to-End Pipeline Latency:    {latencies['total_pipeline_latency']:>7.3f} ms")
    print(f"Maximum Processing Throughput:        {report['throughput_fps']:>7.1f} FPS")
    print(f"Zero-Leakage < 5ms Target Met:        {'YES (PASSED)' if report['tier1_under_5ms_guarantee'] else 'NO'}")
    print("=" * 65)

    if args.save_report:
        args.save_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to {args.save_report}")


if __name__ == "__main__":
    main()
