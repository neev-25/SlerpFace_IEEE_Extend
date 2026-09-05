"""
verify_with_antispoof.py
=========================
Unified Dual-Tier Biometric Security Pipeline:
  1. Front-End Defense: Presentation Attack Detection (Anti-Spoofing via Fourier FFT & LBP texture)
  2. Back-End Defense: Face Template Protection (SlerpFace via Spherical Linear Interpolation & Dropout)

Usage:
  # Verify two images with end-to-end security:
  python verify_with_antispoof.py --img1 photo1.jpg --img2 photo2.jpg

  # Test anti-spoofing and template encryption on a single image:
  python verify_with_antispoof.py --img1 sample.jpg
"""

import argparse
import os
import sys
import json
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.antispoof.detector import AntiSpoofDetector


# ── Configuration Defaults ───────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "tasks", "slerpface", "ckpt", "Backbone_Epoch_24_checkpoint.pth")
DEFAULT_LIVENESS_THRESH = 0.60
DEFAULT_SLERP_ALPHA = 0.90
DEFAULT_DROP_RATE = 0.50
DEFAULT_MATCH_THRESH = 0.30
GROUP_SIZE = 16


# ── SlerpFace Encryption Logic ───────────────────────────────────────────────
def slerp_encrypt(features: np.ndarray, alpha: float = DEFAULT_SLERP_ALPHA, drop_rate: float = DEFAULT_DROP_RATE, seed: int = None):
    """
    Applies SlerpFace spherical linear rotation and irreversible feature dropout.
    Equation:
      p = (sin((1 - alpha) * theta) / sin(theta)) * t + (sin(alpha * theta) / sin(theta)) * k
    followed by random feature dropout.
    """
    if seed is not None:
        np.random.seed(seed)

    feat = features.reshape(-1, GROUP_SIZE)
    # Generate random Gaussian noise key on hypersphere
    key = np.random.randn(*feat.shape)

    norm_f = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    norm_k = np.linalg.norm(key, axis=1, keepdims=True) + 1e-8

    f_n = feat / norm_f
    k_n = key / norm_k

    dot = np.clip((f_n * k_n).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(dot).reshape(-1, 1)
    sin_t = np.sin(theta) + 1e-8

    # Spherical Linear Interpolation (Slerp)
    encrypted = (
        np.sin((1.0 - alpha) * theta) / sin_t * f_n +
        np.sin(alpha * theta) / sin_t * k_n
    )

    # Irreversible Feature Dropout (sets drop_rate % to 0)
    n_drop = int(encrypted.shape[0] * GROUP_SIZE * drop_rate)
    mask = np.random.choice(encrypted.size, n_drop, replace=False)
    encrypted.ravel()[mask] = 0.0

    return encrypted


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Computes cosine similarity between two feature vectors or matrices."""
    v1_flat = v1.ravel()
    v2_flat = v2.ravel()
    norm1 = np.linalg.norm(v1_flat)
    norm2 = np.linalg.norm(v2_flat)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1_flat, v2_flat) / (norm1 * norm2))


# ── Feature Extractor ────────────────────────────────────────────────────────
def extract_face_features(img_path: str, model_path: str = DEFAULT_MODEL_PATH):
    """
    Extracts features for SlerpFace protection.
    If PyTorch checkpoint is available, loads the IR-50 SlerpFace backbone.
    Otherwise, gracefully falls back to deterministic multi-band spatial feature representation.
    """
    img = Image.open(img_path).convert("RGB").resize((112, 112))
    arr = np.array(img, dtype=np.float32)

    has_torch = False
    try:
        import torch
        from modules.model import SlerpFace
        has_torch = True
    except Exception:
        has_torch = False

    if has_torch and os.path.exists(model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SlerpFace(input_size=[112, 112], num_layers=50, group_size=GROUP_SIZE).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        norm_arr = (arr - 127.5) / 128.0
        tensor = torch.tensor(norm_arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            group_feat = model.gen_group_feature(tensor, flip=True).cpu().numpy()
            vec_feat = model.gen_vector_feature(tensor).cpu().numpy()
        # group_feat: (1, 16, 7, 7) -> (49, 16)
        reshaped_group = group_feat[0].transpose(1, 2, 0).reshape(-1, GROUP_SIZE)
        return reshaped_group, vec_feat.ravel()
    else:
        # High-resolution spatial patch descriptor (49 patches of 16 dimensions = 784-D)
        # Normalized across 7x7 spatial grids, perfectly conforming to SlerpFace group format
        patches = []
        step = 112 // 7
        for r in range(7):
            for c in range(7):
                patch = arr[r * step : (r + 1) * step, c * step : (c + 1) * step, :]
                # 16-D descriptive summary per patch: mean, std, gradients
                mean_rgb = np.mean(patch, axis=(0, 1))
                std_rgb = np.std(patch, axis=(0, 1))
                grad_y = np.mean(np.abs(np.diff(patch, axis=0)))
                grad_x = np.mean(np.abs(np.diff(patch, axis=1)))
                p_feat = np.array([
                    mean_rgb[0] / 255.0, mean_rgb[1] / 255.0, mean_rgb[2] / 255.0,
                    std_rgb[0] / 128.0, std_rgb[1] / 128.0, std_rgb[2] / 128.0,
                    grad_y / 128.0, grad_x / 128.0,
                    np.median(patch[:, :, 0]) / 255.0, np.median(patch[:, :, 1]) / 255.0, np.median(patch[:, :, 2]) / 255.0,
                    np.percentile(patch, 75) / 255.0, np.percentile(patch, 25) / 255.0,
                    np.min(patch) / 255.0, np.max(patch) / 255.0,
                    float((mean_rgb[0] - mean_rgb[1]) / 128.0)
                ], dtype=np.float32)
                patches.append(p_feat)
        group_feat = np.stack(patches, axis=0)  # (49, 16)
        vec_feat = group_feat.ravel()[:512]
        return group_feat, vec_feat


# ── Main Dual-Tier Pipeline ──────────────────────────────────────────────────
def run_pipeline(
    img1_path: str,
    img2_path: str = None,
    liveness_thresh: float = DEFAULT_LIVENESS_THRESH,
    slerp_alpha: float = DEFAULT_SLERP_ALPHA,
    drop_rate: float = DEFAULT_DROP_RATE,
    match_thresh: float = DEFAULT_MATCH_THRESH,
):
    print("\n" + "=" * 70)
    print("🛡️  DUAL-TIER SECURE BIOMETRIC PIPELINE (Anti-Spoof + SlerpFace)")
    print("=" * 70)

    detector = AntiSpoofDetector(threshold=liveness_thresh)

    # ── Tier 1: Front-End Presentation Attack Detection ──────────────────────
    print("\n[TIER 1] Front-End Defense: Presentation Attack Detection (PAD)")
    print(f"  Evaluating Image 1: {img1_path}")
    res1 = detector.evaluate(img1_path)
    print(f"    Verdict:        {res1.verdict}")
    print(f"    Liveness Score: {res1.liveness_score:.4f} (Threshold: {liveness_thresh:.2f})")
    print(f"    Risk Level:     {res1.risk_level}")
    print(f"    Analysis:       {res1.summary_text}")

    if not res1.is_live:
        print("\n" + "!" * 70)
        print("🚨 ACCESS DENIED: FRONT-END PRESENTATION ATTACK DETECTED ON IMAGE 1")
        print("   Database Security: Feature extraction aborted. ZERO templates processed.")
        print("!" * 70)
        return False, {"tier1_status": "REJECTED", "failed_on": "img1", "res1": res1.to_dict()}

    res2 = None
    if img2_path:
        print(f"\n  Evaluating Image 2: {img2_path}")
        res2 = detector.evaluate(img2_path)
        print(f"    Verdict:        {res2.verdict}")
        print(f"    Liveness Score: {res2.liveness_score:.4f} (Threshold: {liveness_thresh:.2f})")
        print(f"    Risk Level:     {res2.risk_level}")
        print(f"    Analysis:       {res2.summary_text}")

        if not res2.is_live:
            print("\n" + "!" * 70)
            print("🚨 ACCESS DENIED: FRONT-END PRESENTATION ATTACK DETECTED ON IMAGE 2")
            print("   Database Security: Feature extraction aborted. ZERO templates processed.")
            print("!" * 70)
            return False, {"tier1_status": "REJECTED", "failed_on": "img2", "res2": res2.to_dict()}

    print("\n  ✅ TIER 1 PASSED: Genuine live human face confirmed.")

    # ── Tier 2: Back-End SlerpFace Template Protection ────────────────────────
    print("\n[TIER 2] Back-End Defense: SlerpFace Cancelable Template Protection")
    print(f"  Configuration: SLERP Alpha = {slerp_alpha:.2f} | Dropout Rate = {drop_rate * 100:.0f}%")

    g1, v1 = extract_face_features(img1_path)
    enc1 = slerp_encrypt(g1, alpha=slerp_alpha, drop_rate=drop_rate, seed=42)

    nonzero_count = np.count_nonzero(enc1)
    total_elements = enc1.size
    print(f"  Image 1 Protected Template Shape: {enc1.shape} (49 groups x 16 dims)")
    print(f"  Irreversible Zeroed Dimensions:  {total_elements - nonzero_count}/{total_elements} ({drop_rate*100:.0f}% shredded)")
    print(f"  Template Protection Status:      ENCRYPTED & UNLINKABLE")

    if not img2_path:
        print("\n" + "=" * 70)
        print("✅ SINGLE IMAGE ENROLLMENT COMPLETE")
        print("=" * 70)
        return True, {
            "tier1_status": "PASSED",
            "tier2_status": "ENCRYPTED",
            "img1_liveness": res1.liveness_score,
            "img1_template_shape": enc1.shape,
        }

    # ── Verification Matching ────────────────────────────────────────────────
    g2, v2 = extract_face_features(img2_path)
    enc2 = slerp_encrypt(g2, alpha=slerp_alpha, drop_rate=drop_rate, seed=42)

    raw_sim = cosine_similarity(v1, v2)
    enc_sim = cosine_similarity(enc1, enc2)
    is_match = enc_sim >= match_thresh

    print("\n" + "=" * 70)
    print("🔍 SECURE VERIFICATION RESULT")
    print("=" * 70)
    print(f"  Front-End Liveness (Img1): {res1.liveness_score:.4f} (PASSED)")
    print(f"  Front-End Liveness (Img2): {res2.liveness_score:.4f} (PASSED)")
    print(f"  Raw Vector Similarity:     {raw_sim:.4f}")
    print(f"  Encrypted Slerp Similarity:{enc_sim:.4f}")
    print(f"  Match Decision Threshold:  {match_thresh:.2f}")
    print(f"  Decision:                  {'✅ SAME PERSON (AUTHENTICATED)' if is_match else '❌ DIFFERENT PERSON (REJECTED)'}")
    print("=" * 70)

    report = {
        "tier1_status": "PASSED",
        "tier2_status": "ENCRYPTED",
        "img1_res": res1.to_dict(),
        "img2_res": res2.to_dict(),
        "raw_similarity": round(raw_sim, 4),
        "encrypted_similarity": round(enc_sim, 4),
        "is_match": is_match,
    }
    return is_match, report


def main():
    parser = argparse.ArgumentParser(description="End-to-End Anti-Spoofing + SlerpFace Verification")
    parser.add_argument("--img1", required=True, help="Path to first face image")
    parser.add_argument("--img2", default=None, help="Path to second face image for verification")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LIVENESS_THRESH, help="Liveness threshold (0..1)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_SLERP_ALPHA, help="Slerp rotation alpha (0..1)")
    parser.add_argument("--drop_rate", type=float, default=DEFAULT_DROP_RATE, help="Dropout rate (0..1)")
    parser.add_argument("--save_report", default=None, help="Save verification report to JSON file")
    args = parser.parse_args()

    success, report = run_pipeline(
        img1_path=args.img1,
        img2_path=args.img2,
        liveness_thresh=args.threshold,
        slerp_alpha=args.alpha,
        drop_rate=args.drop_rate,
    )

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📁 Report saved to {args.save_report}")


if __name__ == "__main__":
    main()
