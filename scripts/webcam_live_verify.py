"""
scripts/webcam_live_verify.py
=============================
Physical Webcam Walkthrough & Real-Time Verification.

Connects to a physical webcam or video feed, detects faces, estimates pose (yaw, pitch, roll),
evaluates Tier-1 Presentation Attack Detection (FFT/LBP), enforces the Zero-Leakage Protocol,
and runs Keyless PIN SlerpFace Tier-2 comparison.

Controls:
  'r' - Enroll/Capture Reference Face
  'v' - Verify Current Frame against Reference
  'p' - Set/Change Keyless PIN
  'q' - Quit
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# These imports follow the direct-script project-path bootstrap above.
from modules.antispoof.detector import AntiSpoofDetector  # noqa: E402
from modules.face_backend import extract_pretrained_face, FaceInputError, FaceBackendUnavailable  # noqa: E402
from modules.keyless_pin import create_pin_protection_context  # noqa: E402
from modules.pose_estimation import estimate_head_pose  # noqa: E402
from verify_with_antispoof import (  # noqa: E402
    DEFAULT_DROP_RATE,
    DEFAULT_MATCH_THRESH,
    DEFAULT_SLERP_ALPHA,
    cosine_similarity,
    create_protection_context,
    slerp_encrypt,
)


def run_live_webcam(
    device_index: int = 0,
    liveness_thresh: float = 0.60,
    match_thresh: float = DEFAULT_MATCH_THRESH,
    pin: Optional[str] = None,
    headless: bool = False,
    max_frames: Optional[int] = None,
) -> dict:
    """
    Runs the live webcam loop or offline frame sequence.
    """
    # Offline CI/test mode must never open or read a physical camera.
    cap = None if headless else cv2.VideoCapture(device_index)
    if cap is not None and not cap.isOpened():
        cap.release()
        print(f"[ERROR] Could not open webcam device index {device_index}.", file=sys.stderr)
        return {"status": "CAMERA_UNAVAILABLE", "frames_processed": 0}

    detector = AntiSpoofDetector(threshold=liveness_thresh, require_face=False)
    reference_template = None
    reference_vector = None
    user_pin = pin
    frame_count = 0
    verification_history = []

    print("=" * 60)
    print("GuardFace Live Webcam Walkthrough Initialized")
    print("Press 'r' to Enroll Reference, 'v' to Verify, 'p' for PIN, 'q' to Quit")
    print("=" * 60)

    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break

            ret, frame = cap.read() if cap is not None and cap.isOpened() else (False, None)
            if not ret or frame is None:
                if headless:
                    # Synthesize test frame for headless/test validation
                    frame = np.full((480, 640, 3), 128, dtype=np.uint8)
                else:
                    print("[WARNING] Frame capture returned empty. Ending stream.")
                    break

            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            # 1. Tier-1 & Face Analysis
            display_info = {
                "badge": "SCANNING",
                "liveness_score": 0.0,
                "tier2_status": "WAITING",
                "pose": None,
                "zero_leakage_active": False,
            }

            try:
                face = extract_pretrained_face(rgb_frame)
                pad_res = detector.evaluate(face.aligned_rgb)
                display_info["badge"] = pad_res.badge
                display_info["liveness_score"] = pad_res.liveness_score
                display_info["attack_type"] = pad_res.attack_type

                pose = estimate_head_pose(face.landmarks)
                display_info["pose"] = pose

                x1, y1, x2, y2 = [int(v) for v in face.face_box]

                if not pad_res.is_live:
                    # Zero-Leakage Protocol Interception
                    display_info["zero_leakage_active"] = True
                    display_info["tier2_status"] = "ZERO-LEAKAGE ACTIVE: Feature extraction aborted..."
                    box_color = (0, 0, 255)  # Red for spoof attack
                else:
                    display_info["tier2_status"] = "TIER-1 PASSED: Ready for SlerpFace Protection"
                    box_color = (0, 255, 0)  # Green for genuine

                # Draw UI overlays on frame if not headless
                if not headless:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    for lx, ly in face.landmarks:
                        cv2.circle(frame, (int(lx), int(ly)), 3, (255, 255, 0), -1)

                    badge_text = f"{pad_res.badge} [{pad_res.liveness_score:.2f}]"
                    cv2.putText(frame, badge_text, (x1, max(25, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

                    pose_text = f"Yaw: {pose.yaw:+.1f} deg | Pitch: {pose.pitch:+.1f} deg ({pose.category})"
                    cv2.putText(frame, pose_text, (20, h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    status_color = (0, 0, 255) if display_info["zero_leakage_active"] else (0, 255, 0)
                    cv2.putText(frame, display_info["tier2_status"], (20, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

            except FaceInputError as exc:
                display_info["badge"] = f"GUIDANCE: {exc.code}"
                if not headless:
                    cv2.putText(frame, str(exc), (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            except FaceBackendUnavailable:
                display_info["badge"] = "BACKEND_UNAVAILABLE"
                if not headless:
                    cv2.putText(frame, "Face models unavailable. Run scripts/provision_face_models.py",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if not headless:
                # Keyless PIN badge
                pin_status = f"Keyless PIN: {'SET' if user_pin else 'NONE (Ephemeral)'}"
                cv2.putText(frame, pin_status, (w - 260, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

                cv2.imshow("GuardFace Biometric Walkthrough", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Enroll Reference
                    try:
                        ref_face = extract_pretrained_face(rgb_frame)
                        ref_pad = detector.evaluate(ref_face.aligned_rgb)
                        if not ref_pad.is_live:
                            print(f"[REJECT] Reference rejected by Tier-1: {ref_pad.badge}")
                        else:
                            context = (create_pin_protection_context(user_pin, ref_face.groups.shape, DEFAULT_DROP_RATE)
                                       if user_pin else create_protection_context(ref_face.groups.shape, DEFAULT_DROP_RATE))
                            reference_template = slerp_encrypt(ref_face.groups, DEFAULT_SLERP_ALPHA, DEFAULT_DROP_RATE, context=context)
                            reference_vector = ref_face.vector
                            print(f"[ENROLLED] Reference enrolled successfully. Groups: {ref_face.groups.shape}")
                    except Exception as exc:
                        print(f"[ERROR] Enrollment failed: {exc}")

                elif key == ord('v'):
                    # Verify probe against reference
                    if reference_template is None:
                        print("[INFO] No reference face enrolled yet. Press 'r' first.")
                    else:
                        try:
                            probe_face = extract_pretrained_face(rgb_frame)
                            probe_pad = detector.evaluate(probe_face.aligned_rgb)
                            if not probe_pad.is_live:
                                print(f"[ZERO-LEAKAGE ACTIVE] Probe presentation rejected: {probe_pad.badge}")
                                verification_history.append({"matched": False, "reason": "ZERO_LEAKAGE_TRIGGERED"})
                            else:
                                context = (create_pin_protection_context(user_pin, probe_face.groups.shape, DEFAULT_DROP_RATE)
                                           if user_pin else create_protection_context(probe_face.groups.shape, DEFAULT_DROP_RATE))
                                probe_template = slerp_encrypt(probe_face.groups, DEFAULT_SLERP_ALPHA, DEFAULT_DROP_RATE, context=context)
                                raw_sim = cosine_similarity(reference_vector, probe_face.vector)
                                enc_sim = cosine_similarity(reference_template, probe_template)
                                is_match = raw_sim >= match_thresh
                                print(f"[VERIFY] Raw Cosine: {raw_sim:.4f} | Encrypted Cosine: {enc_sim:.4f} | Match: {is_match}")
                                verification_history.append({"matched": is_match, "raw_similarity": raw_sim, "encrypted_similarity": enc_sim})
                        except Exception as exc:
                            print(f"[ERROR] Verification failed: {exc}")

                elif key == ord('p'):
                    new_pin = input("\nEnter memorized Keyless PIN (or empty to clear): ").strip()
                    user_pin = new_pin or None
                    print(f"Keyless PIN updated: {'SET' if user_pin else 'CLEARED'}")

    finally:
        if cap is not None:
            cap.release()
        if not headless:
            cv2.destroyAllWindows()

    return {
        "status": "COMPLETED",
        "frames_processed": frame_count,
        "verifications_performed": len(verification_history),
        "history": verification_history,
    }


def main():
    parser = argparse.ArgumentParser(description="GuardFace Physical Webcam Walkthrough")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index")
    parser.add_argument("--pin", type=str, default=None, help="Optional Keyless PIN")
    parser.add_argument("--liveness_thresh", type=float, default=0.60, help="PAD threshold")
    parser.add_argument("--headless", action="store_true", help="Run without graphical window for CI/testing")
    parser.add_argument("--max_frames", type=int, default=None, help="Limit frames to process")
    args = parser.parse_args()

    run_live_webcam(
        device_index=args.device,
        liveness_thresh=args.liveness_thresh,
        pin=args.pin,
        headless=args.headless,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
