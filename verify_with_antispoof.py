"""Local pretrained face comparisons and research transforms; never authorizes."""

import argparse
from dataclasses import asdict, dataclass, field
from functools import lru_cache
import json
from pathlib import Path
import pickle
import secrets
import sys

import numpy as np

from modules.antispoof.detector import AntiSpoofDetector, FaceDetectionUnavailable
from modules.image_input import load_rgb_image
from modules.face_backend import (
    FaceBackendUnavailable, FaceInputError, extract_pretrained_face,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "ckpt" / "Backbone_Epoch_24_checkpoint.pth"
DEFAULT_LIVENESS_THRESH = 0.60
DEFAULT_SLERP_ALPHA = 0.90
DEFAULT_DROP_RATE = 0.50
DEFAULT_MATCH_THRESH = 0.363  # OpenCV SFace example operating point, not locally calibrated.
COMPARISON_MARGIN = 0.03  # Display an explicit experimental uncertainty band.
GROUP_SIZE = 16


class FeatureExtractorUnavailable(RuntimeError):
    """The requested trained extractor cannot be used; no fallback is selected."""


def validate_protection_parameters(alpha, drop_rate):
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError("alpha must be finite and in [0, 1); alpha=1 destroys identity.")
    if not np.isfinite(drop_rate) or not 0 <= drop_rate <= 0.9:
        raise ValueError("drop_rate must be finite and in [0, 0.9].")


@dataclass(frozen=True)
class ProtectionContext:
    """Ephemeral research material, not an encryption credential or public report."""

    key: np.ndarray = field(repr=False)
    keep_mask: np.ndarray = field(repr=False)


def create_protection_context(shape, drop_rate=DEFAULT_DROP_RATE, seed=None):
    validate_protection_parameters(DEFAULT_SLERP_ALPHA, drop_rate)
    if len(shape) != 2 or shape[0] < 1 or shape[1] != GROUP_SIZE:
        raise ValueError("Features must contain nonempty groups of 16 dimensions.")
    # Explicit seeds are for reproducible research only; never touch global RNG.
    rng = np.random.default_rng(secrets.randbits(256) if seed is None else seed)
    key = rng.standard_normal(shape)
    key /= np.linalg.norm(key, axis=1, keepdims=True)
    keep = np.ones(shape, dtype=bool)
    for group in range(shape[0]):
        keep[group, rng.choice(GROUP_SIZE, int(GROUP_SIZE * drop_rate), replace=False)] = False
    key.setflags(write=False)
    keep.setflags(write=False)
    return ProtectionContext(key, keep)


def slerp_encrypt(features, alpha=DEFAULT_SLERP_ALPHA, drop_rate=DEFAULT_DROP_RATE,
                  seed=None, context=None):
    """Research SLERP/dropout transform; not encryption or proof of privacy.

    Use one explicit context for a diagnostic comparison. Common noise makes
    naive cosine unsuitable for identity decisions. No such decision is enabled.
    """
    validate_protection_parameters(alpha, drop_rate)
    if np.iscomplexobj(features):
        raise ValueError("Features must be real numeric values.")
    source = np.asarray(features, dtype=np.float64)
    if source.size == 0 or source.size % GROUP_SIZE or not np.isfinite(source).all():
        raise ValueError("Features must be finite, nonempty groups of 16 dimensions.")
    feat = source.reshape(-1, GROUP_SIZE)
    scale = np.max(np.abs(feat), axis=1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("Cannot protect a zero feature group.")
    scaled = feat / scale
    f_n = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    if context is not None and seed is not None:
        raise ValueError("Supply either context or a research seed, not both.")
    ctx = context or create_protection_context(feat.shape, drop_rate, seed)
    if ctx.key.shape != feat.shape or ctx.keep_mask.shape != feat.shape:
        raise ValueError("Protection context shape does not match features.")
    if np.iscomplexobj(ctx.key) or ctx.key.dtype.kind not in "uif":
        raise ValueError("Protection keys must be real numeric unit vectors.")
    if not np.isfinite(ctx.key).all() or not np.allclose(np.linalg.norm(ctx.key, axis=1), 1):
        raise ValueError("Protection key groups must be finite unit vectors.")
    if ctx.keep_mask.dtype != np.bool_ or not np.all(ctx.keep_mask.sum(axis=1) == GROUP_SIZE - int(GROUP_SIZE * drop_rate)):
        raise ValueError("Protection mask does not match the requested per-group dropout.")
    dot = np.clip(np.sum(f_n * ctx.key, axis=1, keepdims=True), -1, 1)
    theta = np.arccos(dot)
    tangent = ctx.key - dot * f_n
    lengths = np.linalg.norm(tangent, axis=1, keepdims=True)
    # Parallel/antipodal inputs need an explicit stable tangent direction.
    for i in np.flatnonzero(lengths[:, 0] < 1e-10):
        axis = np.zeros(GROUP_SIZE)
        axis[np.argmin(np.abs(f_n[i]))] = 1
        tangent[i] = axis - np.dot(axis, f_n[i]) * f_n[i]
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    transformed = np.cos(alpha * theta) * f_n + np.sin(alpha * theta) * tangent
    return np.where(ctx.keep_mask, transformed, 0.0)


def cosine_similarity(v1, v2):
    """Finite diagnostic cosine; never an authentication verdict."""
    if np.iscomplexobj(v1) or np.iscomplexobj(v2):
        raise ValueError("Similarity inputs must be real numeric values.")
    a, b = np.asarray(v1, dtype=float).ravel(), np.asarray(v2, dtype=float).ravel()
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Similarity inputs must be finite, nonempty and the same size.")
    scale_a, scale_b = np.max(np.abs(a)), np.max(np.abs(b))
    if scale_a == 0 or scale_b == 0:
        return 0.0
    a, b = a / scale_a, b / scale_b
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return float(np.clip(np.dot(a, b), -1, 1))


@lru_cache(maxsize=2)
def _load_model(path, modification_time, file_size):
    # Include metadata to avoid silently using an overwritten checkpoint.
    del modification_time, file_size
    try:
        import torch
        from modules.model import SlerpFace
    except (ImportError, OSError) as exc:
        raise FeatureExtractorUnavailable("Trained extraction needs PyTorch and the pinned local TFace backbone.") from exc
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SlerpFace(input_size=[112, 112], num_layers=50, group_size=GROUP_SIZE).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
    except (OSError, RuntimeError, ValueError, AttributeError, pickle.UnpicklingError) as exc:
        raise FeatureExtractorUnavailable("Checkpoint could not be loaded with the configured IR-50 architecture.") from exc
    return torch, model, device


def extract_face_features(img_path, model_path=None, *, demo=False):
    """Extract trained features or explicitly requested untrained demo features.

    Accepts paths, PIL images, arrays and streams. No raw input is saved.
    """
    arr = np.asarray(load_rgb_image(img_path, size=(112, 112)), dtype=np.float32)
    if not demo:
        path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
        if not path.is_file():
            raise FeatureExtractorUnavailable("Trained checkpoint is missing. Supply --model_path; use --demo only for untrained diagnostics.")
        stat = path.stat()
        torch, model, device = _load_model(str(path), stat.st_mtime_ns, stat.st_size)
        tensor = torch.from_numpy(((arr - 127.5) / 127.5).transpose(2, 0, 1).copy()).unsqueeze(0).to(device)
        with torch.inference_mode():
            vector, groups = model(tensor, flip=True)
        group = groups[0].cpu().numpy().transpose(1, 2, 0).reshape(-1, GROUP_SIZE)
        return group, vector.cpu().numpy().ravel()
    patches = []
    for row in range(7):
        for col in range(7):
            patch = arr[row * 16:(row + 1) * 16, col * 16:(col + 1) * 16]
            mean, std = patch.mean(axis=(0, 1)), patch.std(axis=(0, 1))
            patches.append(np.array([
                *(mean / 255), *(std / 128),
                np.abs(np.diff(patch, axis=0)).mean() / 128,
                np.abs(np.diff(patch, axis=1)).mean() / 128,
                *(np.median(patch, axis=(0, 1)) / 255),
                np.percentile(patch, 75) / 255, np.percentile(patch, 25) / 255,
                patch.min() / 255, patch.max() / 255, (mean[0] - mean[1]) / 128,
            ], dtype=np.float32))
    group = np.stack(patches)
    return group, group.ravel()[:512].copy()


def _face_metadata(face):
    return {"box": [int(value) for value in face.face_box],
            "landmarks": np.asarray(face.landmarks, dtype=float).tolist(),
            "detection_score": float(face.detection_score),
            "model": face.model_name, "quality": face.quality,
            "alignment": "five_landmark_112x112"}


def _presentation_estimate(image, face_box):
    try:
        from modules.presentation_backend import evaluate_presentation
        result = evaluate_presentation(image, face_box)
        return result.to_dict() if hasattr(result, "to_dict") else asdict(result)
    except (ImportError, OSError, RuntimeError, ValueError):
        # This model is advisory. Its absence cannot invent a liveness verdict
        # or prevent an explicitly labelled face-comparison experiment.
        return {"status": "UNAVAILABLE", "bona_fide_score": None,
                "reason": "Learned presentation estimate unavailable; check local model setup.",
                "model_name": None, "class_scores": {}}


def _real_image_analysis(image, liveness_thresh):
    face = extract_pretrained_face(image)
    heuristic = AntiSpoofDetector(threshold=liveness_thresh, require_face=False).evaluate(face.aligned_rgb).to_dict()
    heuristic.update(advisory=True, verdict="INCONCLUSIVE", risk_level="ADVISORY", face_count=1,
                     interpretation="Texture/frequency measurements do not establish liveness. A low score does not prove spoofing.")
    return face, heuristic, _presentation_estimate(image, face.face_box)


def inspect_real_image(image, liveness_thresh=DEFAULT_LIVENESS_THRESH):
    """Public diagnostics for the same aligned models used by comparisons.

    Embeddings and image pixels remain local and are never returned in reports.
    No-face/multiple-face inputs raise actionable FaceInputError values.
    """
    prepared = load_rgb_image(image)
    face, heuristic, presentation = _real_image_analysis(prepared, liveness_thresh)
    return {"face": _face_metadata(face), "heuristic": heuristic, "presentation": presentation}


def _run_real_comparison(img1, img2, *, liveness_thresh, slerp_alpha, drop_rate,
                         match_thresh, model_path):
    report = {"status": "UNVERIFIED", "processed": False, "is_match": None,
              "experimental_match": None, "authenticated": False,
              "tier1_status": "NOT_RUN", "tier2_status": "NOT_RUN",
              "backend": "slerpface_ir50" if model_path else "opencv_sface",
              "assurance": "Experimental face comparison only; no verified liveness or identity authorization.",
              "match_threshold": float(match_thresh), "uncertainty_margin": COMPARISON_MARGIN,
              "threshold_source": ("Custom IR-50 experiment; no source-calibrated operating point" if model_path else
                                   "User experiment setting" if match_thresh != DEFAULT_MATCH_THRESH else
                                   "OpenCV SFace example threshold; uncalibrated on this camera/data")}
    if model_path is not None and not Path(model_path).is_file():
        report.update(status="MODEL_UNAVAILABLE", reason="Configured IR-50 checkpoint is missing; no fallback was selected.")
        return False, report
    features = []
    try:
        for name, image in (("img1", img1), ("img2", img2)):
            if name == "img2" and image is None:
                continue
            prepared = load_rgb_image(image)
            face, heuristic, presentation = _real_image_analysis(prepared, liveness_thresh)
            report[f"{name}_face"] = _face_metadata(face)
            report[f"{name}_res"] = heuristic
            report[f"{name}_presentation"] = presentation
            groups, vector = (extract_face_features(face.aligned_rgb, model_path=model_path)
                              if model_path else (face.groups, face.vector))
            features.append((groups, vector))
    except FaceInputError as exc:
        report.update(status=exc.code, failed_on=name, tier1_status=exc.code,
                      reason=str(exc))
        return False, report
    except (FaceBackendUnavailable, FeatureExtractorUnavailable) as exc:
        report.update(status="MODEL_UNAVAILABLE", reason=str(exc))
        return False, report
    # A texture heuristic cannot veto a known face before a research comparison.
    # Its estimate and the learned PAD estimate remain visible and inconclusive
    # for authentication; neither one enables access or asserts a live person.
    report["tier1_status"] = "FACE_ALIGNED_PAD_ADVISORY"
    groups, vector = features[0]
    context = create_protection_context(groups.shape, drop_rate)
    protected = slerp_encrypt(groups, slerp_alpha, drop_rate, context=context)
    report.update(status="RESEARCH_TEMPLATE", processed=True, tier2_status="TRAINED_TRANSFORMED",
                  img1_template_shape=list(protected.shape),
                  zeroed_dimensions=int(np.count_nonzero(protected == 0)),
                  transform_source="IR-50 spatial groups" if model_path else "SFace embedding reshaped into groups of 16; not the SlerpFace spatial backbone",
                  reason="Local pretrained features processed. This research output grants no authentication.")
    if len(features) == 2:
        other_groups, other_vector = features[1]
        transformed = slerp_encrypt(other_groups, slerp_alpha, drop_rate, context=context)
        raw = cosine_similarity(vector, other_vector)
        if model_path and match_thresh == DEFAULT_MATCH_THRESH:
            label, experimental = "uncertain", None
            report["comparison_note"] = "The SFace default threshold cannot classify an IR-50 model. Calibrate or explicitly choose a different research operating point."
        elif raw > match_thresh + COMPARISON_MARGIN:
            label, experimental = "similar", True
        elif raw < match_thresh - COMPARISON_MARGIN:
            label, experimental = "different", False
        else:
            label, experimental = "uncertain", None
        report.update(status="RESEARCH_COMPARISON", raw_similarity=round(raw, 6),
                      encrypted_similarity=round(cosine_similarity(protected, transformed), 6),
                      comparison_label=label, experimental_match=experimental,
                      score_warning="Only the original pretrained cosine is used for the experimental label. Common noise biases transformed cosine; it is not an identity match score.")
    return False, report


def run_pipeline(img1_path, img2_path=None, liveness_thresh=DEFAULT_LIVENESS_THRESH,
                 slerp_alpha=DEFAULT_SLERP_ALPHA, drop_rate=DEFAULT_DROP_RATE,
                 match_thresh=DEFAULT_MATCH_THRESH, *, model_path=None, demo=False):
    """Return (authorized=False, report) for an actual local research operation.

    Real inputs use aligned pretrained features, a visible experimental
    comparison and advisory presentation estimates. Synthetic features require
    explicit demo selection and retain their illustrative heuristic gate.
    """
    validate_protection_parameters(slerp_alpha, drop_rate)
    if not np.isfinite(match_thresh) or not -1 <= match_thresh <= 1:
        raise ValueError("Diagnostic match threshold must be finite and in [-1, 1].")
    if not np.isfinite(liveness_thresh) or not 0 < liveness_thresh < 1:
        raise ValueError("Heuristic threshold must be finite and strictly between 0 and 1.")
    if not demo:
        return _run_real_comparison(img1_path, img2_path, liveness_thresh=liveness_thresh,
                                    slerp_alpha=slerp_alpha, drop_rate=drop_rate,
                                    match_thresh=match_thresh, model_path=model_path)
    detector = AntiSpoofDetector(threshold=liveness_thresh, require_face=False)
    report = {"status": "UNVERIFIED", "processed": False, "is_match": None,
              "authenticated": False, "tier1_status": "NOT_RUN", "tier2_status": "NOT_RUN",
              "backend": "statistical_demo" if demo else "trained_model",
              "assurance": "Research only: no validated liveness or identity policy."}
    images = []
    try:
        for name, image in (("img1", img1_path), ("img2", img2_path)):
            if name == "img2" and image is None:
                continue
            # Bind extraction to the same decoded pixels inspected by PAD.
            prepared = load_rgb_image(image)
            result = detector.evaluate(prepared)
            report[f"{name}_res"] = result.to_dict()
            if not result.is_live:
                report.update(status="REJECTED", tier1_status="REJECTED", failed_on=name,
                              reason="Input did not pass heuristic/presence checks; extraction was not run.")
                return False, report
            images.append(prepared)
    except FaceDetectionUnavailable as exc:
        report.update(status="UNAVAILABLE", reason=str(exc))
        return False, report
    report["tier1_status"] = "HEURISTIC_PASS"
    groups, vector = extract_face_features(images[0], demo=True)
    context = create_protection_context(groups.shape, drop_rate)
    protected = slerp_encrypt(groups, slerp_alpha, drop_rate, context=context)
    report.update(status="DEMO_ONLY", processed=True, tier2_status="DEMO_TRANSFORMED",
                  img1_template_shape=list(protected.shape),
                  zeroed_dimensions=int(np.count_nonzero(protected == 0)),
                  reason="Transient untrained feature demonstration; no identity verified or credential enrolled.")
    if len(images) == 2:
        group2, vector2 = extract_face_features(images[1], demo=True)
        protected2 = slerp_encrypt(group2, slerp_alpha, drop_rate, context=context)
        report.update(raw_similarity=round(cosine_similarity(vector, vector2), 6),
                      encrypted_similarity=round(cosine_similarity(protected, protected2), 6),
                      score_warning="Common noise biases transformed cosine; it is not an identity match score.")
    return False, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="GuardFace pretrained face comparison prototype; no identity authorization.")
    parser.add_argument("--img1", required=True)
    parser.add_argument("--img2")
    parser.add_argument("--threshold", type=float, default=DEFAULT_LIVENESS_THRESH)
    parser.add_argument("--alpha", type=float, default=DEFAULT_SLERP_ALPHA)
    parser.add_argument("--drop_rate", type=float, default=DEFAULT_DROP_RATE)
    parser.add_argument("--model_path", type=Path, help="Optional compatible IR-50 checkpoint override; default uses provisioned OpenCV SFace.")
    parser.add_argument("--demo", action="store_true", help="Explicit untrained demonstration; never authenticates.")
    parser.add_argument("--save_report", type=Path)
    args = parser.parse_args(argv)
    try:
        _, report = run_pipeline(args.img1, args.img2, args.threshold, args.alpha,
                                 args.drop_rate, model_path=args.model_path, demo=args.demo)
        exit_code = 0 if report["processed"] else 1
        if report["status"] in ("MODEL_UNAVAILABLE", "UNAVAILABLE"):
            exit_code = 2
    except (ValueError, OSError, FeatureExtractorUnavailable) as exc:
        report = {"status": "ERROR", "processed": False, "authenticated": False, "reason": str(exc)}
        exit_code = 2
    if args.save_report:
        try:
            args.save_report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        except OSError as exc:
            print(f"Unable to save report: {exc}", file=sys.stderr)
            return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
