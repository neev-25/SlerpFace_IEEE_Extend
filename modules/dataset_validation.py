"""Offline evaluation with separate calibration/test subjects; never authorizes.

Dataset declarations describe supplied evidence, not independently certified
ground truth. Reports exclude image paths, subject identifiers and embeddings.
"""

from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from modules.antispoof.detector import AntiSpoofDetector
from modules.image_input import MAX_IMAGE_BYTES, load_rgb_image
from modules.face_backend import FaceBackendUnavailable, extract_pretrained_face, face_backend_status
from verify_with_antispoof import (
    cosine_similarity, create_protection_context, extract_face_features,
    slerp_encrypt, validate_protection_parameters,
)

MAX_RECORDS = 100_000
MAX_MANIFEST_BYTES = 20 * 1024 * 1024


def _bounded_bytes(path, limit):
    with path.open("rb") as source:
        value = source.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Input exceeds the evaluation file-size bound.")
    return value


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Image paths must be relative to the manifest directory.")
    result = (root / relative).resolve()
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("An image is missing or outside the dataset directory.")
    return result


def _identifier(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Each record needs a nonempty subject identifier of at most 256 characters.")
    return value


def load_manifest(path):
    path = Path(path).resolve()
    raw = _bounded_bytes(path, MAX_MANIFEST_BYTES)
    document = json.loads(raw)
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise ValueError("Expected dataset manifest schema_version 1.")
    if document.get("data_kind") not in ("real", "synthetic"):
        raise ValueError("data_kind must explicitly be real or synthetic.")
    if not isinstance(document.get("source"), str) or not document["source"].strip():
        raise ValueError("Declare the dataset source/provenance in source.")
    pairs, presentations = document.get("pairs", []), document.get("presentations", [])
    if not isinstance(pairs, list) or not isinstance(presentations, list) or not pairs and not presentations:
        raise ValueError("Supply a nonempty pairs or presentations list.")
    if len(pairs) + len(presentations) > MAX_RECORDS:
        raise ValueError("Too many evaluation records.")
    root = path.parent
    subjects = {"calibration": set(), "test": set()}
    digests = {"calibration": set(), "test": set()}
    labels = {"calibration": [], "test": []}
    resolved_pairs, resolved_presentations = [], []
    image_hashes = {}
    image_subjects = {}
    seen_pairs = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or not isinstance(pair.get("split"), str)
                or pair["split"] not in subjects):
            raise ValueError("Every identity pair needs split calibration or test.")
        same = pair.get("same_person")
        if type(same) is not bool:
            raise ValueError("same_person must be a JSON boolean.")
        left_id, right_id = _identifier(pair.get("left_subject")), _identifier(pair.get("right_subject"))
        if (left_id == right_id) != same:
            raise ValueError("Pair labels disagree with the declared subject identities.")
        left, right = _image_path(root, pair.get("left")), _image_path(root, pair.get("right"))
        if left == right:
            raise ValueError("Identity pairs must use distinct captures, not an image paired with itself.")
        split = pair["split"]
        subjects[split].update((left_id, right_id))
        if subjects["calibration"] & subjects["test"]:
            raise ValueError("Calibration and test subjects must be disjoint.")
        pair_digests = []
        for image, subject in ((left, left_id), (right, right_id)):
            digest = hashlib.sha256(_bounded_bytes(image, MAX_IMAGE_BYTES)).hexdigest()
            if image in image_hashes and image_hashes[image] != digest:
                raise ValueError("Dataset image changed during manifest validation.")
            image_hashes[image] = digest
            if digest in image_subjects and image_subjects[digest] != subject:
                raise ValueError("An image has conflicting declared subject identities.")
            image_subjects[digest] = subject
            digests[split].add(digest)
            pair_digests.append(digest)
        if pair_digests[0] == pair_digests[1]:
            raise ValueError("Identity pairs need distinct captures; copied image bytes are not a second capture.")
        pair_key = tuple(sorted(pair_digests))
        if pair_key in seen_pairs:
            raise ValueError("Duplicate or reversed image pair, including renamed copies.")
        seen_pairs.add(pair_key)
        labels[split].append(same)
        resolved_pairs.append({**pair, "left": left, "right": right})
    if pairs:
        if subjects["calibration"] & subjects["test"]:
            raise ValueError("Calibration and test subjects must be disjoint.")
        if digests["calibration"] & digests["test"]:
            raise ValueError("Calibration and test image contents overlap.")
        for split in subjects:
            if set(labels[split]) != {True, False}:
                raise ValueError("Both splits need genuine and impostor identity pairs.")
    seen_presentations = set()
    for item in presentations:
        if not isinstance(item, dict) or item.get("kind") not in ("bona_fide", "print", "screen", "mask", "other_attack"):
            raise ValueError("Presentation kind must declare bona_fide or an explicit attack type.")
        _identifier(item.get("subject"))
        capture = item.get("capture_kind")
        if capture not in ("camera", "physical_attack", "digital_replay", "synthetic"):
            raise ValueError("Presentation capture_kind must explicitly describe its acquisition.")
        if document["data_kind"] == "real" and capture == "synthetic":
            raise ValueError("Synthetic presentations cannot be declared as real data.")
        image = _image_path(root, item.get("image"))
        digest = hashlib.sha256(_bounded_bytes(image, MAX_IMAGE_BYTES)).hexdigest()
        if digest in seen_presentations:
            raise ValueError("Duplicate presentation image contents, including renamed copies.")
        seen_presentations.add(digest)
        if image in image_hashes and image_hashes[image] != digest:
            raise ValueError("Dataset image changed during manifest validation.")
        image_hashes[image] = digest
        resolved_presentations.append({**item, "image": image})
    return {"data_kind": document["data_kind"], "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "pairs": resolved_pairs, "presentations": resolved_presentations,
            "image_hashes": image_hashes}


def _snapshot_image(path, manifest):
    contents = _bounded_bytes(path, MAX_IMAGE_BYTES)
    if hashlib.sha256(contents).hexdigest() != manifest["image_hashes"][path]:
        raise ValueError("Dataset image changed after manifest validation; restart evaluation.")
    return load_rgb_image(io.BytesIO(contents))


def _error_rate(errors, trials):
    if trials <= 0:
        raise ValueError("Error rates need observations.")
    interval = binomtest(errors, trials).proportion_ci(confidence_level=0.95, method="exact")
    return {"errors": int(errors), "trials": int(trials), "rate": float(errors / trials),
            "confidence_level": 0.95, "interval": [float(interval.low), float(interval.high)],
            "interval_assumption": "Nominal binomial interval assumes independent trials; repeated subjects/captures can violate this."}


def identity_rates(scores, labels, threshold):
    if np.iscomplexobj(scores) or np.iscomplexobj(threshold):
        raise ValueError("Scores and threshold must be real.")
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.ndim != 1 or labels.shape != scores.shape or labels.dtype != np.bool_:
        raise ValueError("Scores and boolean identity labels must be equal-length vectors.")
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or np.any(np.abs(scores) > 1):
        raise ValueError("Scores must be finite cosines; threshold must be finite.")
    if not labels.any() or labels.all():
        raise ValueError("Both genuine and impostor comparisons are required.")
    predicted = scores >= threshold
    return {"false_match": _error_rate(np.count_nonzero(predicted & ~labels), np.count_nonzero(~labels)),
            "false_nonmatch": _error_rate(np.count_nonzero(~predicted & labels), np.count_nonzero(labels))}


def calibrate_threshold(scores, labels, target_fmr=0.01):
    """Choose a threshold using calibration observations only, with explicit CI.

    This meets an empirical target on supplied pairs, not a population guarantee.
    Test observations are evaluated later and never used in this selection.
    """
    if np.iscomplexobj(scores) or np.iscomplexobj(target_fmr):
        raise ValueError("Scores and target_fmr must be real.")
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if not np.isfinite(target_fmr) or not 0 <= target_fmr < 1:
        raise ValueError("target_fmr must be finite and in [0, 1).")
    identity_rates(scores, labels, 0)
    impostor = np.sort(scores[~labels])
    candidates = np.unique(np.append(scores, np.nextafter(scores.max(), np.inf)))
    accepted = impostor.size - np.searchsorted(impostor, candidates, side="left")
    feasible = candidates[accepted / impostor.size <= target_fmr]
    threshold = float(feasible[0])
    return {"threshold": threshold, "target_empirical_fmr": float(target_fmr),
            "selection": "calibration_only", **identity_rates(scores, labels, threshold)}


def group_cosine(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.shape != left.shape or left.shape[0] == 0 or left.shape[1] != 16:
        raise ValueError("Expected matching feature groups of shape (N,16).")
    return float(np.mean([cosine_similarity(a, b) for a, b in zip(left, right)]))


def evaluate_identity_pairs(manifest, *, model_path=None, demo=False, seed=42,
                            alpha=0.9, drop_rate=0.5, target_fmr=0.01, extractor=None, backend="ir50"):
    validate_protection_parameters(alpha, drop_rate)
    if demo and manifest["data_kind"] != "synthetic":
        raise ValueError("Untrained features cannot evaluate a dataset declared real.")
    if not manifest["pairs"]:
        return None
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Research seed must be a nonnegative integer.")
    selected_extractor = extractor or extract_face_features

    @lru_cache(maxsize=256)
    def feature(path):
        image = _snapshot_image(path, manifest)
        if not demo and extractor is None:
            face = extract_pretrained_face(image)
            return ((face.groups, face.vector) if backend == "sface" else
                    selected_extractor(face.aligned_rgb, model_path=model_path, demo=False))
        return selected_extractor(image, model_path=model_path, demo=demo)

    scores = {"raw_vector_cosine": {"calibration": [], "test": []},
              "slerp_group_cosine": {"calibration": [], "test": []}}
    labels = {"calibration": [], "test": []}
    for pair in manifest["pairs"]:
        g1, v1 = feature(pair["left"])
        g2, v2 = feature(pair["right"])
        # Reproducible evaluation-only context. Public seeds must never become credentials.
        image_digest = manifest["image_hashes"][pair["left"]]
        derived = hashlib.sha256(f"{seed}:{pair['split']}:{image_digest}".encode()).digest()
        context = create_protection_context(g1.shape, drop_rate, seed=int.from_bytes(derived, "big"))
        p1 = slerp_encrypt(g1, alpha, drop_rate, context=context)
        p2 = slerp_encrypt(g2, alpha, drop_rate, context=context)
        split = pair["split"]
        scores["raw_vector_cosine"][split].append(cosine_similarity(v1, v2))
        scores["slerp_group_cosine"][split].append(group_cosine(p1, p2))
        labels[split].append(pair["same_person"])
    results = {}
    for method, observations in scores.items():
        calibrated = calibrate_threshold(observations["calibration"], labels["calibration"], target_fmr)
        results[method] = {"calibration": calibrated,
                           "test": identity_rates(observations["test"], labels["test"], calibrated["threshold"])}
    return {"backend": "untrained_synthetic_demo" if demo else f"{backend}_trained_model_research",
            "methods": results, "alpha": alpha, "drop_rate": drop_rate, "evaluation_seed": seed,
            "warning": "Common-noise group scores need held-out validation. No identity decision is enabled."}


def evaluate_presentations(manifest, *, threshold=0.6, detector=None):
    if not manifest["presentations"]:
        return None
    detector = detector or AntiSpoofDetector(threshold=threshold, require_face=manifest["data_kind"] != "synthetic")
    counts = {}
    for item in manifest["presentations"]:
        image = _snapshot_image(item["image"], manifest)
        result = detector.evaluate(image)
        group = counts.setdefault(item["kind"], {"trials": 0, "errors": 0, "face_detection_failures": 0})
        group["trials"] += 1
        group["errors"] += int(result.is_live != (item["kind"] == "bona_fide"))
        group["face_detection_failures"] += int(result.verdict in ("NO_FACE", "MULTIPLE_FACES"))
    rates = {name: {**_error_rate(item["errors"], item["trials"]),
                    "face_detection_failures": item["face_detection_failures"]} for name, item in counts.items()}
    return {"threshold": threshold, "classes": rates, "backend": "static_image_heuristic",
            "warning": "Rates include face/presence rejection. Capture labels are supplied ground truth, not certified physical trials."}


def evaluate_manifest(path, *, model_path=None, demo=False, threshold=0.6,
                      alpha=0.9, drop_rate=0.5, target_fmr=0.01, seed=42, backend="ir50"):
    manifest = load_manifest(path)
    if backend not in ("ir50", "sface"):
        raise ValueError("Identity backend must be ir50 or sface.")
    if backend == "sface" and model_path is not None:
        raise ValueError("--model supplies an IR-50 checkpoint; select --backend ir50 for it.")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Presentation threshold must be finite and strictly between 0 and 1.")
    if manifest["data_kind"] == "synthetic" and not demo:
        raise ValueError("A synthetic manifest requires explicit --demo.")
    if manifest["data_kind"] == "real" and demo:
        raise ValueError("Untrained features cannot evaluate a dataset declared real.")
    if manifest["data_kind"] == "real" and manifest["pairs"] and backend == "ir50" and model_path is None:
        raise ValueError("Real identity evaluation requires an explicit model checkpoint path.")
    if model_path is not None and not Path(model_path).is_file():
        raise ValueError("Configured model checkpoint is missing.")
    checkpoint_hash = _file_sha256(model_path) if model_path else None
    sface_artifacts = None
    if backend == "sface" and manifest["pairs"] and not demo:
        status = face_backend_status()
        if not status["ready"]:
            raise FaceBackendUnavailable(status["reason"])
        sface_artifacts = status["models"]
    report = {"scope": "offline_evaluation", "authentication_enabled": False,
            "data_kind": manifest["data_kind"], "manifest_sha256": manifest["manifest_sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "model_artifacts": sface_artifacts,
            "identity": evaluate_identity_pairs(manifest, model_path=model_path, demo=demo, seed=seed,
                                                 alpha=alpha, drop_rate=drop_rate, target_fmr=target_fmr, backend=backend),
            "presentation": evaluate_presentations(manifest, threshold=threshold)}
    if model_path is not None and _file_sha256(model_path) != checkpoint_hash:
        raise ValueError("Checkpoint changed during evaluation; restart with a stable model snapshot.")
    if sface_artifacts is not None and face_backend_status()["models"] != sface_artifacts:
        raise ValueError("Pretrained face models changed during evaluation; restart with stable files.")
    return report
