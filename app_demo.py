"""GuardFace research dashboard. Run with ``streamlit run app_demo.py``.

Static-image PAD scores and transformed descriptors are research diagnostics;
this application does not make an authentication or liveness assurance decision.
Uploaded images stay in memory and are never written by this dashboard.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

from modules.antispoof.detector import AntiSpoofDetector
from modules.antispoof.frequency_analysis import compute_fft_spectrum
from modules.image_input import load_rgb_image
from verify_with_antispoof import inspect_real_image, run_pipeline
from modules.face_backend import FaceBackendUnavailable, FaceInputError, face_backend_status


PRESETS = [
    "Synthetic skin-texture pattern",
    "Synthetic print pattern",
    "Synthetic screen pattern",
]


def get_synthetic_sample(sample_type: str, seed: int = 42) -> np.ndarray:
    """Create a reproducible illustration without modifying NumPy's global RNG."""
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:112, :112]
    radial_light = np.exp(-((x - 56) ** 2 + (y - 56) ** 2) / (2 * 45 ** 2))
    skin = np.zeros((112, 112, 3), dtype=np.float32)
    for channel, (mean, deviation) in enumerate([(180, 3.5), (135, 3.0), (105, 2.5)]):
        skin[:, :, channel] = mean * (0.7 + 0.3 * radial_light) + rng.normal(0, deviation, (112, 112))
    sample = np.clip(skin, 0, 255).astype(np.uint8)
    if sample_type == PRESETS[0]:
        return sample
    if sample_type == PRESETS[1]:
        pattern = sample.astype(np.float32) * 0.85 + 20.0
        pattern += ((np.sin(x * 1.5) * np.cos(y * 1.5)) * 14.0)[:, :, None]
        pattern = pattern * 0.70 + np.mean(pattern, axis=2, keepdims=True) * 0.30
        return np.clip(pattern, 0, 255).astype(np.uint8)
    if sample_type == PRESETS[2]:
        pattern = sample.astype(np.float32)
        moire = (np.sin(x * 0.8 + y * 0.4) + np.cos(x * 0.4 - y * 0.8)) * 20.0
        for channel, scale in enumerate([0.8, 1.1, 1.5]):
            pattern[:, :, channel] += moire * scale
        pattern[np.exp(-((x - 80) ** 2 + (y - 30) ** 2) / (2 * 12 ** 2)) > 0.6] = 254
        return np.clip(pattern, 0, 255).astype(np.uint8)
    raise ValueError("Unknown synthetic sample type")


def preset_sample(slot: str, sample_type: str) -> np.ndarray:
    """Retain the selected sample in this session across unrelated reruns."""
    key = f"sample_{slot}"
    if key not in st.session_state or st.session_state[key][0] != sample_type:
        st.session_state[key] = (sample_type, get_synthetic_sample(sample_type))
    return st.session_state[key][1]


def decode_capture(capture):
    if capture is None:
        return None
    try:
        if getattr(capture, "size", 0) > 20 * 1024 * 1024:
            raise ValueError("Image file exceeds the upload limit")
        return np.asarray(load_rgb_image(capture))
    except (ValueError, OSError, TypeError):
        st.error("Unable to read this image. Use a valid JPG or PNG within the image size limits.")
        return None


def select_image(slot: str, label: str):
    mode = st.radio(label, ["Preset", "Upload", "Camera"], horizontal=True, key=f"in{slot}")
    if mode == "Preset":
        sample_type = st.selectbox(f"Sample {slot}:", PRESETS, key=f"t{slot}")
        img = preset_sample(slot, sample_type)
        st.caption("Synthetic illustration; it is not a captured face or liveness evidence.")
    elif mode == "Upload":
        img = decode_capture(st.file_uploader(f"Upload Image {slot}", type=["jpg", "jpeg", "png"], key=f"up{slot}"))
    else:
        img = decode_capture(st.camera_input(f"Capture Image {slot}", key=f"cam{slot}"))
    if img is None:
        st.info("Supply an image to continue.")
    elif mode != "Camera":
        st.image(img, caption="Selected input", width="stretch")
    return img, mode == "Preset"


def inspect_pad(img, threshold, *, synthetic=False):
    """Cache only the latest diagnostics within the user's session."""
    digest = hashlib.sha256(img.tobytes()).digest()
    key = (digest, img.shape, threshold, synthetic, backend.get("ready", False))
    cached = st.session_state.get("_inspector_pad")
    if cached is None or cached[0] != key:
        if synthetic:
            result = {"heuristic": AntiSpoofDetector(threshold=threshold, require_face=False).evaluate(img).to_dict()}
        else:
            result = inspect_real_image(img, liveness_thresh=threshold)
        st.session_state["_inspector_pad"] = (key, result)
    return st.session_state["_inspector_pad"][1]


def render_face_details(face, label):
    """Show only public quality metadata, never descriptors or protection material."""
    if not isinstance(face, dict):
        return
    confidence = face.get("confidence", face.get("detection_score"))
    if isinstance(confidence, (int, float)):
        st.caption(f"{label}: one aligned face; detection score {confidence:.3f}.")
    quality = face.get("quality")
    if isinstance(quality, dict):
        width = quality.get("face_width_pixels")
        height = quality.get("face_height_pixels")
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            st.caption(f"{label} detected face size: {width:.0f} × {height:.0f} pixels before alignment.")
        warnings = quality.get("warnings", [])
        if isinstance(warnings, list):
            for warning in warnings:
                if isinstance(warning, str):
                    st.warning(f"{label}: {warning}")


def render_presentation(presentation, label):
    """A learned estimate is advisory; it must never become a liveness claim."""
    if not isinstance(presentation, dict):
        return
    status = presentation.get("status", "UNAVAILABLE")
    st.caption(f"{label} presentation analysis: {status}. Static-image estimates do not establish liveness.")
    # A public model label is useful; avoid dumping arbitrary report payloads.
    estimate = presentation.get("label", presentation.get("prediction"))
    if isinstance(estimate, str):
        st.write(f"Model estimate: {estimate}")
    score = presentation.get("bona_fide_score")
    if isinstance(score, (int, float)) and np.isfinite(score):
        st.write(f"**Bona fide model score (uncalibrated):** {score:.4f}")
        st.caption("This model score has no validated threshold for your camera and does not determine the face similarity result.")
    if isinstance(presentation.get("model_name"), str):
        st.caption(f"Presentation model: {presentation['model_name']}")
    if status == "UNAVAILABLE":
        st.caption("Presentation model unavailable; face comparison can still run as an unverified experiment.")


def render_report(report):
    """Render actual core outcomes, without turning diagnostic scores into identity claims."""
    st.write(f"Processing status: {report.get('status', 'UNVERIFIED')}")
    if report.get("processed"):
        if report.get("status") == "DEMO_ONLY":
            st.info("Template demonstration completed. Identity remains unverified; synthetic descriptors do not measure face identity.")
        else:
            label = report.get("comparison_label")
            if label in {"similar", "different", "uncertain"}:
                st.info(f"Experimental comparison: {label}. Identity remains unverified; this is not an authentication decision.")
                threshold = report.get("match_threshold", match_thresh)
                st.caption(f"Based on raw face cosine, threshold {threshold:.3f}, with an uncertainty band of ±0.030. This operating point is not calibrated to your camera or population.")
            else:
                st.info("Research template extracted from the aligned face. Identity remains unverified; no comparison was requested.")
        columns = st.columns(2)
        if "raw_similarity" in report:
            columns[0].metric("Raw descriptor cosine (diagnostic)", f"{report['raw_similarity']:.4f}")
        if "encrypted_similarity" in report:
            columns[1].metric("Transformed descriptor cosine (diagnostic)", f"{report['encrypted_similarity']:.4f}")
        if "encrypted_similarity" in report:
            st.caption("The SLERP/dropout cosine illustrates the feature transform. It does not determine the experimental comparison result.")
    else:
        st.warning("No template demonstration completed. Identity remains unverified.")
    for number in (1, 2):
        render_face_details(report.get(f"img{number}_face"), f"Input {number}")
        render_presentation(report.get(f"img{number}_presentation"), f"Input {number}")
    if report.get("reason"):
        st.write(report["reason"])


def run_experiment(img1, img2=None):
    try:
        _, report = run_pipeline(
            img1, img2,
            liveness_thresh=liveness_thresh,
            slerp_alpha=slerp_alpha,
            drop_rate=drop_rate,
            match_thresh=match_thresh,
            model_path=model_path.strip() or None,
            demo=demo_features,
        )
    except (ValueError, OSError, RuntimeError, TypeError):
        st.error("Processing unavailable. Check the input images and local model configuration.")
        return {"status": "PROCESSING_ERROR", "processed": False,
                "reason": "Processing unavailable. Check the input images and local model configuration."}
    return report


def experiment_key(img1, img2=None):
    """Invalidate stale results on input/configuration changes, without retaining pixels."""
    if img1 is None:
        return None
    images = tuple(None if image is None else (image.shape, image.dtype.str, hashlib.sha256(image.tobytes()).digest())
                   for image in (img1, img2))
    checkpoint = model_path.strip()
    metadata = None
    if checkpoint:
        try:
            stat = Path(checkpoint).stat()
            metadata = (stat.st_size, stat.st_mtime_ns)
        except (OSError, ValueError):
            pass
    return (images, liveness_thresh, slerp_alpha, drop_rate, match_thresh,
            checkpoint, metadata, demo_features)


def experiment_action(slot, img1, img2=None):
    """Keep a result visible across reruns until its inputs/settings change."""
    key = experiment_key(img1, img2)
    storage = f"_experiment_{slot}"
    cached = st.session_state.get(storage)
    if cached is not None and cached[0] != key:
        del st.session_state[storage]
        cached = None
    label = "Run template demonstration" if slot == "inspect" else "Run pair comparison"
    clicked = st.button(label, key=f"{slot}_run", type="primary" if slot == "pair" else "secondary",
                        disabled=img1 is None or (slot == "pair" and img2 is None))
    if clicked:
        with st.spinner("Detecting, aligning and extracting research features..." if not demo_features else "Running synthetic feature demonstration..."):
            report = run_experiment(img1, img2)
        cached = (key, report)
        st.session_state[storage] = cached
    if cached is not None:
        render_report(cached[1])
    else:
        st.caption("No result for the current inputs and settings. Run the explicit operation above.")


st.set_page_config(page_title="GuardFace Research Demo", page_icon="🛡️", layout="wide")
st.title("🛡️ GuardFace Research Demo")
st.warning("Research diagnostics only. Face similarity is experimental; static-image presentation checks do not establish liveness. This prototype does not authenticate anyone.")

with st.sidebar:
    st.header("Research configuration")
    liveness_thresh = st.slider("PAD heuristic threshold", 0.40, 0.80, 0.60, 0.05,
                                help="Controls the synthetic illustration gate. On real photos the texture heuristic is advisory and cannot block face comparison.")
    slerp_alpha = st.slider("SLERP rotation alpha", 0.00, 0.95, 0.90, 0.05,
                            help="Alpha 1 discards identity information and is excluded.")
    drop_rate = st.slider("Feature dropout rate", 0.00, 0.90, 0.50, 0.05)
    match_thresh = st.slider("Experimental cosine threshold", -1.000, 1.000, 0.363, 0.001, format="%.3f",
                             help="Starting SFace cosine operating point from OpenCV's LFW example; not calibrated locally. Scores within ±0.030 are uncertain. Only raw face cosine determines the experimental label.")
    demo_features = st.checkbox("Use synthetic feature demo", value=False,
                                help="Explicitly use image statistics and skip the face-presence check. These are not biometric identity descriptors; inputs need not contain a face.")
    model_path = st.text_input("Trained checkpoint path (optional)",
                               help="Leave empty to use the provisioned YuNet + SFace backend. A custom compatible IR-50 checkpoint is an advanced research option, with a separately calibrated operating point required.")
    try:
        backend = face_backend_status()
    except (ValueError, OSError, RuntimeError, TypeError):
        backend = {"ready": False}
    if demo_features:
        st.caption("Synthetic features selected. The face threshold and trained checkpoint do not turn this illustration into a biometric comparison.")
    elif model_path.strip():
        st.caption("Custom checkpoint selected. The SFace default operating point is not calibrated for a custom model.")
    else:
        if backend.get("ready"):
            st.caption("Face backend ready: YuNet detection/alignment + SFace recognition (CPU). No checkpoint entry required.")
        else:
            st.warning("Face backend unavailable. Run .\\.venv\\Scripts\\python.exe scripts/provision_face_models.py to install the reviewed models.")
    st.caption("Capture or upload one clear face per image for a real comparison. Images remain in memory; no GPU or cryptographic security guarantee is implied.")

tab1, tab2, tab3 = st.tabs(["🔍 PAD & Template Inspector", "🤝 Pair Comparison", "📊 Synthetic Benchmark"])

with tab1:
    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("1. Image input")
        inspector_img, synthetic_input = select_image("inspect", "Inspector input:")
        if synthetic_input and not demo_features:
            st.info("Presets contain generated textures, not faces. Enable Use synthetic feature demo for these illustrations, or select Upload/Camera for real face features.")
    with right:
        st.subheader("2. Heuristic diagnostics")
        if inspector_img is not None:
            try:
                result = inspect_pad(inspector_img, liveness_thresh, synthetic=synthetic_input)
            except FaceInputError as exc:
                result = None
                st.warning(str(exc))
            except FaceBackendUnavailable:
                result = None
                st.warning("Face analysis unavailable. Check the model setup shown in the sidebar, then retry.")
            except (ValueError, OSError, RuntimeError, TypeError):
                result = None
                st.error("Image analysis could not process this input. Try a clear, evenly lit photo with exactly one face.")
            if result is not None and isinstance(result.get("heuristic"), dict):
                heuristic = result["heuristic"]
                score = heuristic.get("liveness_score", 0.0)
                st.metric("PAD heuristic score", f"{score:.4f}")
                st.progress(float(np.clip(score, 0, 1)))
                if synthetic_input:
                    st.write("Synthetic illustration gate: " + ("passed" if heuristic.get("is_live") else "rejected"))
                else:
                    st.write("Heuristic assessment: inconclusive (advisory only)")
                    render_face_details(result.get("face"), "Inspector")
                    render_presentation(result.get("presentation"), "Inspector")
                st.caption("Texture scores do not establish liveness. Real face comparison uses the detected, aligned face and does not reject a capture because of this heuristic score.")
                diagnostics = heuristic.get("diagnostics", {})
                # Plain text inherits the readable theme color; JSON number
                # highlighting has insufficient contrast in the light theme.
                for name, label in [
                    ("moire_score", "Moiré score"),
                    ("high_freq_ratio", "High frequency ratio"),
                    ("lbp_entropy", "LBP entropy"),
                    ("chrominance_diversity", "Chrominance diversity"),
                ]:
                    if name in diagnostics:
                        st.write(f"**{label}:** {diagnostics[name]:.4f}")
                gray = np.mean(inspector_img, axis=2).astype(np.uint8)
                # Bound this optional visual independently of the detector's own analysis size.
                spectrum_input = np.asarray(Image.fromarray(gray).resize((256, 256)))
                spectrum = compute_fft_spectrum(spectrum_input)
                normalized = ((spectrum - spectrum.min()) / (np.ptp(spectrum) + 1e-8) * 255).astype(np.uint8)
                st.image(normalized, caption="Fourier spectrum visualization", width=200)
            elif result is not None and result.get("reason"):
                st.info(result["reason"])
        st.subheader("3. Template operation")
        experiment_action("inspect", inspector_img)

with tab2:
    st.subheader("Pair comparison experiment")
    st.write("Supply two photos, each with exactly one clear face, then run comparison. Faces are detected and aligned before trained extraction. Presentation analysis is advisory; its score cannot establish a live person.")
    first, second = st.columns(2)
    with first:
        st.markdown("**Reference input**")
        img1, synthetic1 = select_image("1", "Image 1 input:")
    with second:
        st.markdown("**Comparison input**")
        img2, synthetic2 = select_image("2", "Image 2 input:")
    if (synthetic1 or synthetic2) and not demo_features:
        st.info("A selected preset is a generated texture. Enable Use synthetic feature demo to compare presets, or use two real face photos with that option off.")
    experiment_action("pair", img1, img2)

with tab3:
    st.subheader("Synthetic heuristic benchmark")
    st.write("These constructed patterns test code behavior. They do not measure physical attacks, real identities, or compliance with ISO/IEC 30107-3.")
    benchmark = st.session_state.get("_benchmark_report")
    if benchmark is not None and benchmark[0] != liveness_thresh:
        del st.session_state["_benchmark_report"]
        benchmark = None
    if st.button("Run synthetic benchmark", key="benchmark_run"):
        from evaluate_antispoof import generate_synthetic_benchmark_dataset

        with st.spinner("Evaluating synthetic patterns..."):
            synthetic_detector = AntiSpoofDetector(threshold=liveness_thresh, require_face=False)
            dataset = generate_synthetic_benchmark_dataset(num_samples_per_class=35)
            bona_scores = [synthetic_detector.evaluate(d["img"]).liveness_score for d in dataset if not d["is_spoof"]]
            spoof_scores = [synthetic_detector.evaluate(d["img"]).liveness_score for d in dataset if d["is_spoof"]]
            bpcer = sum(score < liveness_thresh for score in bona_scores) / len(bona_scores) * 100
            apcer = sum(score >= liveness_thresh for score in spoof_scores) / len(spoof_scores) * 100
            benchmark = (liveness_thresh, bpcer, apcer, len(dataset))
            st.session_state["_benchmark_report"] = benchmark
    if benchmark is not None:
        _, bpcer, apcer, count = benchmark
        cols = st.columns(3)
        cols[0].metric("Synthetic BPCER", f"{bpcer:.2f}%")
        cols[1].metric("Synthetic APCER", f"{apcer:.2f}%")
        cols[2].metric("Synthetic ACER", f"{(bpcer + apcer) / 2:.2f}%")
        st.caption(f"Evaluated {count} generated patterns. Independent camera, print, screen, and identity validation remains required.")
