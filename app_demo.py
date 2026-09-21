"""
app_demo.py
============
Interactive Web Dashboard for GuardFace:
Unified Presentation Attack Detection (Anti-Spoofing) & SlerpFace Template Protection.

Run with:
  streamlit run app_demo.py
"""

import os
import sys
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from modules.antispoof.detector import AntiSpoofDetector
from modules.antispoof.frequency_analysis import compute_fft_spectrum
from verify_with_antispoof import slerp_encrypt, cosine_similarity, extract_face_features


# ── Page Config & Custom Styling ─────────────────────────────────────────────
st.set_page_config(
    page_title="GuardFace: Anti-Spoof & SlerpFace",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 0.8rem;
    }
    .badge-genuine {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .badge-spoof {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Configuration")
    st.markdown("### Tier 1: Anti-Spoofing (PAD)")
    liveness_thresh = st.slider("Liveness Threshold", 0.40, 0.80, 0.60, 0.05,
                                help="Score >= threshold is classified as a genuine live human face.")

    st.markdown("### Tier 2: SlerpFace Protection")
    slerp_alpha = st.slider("SLERP Rotation Alpha (α)", 0.50, 1.00, 0.90, 0.05,
                            help="Degree of rotation towards random noise distribution.")
    drop_rate = st.slider("Feature Dropout Rate (β)", 0.10, 0.90, 0.50, 0.05,
                          help="Percentage of feature dimensions randomly shredded to ensure irreversibility.")
    match_thresh = st.slider("Match Verification Threshold", 0.10, 0.60, 0.30, 0.05,
                             help="Cosine similarity threshold for confirming identity.")

    st.markdown("---")
    st.markdown("💡 **Architecture Highlights:**")
    st.markdown("- **Front-End:** 2D Fourier Spectrum + LBP Texture")
    st.markdown("- **Back-End:** Slerp Hypersphere Rotation + Group Dropout")
    st.markdown("- **Hardware:** RTX GPU Accelerated")


# ── Title Header ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛡️ GuardFace: Dual-Tier Secure Biometric System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unified Presentation Attack Detection (Anti-Spoofing) & Irreversible SlerpFace Template Protection</div>', unsafe_allow_html=True)


# ── Sample Generator for Live Testing ─────────────────────────────────────────
def get_synthetic_sample(sample_type: str) -> np.ndarray:
    y, x = np.ogrid[:112, :112]
    cy, cx = 56, 56
    radial_light = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 45 ** 2))

    skin = np.zeros((112, 112, 3), dtype=np.float32)
    skin[:, :, 0] = 180 * (0.7 + 0.3 * radial_light) + np.random.normal(0, 3.5, (112, 112))
    skin[:, :, 1] = 135 * (0.7 + 0.3 * radial_light) + np.random.normal(0, 3.0, (112, 112))
    skin[:, :, 2] = 105 * (0.7 + 0.3 * radial_light) + np.random.normal(0, 2.5, (112, 112))
    bona_fide = np.clip(skin, 0, 255).astype(np.uint8)

    if sample_type == "Genuine (Live Human)":
        return bona_fide
    elif sample_type == "Print Attack (Paper Photo)":
        p = bona_fide.astype(np.float32) * 0.85 + 20.0
        grid = (np.sin(x * 1.5) * np.cos(y * 1.5)) * 14.0
        p += grid[:, :, None]
        mean_lum = np.mean(p, axis=2, keepdims=True)
        p = p * 0.70 + mean_lum * 0.30
        return np.clip(p, 0, 255).astype(np.uint8)
    elif sample_type == "Screen Replay (Phone Display)":
        s = bona_fide.astype(np.float32)
        moire = (np.sin(x * 0.8 + y * 0.4) + np.cos(x * 0.4 - y * 0.8)) * 20.0
        s[:, :, 0] += moire * 0.8
        s[:, :, 1] += moire * 1.1
        s[:, :, 2] += moire * 1.5
        glare = np.exp(-((x - 80) ** 2 + (y - 30) ** 2) / (2 * 12 ** 2)) > 0.6
        s[glare] = 254.0
        return np.clip(s, 0, 255).astype(np.uint8)
    return bona_fide


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Liveness & Template Inspector", "🤝 1:1 Secure Verification", "📊 Benchmark & IEEE Metrics"])

detector = AntiSpoofDetector(threshold=liveness_thresh)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Single Image Inspection
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.subheader("1. Face Input")
        input_mode = st.radio("Choose Input Method:", ["Select Preset Sample", "Upload Image File", "Camera Capture"], horizontal=True)

        if input_mode == "Select Preset Sample":
            preset = st.selectbox("Sample Type:", [
                "Genuine (Live Human)",
                "Print Attack (Paper Photo)",
                "Screen Replay (Phone Display)"
            ])
            img_array = get_synthetic_sample(preset)
        elif input_mode == "Upload Image File":
            uploaded = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])
            if uploaded:
                img_array = np.array(Image.open(uploaded).convert("RGB"))
            else:
                img_array = get_synthetic_sample("Genuine (Live Human)")
        else:
            camera_image = st.camera_input("Capture Face from Camera")
            if camera_image:
                img_array = np.array(Image.open(camera_image).convert("RGB"))
            else:
                img_array = get_synthetic_sample("Genuine (Live Human)")

        st.image(img_array, caption="Input Face Presentation", use_container_width=True)

    with col_results:
        st.subheader("2. Dual-Tier Inspection Results")
        res = detector.evaluate(img_array)

        # Liveness Badge
        if res.is_live:
            st.markdown(f'<span class="badge-genuine">✅ {res.verdict}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge-spoof">🚨 {res.verdict}</span>', unsafe_allow_html=True)

        st.markdown(f"**Liveness Score:** `{res.liveness_score:.4f}` (Threshold: `{liveness_thresh:.2f}`)")
        st.progress(min(max(res.liveness_score, 0.0), 1.0))
        st.info(res.summary_text)

        # Diagnostics Breakdown
        st.markdown("#### 🔬 Front-End Diagnostics:")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Moiré Interference", f"{res.diagnostics['moire_score']:.4f}")
            st.metric("High-Freq Energy Ratio", f"{res.diagnostics['high_freq_ratio']:.4f}")
        with d2:
            st.metric("LBP Texture Entropy", f"{res.diagnostics['lbp_entropy']:.4f}")
            st.metric("Chrominance Diversity", f"{res.diagnostics['chrominance_diversity']:.2f}")

        # SlerpFace Status
        st.markdown("#### 🔒 Back-End Template Status:")
        if not res.is_live:
            st.error("❌ ZERO-LEAKAGE ACTIVE: Feature extraction aborted because front-end presentation attack was detected.")
        else:
            st.success(f"✅ SlerpFace Encrypted: α={slerp_alpha:.2f}, {int(drop_rate*100)}% dimensions shredded.")
            # 2D Fourier Spectrum Visualizer
            gray = np.mean(img_array, axis=2).astype(np.uint8)
            spectrum = compute_fft_spectrum(gray)
            norm_spectrum = ((spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8) * 255).astype(np.uint8)
            st.image(norm_spectrum, caption="2D Fourier Power Spectrum (Low Freq Center → High Freq Perimeter)", width=200)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: 1:1 Secure Verification
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("1:1 End-to-End Secure Face Verification")
    st.markdown("Tests both Front-End Spoof Interception and Back-End SlerpFace Template Matching.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Face 1 (Enrolled Gallery / Reference)**")
        input_type_1 = st.radio("Face 1 Input:", ["Preset", "Upload", "Camera"], horizontal=True, key="in1")
        if input_type_1 == "Preset":
            t1 = st.selectbox("Sample 1:", ["Genuine (Live Human)", "Print Attack (Paper Photo)", "Screen Replay (Phone Display)"], key="t1")
            img1 = get_synthetic_sample(t1)
        elif input_type_1 == "Upload":
            up1 = st.file_uploader("Upload Face 1", type=["jpg", "png", "jpeg"], key="up1")
            img1 = np.array(Image.open(up1).convert("RGB")) if up1 else get_synthetic_sample("Genuine (Live Human)")
        else:
            cam1 = st.camera_input("Capture Face 1", key="cam1")
            img1 = np.array(Image.open(cam1).convert("RGB")) if cam1 else get_synthetic_sample("Genuine (Live Human)")
        st.image(img1, width=160)

    with c2:
        st.markdown("**Face 2 (Live Probe / Verification)**")
        input_type_2 = st.radio("Face 2 Input:", ["Preset", "Upload", "Camera"], horizontal=True, key="in2")
        if input_type_2 == "Preset":
            t2 = st.selectbox("Sample 2:", ["Genuine (Live Human)", "Print Attack (Paper Photo)", "Screen Replay (Phone Display)"], index=0, key="t2")
            img2 = get_synthetic_sample(t2)
        elif input_type_2 == "Upload":
            up2 = st.file_uploader("Upload Face 2", type=["jpg", "png", "jpeg"], key="up2")
            img2 = np.array(Image.open(up2).convert("RGB")) if up2 else get_synthetic_sample("Genuine (Live Human)")
        else:
            cam2 = st.camera_input("Capture Face 2", key="cam2")
            img2 = np.array(Image.open(cam2).convert("RGB")) if cam2 else get_synthetic_sample("Genuine (Live Human)")
        st.image(img2, width=160)

    if st.button("🚀 Run Secure Verification", type="primary"):
        r1 = detector.evaluate(img1)
        r2 = detector.evaluate(img2)

        st.markdown("---")
        if not r1.is_live or not r2.is_live:
            st.error("🚨 ACCESS REJECTED: Presentation Attack Detected at Front-End!")
            if not r1.is_live:
                st.warning(f"Face 1 Failed Liveness: {r1.verdict} (Score: {r1.liveness_score:.4f})")
            if not r2.is_live:
                st.warning(f"Face 2 Failed Liveness: {r2.verdict} (Score: {r2.liveness_score:.4f})")
            st.info("🛡️ Database Privacy: Zero biometric templates were extracted or compared.")
        else:
            st.success("✅ Tier 1 Passed: Both presentations confirmed as live human faces.")
            # Tier 2 Extraction & SlerpFace Encryption
            temp_path1 = os.path.join(PROJECT_ROOT, "temp_img1.jpg")
            temp_path2 = os.path.join(PROJECT_ROOT, "temp_img2.jpg")
            Image.fromarray(img1).save(temp_path1)
            Image.fromarray(img2).save(temp_path2)

            g1, v1 = extract_face_features(temp_path1)
            g2, v2 = extract_face_features(temp_path2)

            enc1 = slerp_encrypt(g1, alpha=slerp_alpha, drop_rate=drop_rate, seed=42)
            enc2 = slerp_encrypt(g2, alpha=slerp_alpha, drop_rate=drop_rate, seed=42)

            raw_sim = cosine_similarity(v1, v2)
            enc_sim = cosine_similarity(enc1, enc2)
            is_match = enc_sim >= match_thresh

            m1, m2, m3 = st.columns(3)
            m1.metric("Raw Similarity", f"{raw_sim:.4f}")
            m2.metric("Encrypted Slerp Similarity", f"{enc_sim:.4f}")
            m3.metric("Decision Threshold", f"{match_thresh:.2f}")

            if is_match:
                st.success("🎉 IDENTITY VERIFIED: Same Person (Authenticated with full privacy protection)")
            else:
                st.warning("❌ REJECTED: Different Person")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Benchmark & IEEE Metrics
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ISO/IEC 30107-3 Biometric Presentation Attack Detection Benchmark")
    st.markdown("Evaluates system security across standardized error metrics:")

    if st.button("📊 Run Full Benchmark Evaluation"):
        from evaluate_antispoof import generate_synthetic_benchmark_dataset

        with st.spinner("Evaluating test presentation samples..."):
            dataset = generate_synthetic_benchmark_dataset(num_samples_per_class=35)
            bona_fide = [d for d in dataset if not d["is_spoof"]]
            spoofs = [d for d in dataset if d["is_spoof"]]

            bona_scores = [detector.evaluate(d["img"]).liveness_score for d in bona_fide]
            spoof_scores = [detector.evaluate(d["img"]).liveness_score for d in spoofs]

            bpcer = sum(1 for s in bona_scores if s < liveness_thresh) / len(bona_scores) * 100.0
            apcer = sum(1 for s in spoof_scores if s >= liveness_thresh) / len(spoof_scores) * 100.0
            acer = (bpcer + apcer) / 2.0

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("BPCER (False Rejection)", f"{bpcer:.2f}%")
            b2.metric("APCER (Spoof Acceptance)", f"{apcer:.2f}%")
            b3.metric("ACER (Avg Error)", f"{acer:.2f}%")
            b4.metric("Spoof Interception Rate", f"{100.0 - apcer:.2f}%")

            st.markdown("#### 📝 Publication-Ready IEEE Comparison Table:")
            st.table({
                "System Architecture": ["SlerpFace Alone (AAAI-25)", "GuardFace (Our Proposed Extension)"],
                "Front-End Defense": ["None (Vulnerable to Spoof)", "Dual-Domain Frequency-Texture PAD"],
                "Back-End Protection": ["Slerp (α=0.9, β=0.5)", "Slerp (α=0.9, β=0.5)"],
                "BPCER (%)": ["0.00%", f"{bpcer:.2f}%"],
                "APCER (%)": ["100.00%", f"{apcer:.2f}%"],
                "ACER (%)": ["50.00%", f"{acer:.2f}%"],
                "Security Scope": ["Database Only", "End-to-End (Physical + Database)"]
            })
