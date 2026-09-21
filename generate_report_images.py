"""
generate_report_images.py
==========================
Passes a real face image through the complete SlerpFace + PAD pipeline and
saves image-domain visualizations at EVERY stage for use in papers/reports.

Pipeline Stages (what happens to the image at each step):
  Stage 1 → Input Image          — raw face (original pixels)
  Stage 2 → PAD Analysis         — FFT spectrum + LBP texture map derived from the face
  Stage 3 → Feature Extraction   — 7×7 patch grid overlay + feature magnitude heatmap
  Stage 4 → SLERP Interpolation  — feature space rotation shown as image grids
  Stage 5 → Feature Dropping     — dropout mask applied to feature image
  Stage 6 → Final Output         — original vs protected side-by-side summary

Usage:
  python generate_report_images.py --img   samples/face.jpg      # real image file
  python generate_report_images.py --camera                       # live webcam capture
  python generate_report_images.py --camera --camera-id 1         # external camera index
  python generate_report_images.py --demo                         # synthetic face

Webcam controls:
  SPACE / ENTER  →  capture and run pipeline
  ESC   / Q      →  quit without capturing
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.antispoof.detector import AntiSpoofDetector
from modules.antispoof.frequency_analysis import compute_fft_spectrum
from modules.antispoof.texture_analysis import compute_lbp, rgb_to_ycrcb
from verify_with_antispoof import slerp_encrypt, extract_face_features, GROUP_SIZE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0D1117"        # page background
PANEL    = "#161B22"        # axes background
ACCENT   = "#00FFA3"        # green accent
ACCENT2  = "#FF6B6B"        # red accent
ACCENT3  = "#64DFDF"        # cyan accent
TITLE_C  = "white"
LABEL_C  = "#A0C4FF"
TICK_C   = "#6B7280"
SPINE_C  = "#21262D"


def _fig_style(fig, axes_flat):
    """Apply uniform dark style to a figure."""
    fig.patch.set_facecolor(BG)
    for ax in axes_flat:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(SPINE_C)
        ax.tick_params(colors=TICK_C, labelsize=7)
        ax.xaxis.label.set_color(TICK_C)
        ax.yaxis.label.set_color(TICK_C)


def _norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a float array to uint8 [0,255]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


def _apply_cmap(arr_2d: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """Convert a 2-D float array → (H,W,3) uint8 via matplotlib colormap."""
    norm = _norm_to_uint8(arr_2d).astype(np.float32) / 255.0
    rgb  = (cm.get_cmap(cmap_name)(norm)[:, :, :3] * 255).astype(np.uint8)
    return rgb


def _save_fig(fig, path: str):
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    kb = os.path.getsize(path) / 1024
    print(f"  ✔  {os.path.basename(path):50s}  ({kb:.0f} KB)")


def _stage_title(ax, num: int, text: str):
    """Put a numbered badge + title above an axes."""
    ax.set_title(f"  {text}", color=LABEL_C, fontsize=10, loc="left", pad=6)


# ─────────────────────────────────────────────────────────────────────────────
# Webcam capture
# ─────────────────────────────────────────────────────────────────────────────

def capture_from_webcam(camera_id: int = 0, save_path: str = None) -> np.ndarray:
    """
    Opens a live webcam preview. Press SPACE / ENTER to capture.
    Returns RGB numpy array.
    """
    try:
        import cv2
    except ImportError:
        print("❌  OpenCV not found.  pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"❌  Cannot open camera id={camera_id}. Try --camera-id 1.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n📷  Webcam preview open.")
    print("    SPACE / ENTER  →  capture     ESC / Q  →  quit\n")

    captured = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (frame.shape[1], 52), (13, 17, 23), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "SPACE/ENTER: Capture    ESC/Q: Quit",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 163), 2)
        cv2.imshow("SlerpFace — Webcam Capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(' '), 13):
            captured = frame.copy()
            print("  ✔  Frame captured!")
            break
        elif key in (27, ord('q'), ord('Q')):
            print("  ✘  Cancelled.")
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)

    cap.release(); cv2.destroyAllWindows()
    if captured is None:
        print("❌  No frame captured."); sys.exit(1)

    rgb = cv2.cvtColor(captured, cv2.COLOR_BGR2RGB)
    if save_path:
        cv2.imwrite(save_path, captured)
        print(f"  ✔  Saved webcam image → {save_path}")
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic demo face
# ─────────────────────────────────────────────────────────────────────────────

def make_synthetic_face(size: int = 256) -> np.ndarray:
    y_g, x_g = np.ogrid[:size, :size]
    cy, cx   = size // 2, size // 2
    r        = np.sqrt((x_g - cx)**2 + (y_g - cy)**2)
    radial   = np.exp(-r**2 / (2*(size*0.35)**2))
    skin     = np.zeros((size, size, 3), dtype=np.float32)
    skin[:,:,0] = 175*(0.6+0.4*radial) + np.random.normal(0, 4, (size,size))
    skin[:,:,1] = 128*(0.6+0.4*radial) + np.random.normal(0, 3, (size,size))
    skin[:,:,2] =  98*(0.6+0.4*radial) + np.random.normal(0, 2.5,(size,size))
    bg = np.full((size, size, 3), 40, dtype=np.float32)
    face_mask = r < size * 0.45
    for c in range(3):
        bg[:,:,c] = np.where(face_mask, skin[:,:,c], bg[:,:,c])
    # Eyes
    for ey, ex in [(cy-size//7, cx-size//8), (cy-size//7, cx+size//8)]:
        em = ((x_g-ex)**2+(y_g-ey)**2) < (size//14)**2
        bg[em] = [55,35,25]
        pm = ((x_g-ex)**2+(y_g-ey)**2) < (size//28)**2
        bg[pm] = [10,8,8]
    # Mouth
    lm = ((x_g-cx)**2+(y_g-(cy+size//8))**2) < (size//12)**2
    bg[lm,0]=155; bg[lm,1]=55; bg[lm,2]=55
    return np.clip(bg, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Input Image
# ─────────────────────────────────────────────────────────────────────────────

def stage1_input(img_orig: np.ndarray, out_dir: str):
    """Save the raw input face image as Stage 1."""
    h, w = img_orig.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    _fig_style(fig, [ax])
    fig.suptitle("Stage 1 — Input Face Image", color=TITLE_C,
                 fontsize=13, fontweight="bold")
    ax.imshow(img_orig)
    ax.set_title(f"Original face  [{w}×{h} px]", color=LABEL_C, fontsize=10)
    ax.axis("off")
    _save_fig(fig, os.path.join(out_dir, "stage1_input.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — PAD (Presentation Attack Detection) image visuals
# ─────────────────────────────────────────────────────────────────────────────

def stage2_pad(img_rgb: np.ndarray, out_dir: str):
    """
    Show actual image-domain representations used by the PAD module:
      - Grayscale conversion
      - 2D FFT power spectrum (frequency domain image)
      - LBP texture map (spatial texture image)
      - YCrCb chrominance channels (Cr and Cb as images)
      - Liveness verdict overlay
    """
    img_small = np.array(Image.fromarray(img_rgb).resize((224, 224)))
    gray = (0.299*img_small[:,:,0] + 0.587*img_small[:,:,1]
            + 0.114*img_small[:,:,2]).astype(np.uint8)

    # FFT spectrum image
    spectrum = compute_fft_spectrum(gray.astype(np.float64))
    fft_img  = _norm_to_uint8(spectrum)
    fft_rgb  = _apply_cmap(spectrum, "inferno")

    # LBP texture map
    lbp = compute_lbp(gray, radius=1)
    lbp_rgb = _apply_cmap(lbp.astype(np.float32), "hot")

    # YCrCb chrominance
    ycrcb = rgb_to_ycrcb(img_small)
    cr_img = _apply_cmap(ycrcb[:,:,1], "RdBu_r")
    cb_img = _apply_cmap(ycrcb[:,:,2], "PuBu")

    # PAD verdict
    detector = AntiSpoofDetector(threshold=0.60)
    result   = detector.evaluate(img_rgb)
    verdict_color = ACCENT if result.is_live else ACCENT2
    verdict_txt   = f"{'✅ GENUINE — LIVE FACE' if result.is_live else '🚨 SPOOF DETECTED'}   (score {result.liveness_score:.3f})"

    fig = plt.figure(figsize=(18, 7))
    _fig_style(fig, [])
    fig.suptitle("Stage 2 — PAD: Presentation Attack Detection\n"
                 "FFT Frequency Analysis  +  LBP Texture Analysis  +  YCrCb Chrominance",
                 color=TITLE_C, fontsize=13, fontweight="bold")

    gs = GridSpec(2, 5, figure=fig, wspace=0.25, hspace=0.35)

    panels = [
        (gs[0, 0], img_small,     "Original Face\n(input to PAD)", "gray"),
        (gs[0, 1], np.stack([gray]*3,axis=2), "Grayscale\n(FFT input)", "gray"),
        (gs[0, 2], fft_rgb,        "2D FFT Spectrum\n(frequency domain image)", "inferno"),
        (gs[1, 0], img_small,     "Original Face\n(texture input)", "gray"),
        (gs[1, 1], np.stack([gray]*3,axis=2), "Grayscale\n(LBP input)", "gray"),
        (gs[1, 2], lbp_rgb,        "LBP Texture Map\n(spatial micro-texture image)", "hot"),
        (gs[0, 3], cr_img,         "YCrCb  Cr Channel\n(red-chroma image)", "RdBu_r"),
        (gs[1, 3], cb_img,         "YCrCb  Cb Channel\n(blue-chroma image)", "PuBu"),
    ]

    for spec, data, title, _ in panels:
        ax = fig.add_subplot(spec)
        _fig_style(fig, [ax])
        ax.imshow(data)
        ax.set_title(title, color=LABEL_C, fontsize=9)
        ax.axis("off")

    # Verdict panel (right column, full height)
    ax_v = fig.add_subplot(gs[:, 4])
    _fig_style(fig, [ax_v])
    ax_v.axis("off")
    ax_v.set_title("PAD Verdict", color=LABEL_C, fontsize=10)
    metrics = [
        ("Liveness Score",       f"{result.liveness_score:.4f}"),
        ("Verdict",              result.verdict.split('(')[0].strip()),
        ("Risk Level",           result.risk_level),
        ("Moiré Score",          f"{result.diagnostics['moire_score']:.4f}"),
        ("High-Freq Energy",     f"{result.diagnostics['high_freq_ratio']:.4f}"),
        ("LBP Texture Entropy",  f"{result.diagnostics['lbp_entropy']:.4f}"),
        ("Chrominance Diversity",f"{result.diagnostics['chrominance_diversity']:.2f}"),
        ("Glare Ratio",          f"{result.diagnostics['glare_ratio']:.4f}"),
    ]
    y = 0.92
    for k, v in metrics:
        ax_v.text(0.04, y, k+":", color=TICK_C, fontsize=8.5, va="top",
                  transform=ax_v.transAxes)
        ax_v.text(0.96, y, v, color=ACCENT3, fontsize=8.5, va="top",
                  ha="right", fontweight="bold", transform=ax_v.transAxes)
        y -= 0.09
    # Big verdict box at bottom
    rect = plt.Rectangle((0, 0.0), 1, 0.14, facecolor=verdict_color+"22",
                          edgecolor=verdict_color, linewidth=1.5,
                          transform=ax_v.transAxes, clip_on=False)
    ax_v.add_patch(rect)
    ax_v.text(0.5, 0.07, result.verdict.split()[0],
              color=verdict_color, fontsize=11, fontweight="bold",
              ha="center", va="center", transform=ax_v.transAxes)

    _save_fig(fig, os.path.join(out_dir, "stage2_pad_detection.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def stage3_feature_extract(img_112: np.ndarray, group_feat: np.ndarray, out_dir: str):
    """
    Show the face → feature representation transformation:
      - Face with 7×7 patch grid overlaid
      - Each patch's feature magnitude → heatmap image
      - Feature matrix as an image (49×16 → visualized as image pixels)
    """
    patch_magnitudes = np.linalg.norm(group_feat, axis=1).reshape(7, 7)
    mag_img          = _apply_cmap(patch_magnitudes, "plasma")
    mag_large        = np.array(Image.fromarray(mag_img).resize((112, 112),
                                Image.NEAREST))

    # Feature matrix as image: (49,16) → mapped to colorspace
    feat_img = _apply_cmap(group_feat, "coolwarm")         # (49,16,3)
    feat_wide = np.array(Image.fromarray(feat_img).resize((320, 112),
                          Image.NEAREST))

    fig = plt.figure(figsize=(17, 5))
    _fig_style(fig, [])
    fig.suptitle("Stage 3 — Feature Extraction  (IR-50 Backbone / Spatial Patch Descriptor)",
                 color=TITLE_C, fontsize=13, fontweight="bold")

    gs = GridSpec(1, 4, figure=fig, wspace=0.3)

    # Col 0: original face
    ax0 = fig.add_subplot(gs[0])
    _fig_style(fig, [ax0])
    ax0.imshow(img_112)
    ax0.set_title("Input Face\n(112×112, model input size)", color=LABEL_C, fontsize=9)
    ax0.axis("off")

    # Col 1: face + 7x7 grid overlay
    ax1 = fig.add_subplot(gs[1])
    _fig_style(fig, [ax1])
    ax1.imshow(img_112, alpha=0.85)
    step = 112 // 7
    for i in range(1, 7):
        ax1.axhline(i*step - 0.5, color=ACCENT, linewidth=1.1, alpha=0.85)
        ax1.axvline(i*step - 0.5, color=ACCENT, linewidth=1.1, alpha=0.85)
    for r in range(7):
        for c in range(7):
            ax1.text(c*step + step//2, r*step + step//2,
                     f"{r*7+c}", color="white", fontsize=5,
                     ha="center", va="center",
                     bbox=dict(boxstyle="round,pad=0.1", fc="#00000066", ec="none"))
    ax1.set_title("7×7 Spatial Patch Grid\n(49 patches, 16-D descriptor each)",
                  color=LABEL_C, fontsize=9)
    ax1.axis("off")

    # Col 2: patch magnitude heatmap image
    ax2 = fig.add_subplot(gs[2])
    _fig_style(fig, [ax2])
    im = ax2.imshow(patch_magnitudes, cmap="plasma", aspect="equal")
    for r in range(7):
        for c in range(7):
            ax2.text(c, r, f"{patch_magnitudes[r,c]:.2f}",
                     color="white", fontsize=6, ha="center", va="center",
                     fontweight="bold")
    ax2.set_title("Feature Magnitude Heatmap\n(L2 norm per patch → image domain)",
                  color=LABEL_C, fontsize=9)
    ax2.set_xticks(range(7)); ax2.set_yticks(range(7))
    ax2.tick_params(colors=TICK_C, labelsize=6)
    plt.colorbar(im, ax=ax2, shrink=0.8, pad=0.02).ax.tick_params(colors=TICK_C, labelsize=6)

    # Col 3: full feature matrix as image
    ax3 = fig.add_subplot(gs[3])
    _fig_style(fig, [ax3])
    im3 = ax3.imshow(group_feat.T, cmap="coolwarm", aspect="auto",
                     vmin=-group_feat.std()*2, vmax=group_feat.std()*2)
    ax3.set_title("Feature Matrix as Image\n(16 dims × 49 groups → pixel grid)",
                  color=LABEL_C, fontsize=9)
    ax3.set_xlabel("Patch Group (0–48)", color=TICK_C, fontsize=8)
    ax3.set_ylabel("Dimension (0–15)", color=TICK_C, fontsize=8)
    ax3.tick_params(colors=TICK_C, labelsize=6)
    plt.colorbar(im3, ax=ax3, shrink=0.8, pad=0.02).ax.tick_params(colors=TICK_C, labelsize=6)

    _save_fig(fig, os.path.join(out_dir, "stage3_feature_extraction.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — SLERP Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def stage4_slerp(group_feat: np.ndarray, alpha: float, out_dir: str):
    """
    Show how SLERP transforms the feature space image:
      - Original feature matrix image
      - Random key image (noise on hypersphere)
      - Slerp-interpolated result image
      - Difference image (what changed)
    """
    np.random.seed(42)
    feat  = group_feat.reshape(-1, GROUP_SIZE)
    key   = np.random.randn(*feat.shape)

    nf = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    nk = np.linalg.norm(key,  axis=1, keepdims=True) + 1e-8
    f_n = feat / nf;  k_n = key / nk

    dot   = np.clip((f_n * k_n).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(dot).reshape(-1, 1)
    sin_t = np.sin(theta) + 1e-8

    def slerp_at(a):
        return np.sin((1-a)*theta)/sin_t * f_n + np.sin(a*theta)/sin_t * k_n

    enc  = slerp_at(alpha)
    diff = enc - f_n                        # change introduced by slerp

    # Colour-map all as image grids  (shape: 49×16)
    vmax  = max(np.abs(f_n).max(), np.abs(enc).max(), 1.0)
    def to_img(arr, cmap="coolwarm", vmax=vmax):
        norm = np.clip(arr / vmax, -1, 1)
        norm = (norm + 1) / 2               # → [0,1]
        return (cm.get_cmap(cmap)(norm)[:,:,:3] * 255).astype(np.uint8)

    f_img    = to_img(f_n.T)
    k_img    = to_img(k_n.T)
    enc_img  = to_img(enc.T)
    diff_img = to_img(diff.T, cmap="RdBu_r", vmax=np.abs(diff).max()+1e-8)

    # Slerp trajectory curve over multiple alphas
    alphas = np.linspace(0, 1, 51)
    cosims = []
    for a in alphas:
        s = slerp_at(a)
        sf = s.ravel(); ff = f_n.ravel()
        cosims.append(float(np.dot(ff,sf)/(np.linalg.norm(ff)*np.linalg.norm(sf)+1e-8)))

    fig = plt.figure(figsize=(19, 6))
    _fig_style(fig, [])
    fig.suptitle(f"Stage 4 — SLERP Interpolation  (α = {alpha:.2f}  →  feature rotation on hypersphere)",
                 color=TITLE_C, fontsize=13, fontweight="bold")

    gs = GridSpec(2, 5, figure=fig, wspace=0.3, hspace=0.45)

    def img_ax(spec, img, title):
        ax = fig.add_subplot(spec)
        _fig_style(fig, [ax])
        ax.imshow(img, aspect="auto")
        ax.set_title(title, color=LABEL_C, fontsize=9)
        ax.set_xlabel("Group (0–48)", color=TICK_C, fontsize=7)
        ax.set_ylabel("Dim (0–15)", color=TICK_C, fontsize=7)
        ax.tick_params(labelsize=6)
        return ax

    img_ax(gs[0, 0], f_img,
           "Original Feature Grid\n(face descriptor as image)")
    img_ax(gs[0, 1], k_img,
           "Random Key Grid\n(noise vector on hypersphere)")
    img_ax(gs[0, 2], enc_img,
           f"SLERP-Rotated Grid  (α={alpha})\n(encrypted feature image)")
    img_ax(gs[0, 3], diff_img,
           "Difference Image\n(what SLERP changed)")

    # Trajectory plot (top-right)
    ax_t = fig.add_subplot(gs[0, 4])
    _fig_style(fig, [ax_t])
    ax_t.plot(alphas, cosims, color=ACCENT2, linewidth=1.8)
    ax_t.axvline(alpha, color=ACCENT, linestyle="--", linewidth=1.4,
                 label=f"α={alpha}")
    ax_t.axhline(cosims[int(alpha*50)], color="#FFD700", linestyle=":", linewidth=1.1)
    ax_t.set_title("SLERP Trajectory\n(cos-sim vs rotation α)", color=LABEL_C, fontsize=9)
    ax_t.set_xlabel("Alpha (rotation)", color=TICK_C, fontsize=8)
    ax_t.set_ylabel("Cosine similarity\nwith original", color=TICK_C, fontsize=8)
    ax_t.legend(fontsize=7, facecolor=PANEL, labelcolor="white")
    ax_t.set_xlim(0, 1)

    # Bottom row: show alpha=0, 0.5, 1.0 as images to visualise progression
    for col, a_val in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        sa = slerp_at(a_val)
        ax_b = img_ax(gs[1, col], to_img(sa.T),
                      f"α = {a_val:.2f}\n({'original' if a_val==0 else 'key' if a_val==1 else 'interpolated'})")

    _save_fig(fig, os.path.join(out_dir, "stage4_slerp_interpolation.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Feature Dropping
# ─────────────────────────────────────────────────────────────────────────────

def stage5_drop(group_feat: np.ndarray, alpha: float, drop_rate: float, out_dir: str):
    """
    Show feature dropout as image-domain transformations:
      - Pre-SLERP feature image
      - Post-SLERP (pre-drop) image
      - Dropout mask image (green=kept, black=zeroed)
      - Post-dropout final template image
      - Difference (what was erased)
    """
    np.random.seed(42)
    feat = group_feat.reshape(-1, GROUP_SIZE)
    key  = np.random.randn(*feat.shape)
    nf   = np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8
    nk   = np.linalg.norm(key,  axis=1, keepdims=True) + 1e-8
    f_n  = feat / nf;  k_n = key / nk
    dot  = np.clip((f_n*k_n).sum(axis=1), -1, 1)
    theta= np.arccos(dot).reshape(-1,1)
    sin_t= np.sin(theta) + 1e-8
    enc  = (np.sin((1-alpha)*theta)/sin_t*f_n +
            np.sin(alpha*theta)/sin_t*k_n).copy()      # post-SLERP, pre-drop

    n_drop   = int(enc.size * drop_rate)
    drop_idx = np.random.choice(enc.size, n_drop, replace=False)
    mask_2d  = np.ones(enc.shape, dtype=np.float32)
    mask_2d.ravel()[drop_idx] = 0.0

    enc_drop = enc.copy()
    enc_drop.ravel()[drop_idx] = 0.0
    erased = enc - enc_drop                             # what was zeroed

    vmax = max(np.abs(enc).max(), 1.0)
    def to_img(arr, vmax=vmax, cmap="coolwarm"):
        n = np.clip(arr/vmax, -1, 1)
        n = (n+1)/2
        return (cm.get_cmap(cmap)(n)[:,:,:3]*255).astype(np.uint8)

    # Mask as green/black image — shape (49, 16), then shown transposed via imshow
    mask_display = np.zeros((GROUP_SIZE, feat.shape[0], 3), dtype=np.uint8)  # (16, 49, 3)
    for dim in range(GROUP_SIZE):
        for grp in range(feat.shape[0]):
            if mask_2d[grp, dim] > 0:
                mask_display[dim, grp] = [0, 255, 163]   # kept → green
            else:
                mask_display[dim, grp] = [180, 40, 40]   # dropped → red

    fig = plt.figure(figsize=(20, 5.5))
    _fig_style(fig, [])
    fig.suptitle(f"Stage 5 — Feature Dropping  (β = {drop_rate:.0%} dimensions irreversibly zeroed)",
                 color=TITLE_C, fontsize=13, fontweight="bold")

    gs = GridSpec(1, 5, figure=fig, wspace=0.28)
    labels = [
        (feat.T,        "coolwarm", "Original Feature Image\n(before any processing)"),
        (enc.T,         "coolwarm", f"After SLERP (α={alpha})\n(pre-dropout image)"),
        (None,          None,       f"Dropout Mask Image\n({drop_rate:.0%} red = erased)"),
        (enc_drop.T,    "coolwarm", "After Dropout\n(protected template image)"),
        (erased.T,      "RdBu_r",   "Erased Dimensions\n(difference image)"),
    ]

    for i, (arr, cmap, title) in enumerate(labels):
        ax = fig.add_subplot(gs[i])
        _fig_style(fig, [ax])
        if arr is None:
            ax.imshow(mask_display, aspect="auto")
        else:
            vv = max(np.abs(arr).max(), 1e-6)
            n = np.clip(arr/vv, -1, 1); n = (n+1)/2
            ax.imshow((cm.get_cmap(cmap)(n)[:,:,:3]*255).astype(np.uint8),
                      aspect="auto")

        ax.set_title(title, color=LABEL_C, fontsize=9)
        ax.set_xlabel("Group (0–48)", color=TICK_C, fontsize=7)
        ax.set_ylabel("Dim (0–15)",   color=TICK_C, fontsize=7)
        ax.tick_params(labelsize=6)

        # Annotate dropped count on mask panel
        if arr is None:
            ax.text(0.5, -0.14,
                    f"Kept: {int((1-drop_rate)*enc.size)}  |  Dropped: {n_drop}",
                    color=TICK_C, fontsize=8, ha="center",
                    transform=ax.transAxes)

    _save_fig(fig, os.path.join(out_dir, "stage5_feature_dropping.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Final Output
# ─────────────────────────────────────────────────────────────────────────────

def stage6_output(img_orig: np.ndarray, img_112: np.ndarray,
                  group_feat: np.ndarray, encrypted: np.ndarray,
                  alpha: float, drop_rate: float, out_dir: str):
    """
    Final summary panel:
      - Original face image
      - All intermediate feature images stacked
      - Final encrypted template image
      - Security statistics
    """
    vmax = max(np.abs(group_feat).max(), np.abs(encrypted).max(), 1e-6)
    def feat_to_img(arr, cmap="coolwarm"):
        n = np.clip(arr.T/vmax, -1, 1); n=(n+1)/2
        return (cm.get_cmap(cmap)(n)[:,:,:3]*255).astype(np.uint8)

    raw_img  = feat_to_img(group_feat.reshape(-1, GROUP_SIZE))
    enc_img  = feat_to_img(encrypted)

    nonzero  = np.count_nonzero(encrypted)
    total    = encrypted.size
    raw_norm = float(np.linalg.norm(group_feat.ravel()))
    enc_norm = float(np.linalg.norm(encrypted.ravel()))
    rf = group_feat.ravel(); ef = encrypted.ravel()
    cosim = float(np.dot(rf,ef)/(np.linalg.norm(rf)*np.linalg.norm(ef)+1e-8))

    fig = plt.figure(figsize=(20, 7))
    _fig_style(fig, [])
    fig.suptitle("Stage 6 — Final Protected Template  (SlerpFace Cancelable Biometric Output)",
                 color=TITLE_C, fontsize=13, fontweight="bold")

    gs = GridSpec(2, 5, figure=fig, wspace=0.32, hspace=0.40)

    # Original face (spans 2 rows)
    ax_face = fig.add_subplot(gs[:, 0])
    _fig_style(fig, [ax_face])
    ax_face.imshow(np.array(Image.fromarray(img_orig).resize((224,224))))
    ax_face.set_title("Input Face\n(original pixels)", color=LABEL_C, fontsize=9)
    ax_face.axis("off")

    # Raw feature image (top)
    ax_r = fig.add_subplot(gs[0, 1:3])
    _fig_style(fig, [ax_r])
    ax_r.imshow(raw_img, aspect="auto")
    ax_r.set_title("Original Feature Image  (49 groups × 16 dims)\n"
                   "— biometric identity stored here —",
                   color=LABEL_C, fontsize=9)
    ax_r.set_xlabel("Group (0–48)", color=TICK_C, fontsize=7)
    ax_r.set_ylabel("Dim (0–15)", color=TICK_C, fontsize=7)
    ax_r.tick_params(labelsize=6)

    # Encrypted template image (bottom)
    ax_e = fig.add_subplot(gs[1, 1:3])
    _fig_style(fig, [ax_e])
    ax_e.imshow(enc_img, aspect="auto")
    ax_e.set_title(f"Protected Template Image  (α={alpha}, β={drop_rate:.0%} dropped)\n"
                   "— SLERP-rotated + irreversibly shredded —",
                   color=LABEL_C, fontsize=9)
    ax_e.set_xlabel("Group (0–48)", color=TICK_C, fontsize=7)
    ax_e.set_ylabel("Dim (0–15)", color=TICK_C, fontsize=7)
    ax_e.tick_params(labelsize=6)

    # Difference image (top-right)
    diff = group_feat.reshape(-1, GROUP_SIZE) - encrypted
    diff_img = feat_to_img(diff, cmap="RdBu_r")
    ax_diff = fig.add_subplot(gs[0, 3])
    _fig_style(fig, [ax_diff])
    ax_diff.imshow(diff_img, aspect="auto")
    ax_diff.set_title("Difference Image\n(original − protected)",
                      color=LABEL_C, fontsize=9)
    ax_diff.tick_params(labelsize=6)

    # Zero mask image — build per-element to avoid shape mismatch
    enc_reshaped = encrypted.reshape(-1, GROUP_SIZE)  # (49, 16)
    zmask_display = np.zeros((GROUP_SIZE, enc_reshaped.shape[0], 3), dtype=np.uint8)  # (16, 49, 3)
    for dim in range(GROUP_SIZE):
        for grp in range(enc_reshaped.shape[0]):
            if enc_reshaped[grp, dim] == 0:
                zmask_display[dim, grp] = [180, 40, 40]   # zeroed → red
            else:
                zmask_display[dim, grp] = [0, 255, 163]   # active → green
    ax_mask = fig.add_subplot(gs[1, 3])
    _fig_style(fig, [ax_mask])
    ax_mask.imshow(zmask_display, aspect="auto")
    ax_mask.set_title(f"Zeroed Dimensions Mask\n(red={drop_rate:.0%} shredded)",
                      color=LABEL_C, fontsize=9)
    ax_mask.tick_params(labelsize=6)

    # Stats panel (right column, full height)
    ax_s = fig.add_subplot(gs[:, 4])
    _fig_style(fig, [ax_s])
    ax_s.axis("off")
    ax_s.set_title("Security Report", color=LABEL_C, fontsize=10)
    rows = [
        ("Template shape",       f"{encrypted.shape}"),
        ("Total dimensions",     f"{total}"),
        ("Active (non-zero)",    f"{nonzero}"),
        ("Shredded dims",        f"{total-nonzero}  ({drop_rate:.0%})"),
        ("", ""),
        ("Raw feature L2",       f"{raw_norm:.4f}"),
        ("Encrypted L2",         f"{enc_norm:.4f}"),
        ("Self-similarity",      f"{cosim:.4f}"),
        ("SLERP α",              f"{alpha}"),
        ("Dropout β",            f"{drop_rate:.0%}"),
        ("", ""),
        ("Reversibility",        "NONE ✅"),
        ("Cross-system linkage", "NONE ✅"),
        ("Security scope",       "End-to-End"),
    ]
    y = 0.97
    for k, v in rows:
        if k == "":
            y -= 0.03; continue
        ax_s.text(0.03, y, k+":", color=TICK_C, fontsize=8.5, va="top",
                  transform=ax_s.transAxes)
        ax_s.text(0.97, y, v, color=ACCENT3, fontsize=8.5, va="top",
                  ha="right", fontweight="bold", transform=ax_s.transAxes)
        y -= 0.065

    _save_fig(fig, os.path.join(out_dir, "stage6_final_output.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Summary strip — all 6 stages side by side
# ─────────────────────────────────────────────────────────────────────────────

def summary_strip(out_dir: str):
    stage_files = [
        ("stage1_input.png",           "Stage 1\nInput"),
        ("stage2_pad_detection.png",   "Stage 2\nPAD"),
        ("stage3_feature_extraction.png","Stage 3\nFeatures"),
        ("stage4_slerp_interpolation.png","Stage 4\nSLERP"),
        ("stage5_feature_dropping.png","Stage 5\nDropping"),
        ("stage6_final_output.png",    "Stage 6\nOutput"),
    ]
    imgs, titles, badges = [], [], []
    for fn, title in stage_files:
        p = os.path.join(out_dir, fn)
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            imgs.append(None)
        titles.append(title)
        badges.append(fn[5])   # stage number char

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(n*4.5, 3.5))
    _fig_style(fig, axes)
    fig.suptitle("SlerpFace Pipeline — Complete Flow (All Stages)",
                 color=TITLE_C, fontsize=15, fontweight="bold", y=1.03)

    for i, (ax, img, title) in enumerate(zip(axes, imgs, titles)):
        ax.set_facecolor(PANEL)
        if img:
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color=TICK_C, transform=ax.transAxes)
        ax.set_title(title, color=LABEL_C, fontsize=10, fontweight="bold")
        ax.axis("off")
        # Number badge
        circ = plt.Circle((0.09, 0.88), 0.085, color=ACCENT+"44",
                           ec=ACCENT, lw=1.5, transform=ax.transAxes,
                           clip_on=False, zorder=5)
        ax.add_patch(circ)
        ax.text(0.09, 0.88, str(i+1), color=ACCENT, fontsize=10,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes, zorder=6)

    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "pipeline_summary_strip.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pass a face image through the SlerpFace pipeline and save"
                    " image-domain visuals at every stage for your report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python generate_report_images.py --img   face.jpg
  python generate_report_images.py --camera
  python generate_report_images.py --camera --camera-id 1
  python generate_report_images.py --demo
"""
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--img",    metavar="PATH",
                     help="Path to a face image (jpg/png/bmp/webp …)")
    src.add_argument("--camera", action="store_true",
                     help="Capture from webcam (live preview)")
    src.add_argument("--demo",   action="store_true",
                     help="Use a synthetic face (no camera needed)")
    parser.add_argument("--camera-id", type=int, default=0, metavar="N",
                        help="Webcam device index (default 0)")
    parser.add_argument("--out",   default="report_images",
                        help="Output folder (default: report_images/)")
    parser.add_argument("--alpha", type=float, default=0.90,
                        help="SLERP alpha — rotation strength (default 0.90)")
    parser.add_argument("--drop",  type=float, default=0.50,
                        help="Dropout rate β (default 0.50 = 50%%)")
    args = parser.parse_args()

    out_dir = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f"  SlerpFace Pipeline Report Image Generator")
    print(f"  Output  : {out_dir}")
    print(f"  SLERP α : {args.alpha}   Dropout β : {args.drop:.0%}")
    print(f"{'='*65}\n")

    # ── Load image ──────────────────────────────────────────────────────────
    if args.demo:
        img_orig = make_synthetic_face(256)
        img_path = None
        print("🎨  Using synthetic demo face\n")
    elif args.camera:
        save_path = os.path.join(out_dir, "webcam_capture.jpg")
        img_orig  = capture_from_webcam(camera_id=args.camera_id,
                                        save_path=save_path)
        img_path  = save_path
        print(f"📷  Webcam image saved → {save_path}\n")
    else:
        img_path = os.path.abspath(args.img)
        if not os.path.exists(img_path):
            parser.error(f"File not found: {img_path}")
        img_orig = np.array(Image.open(img_path).convert("RGB"))
        print(f"🖼   Loaded: {img_path}\n")

    # Resize to 112×112 for feature extraction
    img_112 = np.array(Image.fromarray(img_orig).resize((112, 112)))

    # Save temp file for extract_face_features (needs a path)
    tmp = os.path.join(out_dir, "_tmp.jpg")
    Image.fromarray(img_112).save(tmp)

    # ── Run all stages ──────────────────────────────────────────────────────
    print("── Stage 1 ─ Input Image ─────────────────────────────────────────")
    stage1_input(img_orig, out_dir)

    print("\n── Stage 2 ─ PAD (Presentation Attack Detection) ─────────────────")
    stage2_pad(img_orig, out_dir)

    print("\n── Stage 3 ─ Feature Extraction ───────────────────────────────────")
    group_feat, vec_feat = extract_face_features(tmp)
    stage3_feature_extract(img_112, group_feat, out_dir)

    print("\n── Stage 4 ─ SLERP Interpolation ──────────────────────────────────")
    stage4_slerp(group_feat, alpha=args.alpha, out_dir=out_dir)

    print("\n── Stage 5 ─ Feature Dropping ─────────────────────────────────────")
    stage5_drop(group_feat, alpha=args.alpha, drop_rate=args.drop, out_dir=out_dir)

    print("\n── Stage 6 ─ Final Output ──────────────────────────────────────────")
    encrypted = slerp_encrypt(group_feat, alpha=args.alpha,
                              drop_rate=args.drop, seed=42)
    stage6_output(img_orig, img_112, group_feat, encrypted,
                  alpha=args.alpha, drop_rate=args.drop, out_dir=out_dir)

    print("\n── Summary Strip ───────────────────────────────────────────────────")
    summary_strip(out_dir)

    try: os.remove(tmp)
    except Exception: pass

    print(f"\n{'='*65}")
    print(f"  ✅  ALL STAGES COMPLETE")
    print(f"  📂  {out_dir}")
    print(f"{'='*65}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".png"):
            kb = os.path.getsize(os.path.join(out_dir,f))/1024
            print(f"  📷  {f:<55} ({kb:.0f} KB)")


if __name__ == "__main__":
    main()
