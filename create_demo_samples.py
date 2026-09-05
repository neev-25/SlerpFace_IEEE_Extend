"""
create_demo_samples.py
=======================
Generates synthetic demo face presentations in the 'samples/' directory:
  - genuine_personA_1.jpg: Live person A (Reference enrollment)
  - genuine_personA_2.jpg: Live person A (Verification probe)
  - genuine_personB_1.jpg: Live person B (Different identity)
  - spoof_print_attack.jpg: Printed photo presentation attack
  - spoof_screen_replay.jpg: Smartphone screen replay presentation attack
"""

import os
import sys
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

y, x = np.ogrid[:112, :112]
cy, cx = 56, 56
radial = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 42 ** 2))

# ── 1. Person A (Reference) ──────────────────────────────────────────────────
pA = np.zeros((112, 112, 3), dtype=np.float32)
pA[:, :, 0] = 190 * (0.65 + 0.35 * radial) + np.random.normal(0, 3, (112, 112))
pA[:, :, 1] = 145 * (0.65 + 0.35 * radial) + np.random.normal(0, 3, (112, 112))
pA[:, :, 2] = 115 * (0.65 + 0.35 * radial) + np.random.normal(0, 2.5, (112, 112))
img_pA1 = np.clip(pA, 0, 255).astype(np.uint8)
Image.fromarray(img_pA1).save(os.path.join(OUTPUT_DIR, "genuine_personA_1.jpg"))

# ── 2. Person A (Slight lighting variation) ──────────────────────────────────
pA2 = pA * 0.96 + np.random.normal(0, 2, (112, 112, 3))
img_pA2 = np.clip(pA2, 0, 255).astype(np.uint8)
Image.fromarray(img_pA2).save(os.path.join(OUTPUT_DIR, "genuine_personA_2.jpg"))

# ── 3. Person B (Different skin tone & shape) ────────────────────────────────
pB = np.zeros((112, 112, 3), dtype=np.float32)
pB[:, :, 0] = 160 * (0.60 + 0.40 * radial) + np.random.normal(0, 4, (112, 112))
pB[:, :, 1] = 110 * (0.60 + 0.40 * radial) + np.random.normal(0, 3, (112, 112))
pB[:, :, 2] = 85 * (0.60 + 0.40 * radial) + np.random.normal(0, 2, (112, 112))
img_pB = np.clip(pB, 0, 255).astype(np.uint8)
Image.fromarray(img_pB).save(os.path.join(OUTPUT_DIR, "genuine_personB_1.jpg"))

# ── 4. Print Attack (Paper photo of Person A) ────────────────────────────────
print_spoof = pA.copy() * 0.82 + 25.0
dot_grid = (np.sin(x * 1.5) * np.cos(y * 1.5)) * 14.0
print_spoof += dot_grid[:, :, None]
mean_lum = np.mean(print_spoof, axis=2, keepdims=True)
print_spoof = print_spoof * 0.72 + mean_lum * 0.28
img_print = np.clip(print_spoof, 0, 255).astype(np.uint8)
Image.fromarray(img_print).save(os.path.join(OUTPUT_DIR, "spoof_print_attack.jpg"))

# ── 5. Screen Replay Attack (Phone screen of Person A) ───────────────────────
screen_spoof = pA.copy()
moire = (np.sin(x * 0.8 + y * 0.4) + np.cos(x * 0.4 - y * 0.8)) * 22.0
screen_spoof[:, :, 0] += moire * 0.8
screen_spoof[:, :, 1] += moire * 1.1
screen_spoof[:, :, 2] += moire * 1.5
glare = np.exp(-((x - 82) ** 2 + (y - 28) ** 2) / (2 * 12 ** 2)) > 0.6
screen_spoof[glare] = 254.0
img_screen = np.clip(screen_spoof, 0, 255).astype(np.uint8)
Image.fromarray(img_screen).save(os.path.join(OUTPUT_DIR, "spoof_screen_replay.jpg"))

print(f"✅ Generated 5 sample face presentations in: {OUTPUT_DIR}")
