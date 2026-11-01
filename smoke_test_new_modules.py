"""
smoke_test_new_modules.py
Smoke checks for the additional PAD modules.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)

from modules.antispoof.face_alignment import preprocess_for_pad
r = preprocess_for_pad(img)
print(f"[OK] face_alignment: face_crop={r['face_crop'].shape}, detected={r['face_detected']}")

from modules.antispoof.chrominance_analysis import analyze_chrominance
c = analyze_chrominance(img)
print(f"[OK] chrominance: cr_std={c['cr_std']:.2f}, skin={c['skin_tone_fraction']*100:.1f}%, score={c['chrominance_liveness_score']:.3f}")

from modules.antispoof.glare_detection import analyze_specular_glare
g = analyze_specular_glare(img)
print(f"[OK] glare: ratio={g['glare_ratio']*100:.3f}%, blobs={g['num_hotspots']}, score={g['glare_liveness_score']:.3f}")

from modules.antispoof.visualizer import generate_pipeline_visuals
vis = generate_pipeline_visuals(img, size=112)
for k, v in vis.items():
    assert v.shape == (112, 112, 3), f"Bad shape for {k}: {v.shape}"
print(f"[OK] visualizer: {list(vis.keys())} all 112x112x3")

from modules.antispoof.metrics import evaluate_pad_system, compute_det_curve, find_eer
m = evaluate_pad_system([0.8, 0.9, 0.75, 0.85], [0.2, 0.3, 0.25], [0.1, 0.15, 0.22], 0.60)
t, f, n = compute_det_curve([0.8, 0.9, 0.75], [0.2, 0.3, 0.1, 0.25])
eer, _ = find_eer(f, n, t)
print(f"[OK] metrics: APCER={m.apcer*100:.1f}% BPCER={m.bpcer*100:.1f}% ACER={m.acer*100:.1f}% EER={eer*100:.1f}%")

from modules.antispoof.deep_pad import DeepPAD
for bk in ['classical', 'cnn_builtin']:
    res = DeepPAD(backend=bk, threshold=0.60).predict(img)
    assert 0.0 <= res.p_live <= 1.0
    verdict = "HEURISTIC_PASS" if res.is_live else "HEURISTIC_REJECT"
    print(f"[OK] deep_pad({bk}): score={res.p_live:.4f} verdict={verdict} (not liveness evidence)")

from modules.antispoof.detector import AntiSpoofDetector
res_det = AntiSpoofDetector(threshold=0.60).evaluate(img)
print(f"[OK] detector (unified): score={res_det.liveness_score:.4f}, verdict={res_det.verdict}")

print()
print("=" * 52)
print("  ALL 6 NEW MODULES VERIFIED SUCCESSFULLY")
print("=" * 52)
