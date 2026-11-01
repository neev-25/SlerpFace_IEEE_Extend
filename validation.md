# SlerpFace / GuardFace — Validation Record

**Date:** 2026-11-01
**Executor:** Automated local revalidation
**Python:** 3.13.2
**Platform:** Windows (win32)
**Project branch:** `main` @ HEAD `683b1e1c`; checks include the uncommitted working-tree changes present during validation
**Scope:** SlerpFace paper prototype + PAD (Presentation Attack Detection) extension

> **Assurance notice:** This document records automated software capability checks and
> synthetic benchmark results only. It does not certify biometric accuracy, liveness,
> privacy, standards compliance, or authentication readiness. Physical camera trials,
> independent identity datasets, and calibrated matching thresholds remain outstanding.

---

## Current Working-Tree Revalidation

The checks below were rerun with the repository's `.venv` on the current working tree:

| Check | Result |
|---|---|
| `.\.venv\Scripts\python.exe -B scripts/check_runtime.py` | PASS; Python 3.13.2, demo/prototype ready, face and presentation models verified, authentication disabled |
| `.\.venv\Scripts\python.exe -B -m pytest -q --tb=short` | **305 passed, 7 subtests passed, 0 failures** in 65.46s |
| `.\.venv\Scripts\python.exe -B smoke_test_new_modules.py` | PASS; PAD alignment, chrominance, glare, visualization, metrics and estimator smoke checks |
| Synthetic PAD benchmark (30 samples/class, seed 42) | PASS; 0/90 disagreements on generated patterns only |
| CLI help checks | PASS for comparison, dataset extraction, reports, webcam research comparison, manifest calibration/PAD and local timing |
| Synthetic comparison demo | PASS as a research operation; `authenticated=false`, `is_match=null`, and heuristic output explicitly states it does not establish a genuine person or live capture |
| Calibration/PAD entry points | PASS; require supplied dataset manifests and reject synthetic-only material instead of fabricating real identity/physical-attack results |
| Webcam research flow | PASS; low heuristic scores remain advisory, outputs state `authenticated=false`, and comparisons are labeled experimental |
| Timing benchmark | PASS; reports local component timings without deployment guarantees or throughput claims |

Using the unrelated `python` found on `PATH` is not equivalent to this project runtime. That environment has OpenCV 5.0.0.93 and Streamlit 1.46.1 instead of the `.venv` versions (4.12.0.88 and 1.63.0); runtime readiness is false and its full suite has 15 environment-compatibility failures. Use the repository `.venv` for the results in this record.

The validation report is not a certification. The owner reports that physical trial equipment is available, but has no evaluation dataset yet. Device details, acceptance criteria, credential/deployment policy and project-license decision have not been supplied. Independent-data, threat-model, external-weight, deployment and owner-decision items in Section 9 remain open; this checkout cannot complete them alone.

---

## 1. Environment & Runtime Check

**Command:** `.\.venv\Scripts\python.exe -B scripts/check_runtime.py`
**Result:** PASS

| Component | Status | Detail |
|---|---|---|
| Python | ✅ Ready | 3.13.2 |
| numpy | ✅ 2.2.6 | |
| scipy | ✅ 1.17.1 | |
| Pillow | ✅ 12.3.0 | |
| streamlit | ✅ 1.63.0 | |
| matplotlib | ✅ 3.11.1 | |
| opencv-python | ✅ 4.12.0.88 | |
| Face detection (YuNet) | ✅ Verified | SHA-256: `8f2383e4…` |
| Face recognition (SFace) | ✅ Verified | SHA-256: `0ba9fbfa…` |
| Model backend (PyTorch) | ✅ 2.15.0+cpu | Pinned Tencent/TFace subset |
| IR-50 trained checkpoint | ⚠️ Missing | External — not supplied (optional) |
| MiniFASNet V2 (PAD) | ✅ Verified | SHA-256: `a5eb02e1…` |
| MiniFASNet V1SE (PAD) | ✅ Verified | SHA-256: `84ee1d37…` |
| Authentication enabled | ✅ FALSE | Disabled in all code paths |
| `demo_runtime_ready` | ✅ TRUE | |
| `prototype_runtime_ready` | ✅ TRUE | |

---

## 2. Full Regression Test Suite

**Command:** `.\.venv\Scripts\python.exe -B -m pytest -q --tb=short`
**Result:** ✅ **305 passed, 7 subtests passed in 65.46s — 0 failures**

**pytest version:** 9.2.0
**pluggy version:** 1.7.0

### Test Coverage by Area

| Test File | Tests | Result |
|---|---|---|
| `comprehensive_test_suite.py` — ResearchSafetyTests | 16 | ✅ All PASSED |
| `test_antispoof.py` | 5 | ✅ All PASSED |
| `test_attack_categorization.py` | 4 | ✅ All PASSED |
| `test_dashboard.py` | 14 | ✅ All PASSED |
| `test_data_and_launchers.py` | 34 | ✅ All PASSED |
| `test_dataset_validation.py` | 28 | ✅ All PASSED |
| `test_face_backend.py` | 26 | ✅ All PASSED |
| `test_image_input.py` | 14 | ✅ All PASSED |
| `test_keyless_pin.py` | 4 | ✅ All PASSED |
| `test_pipeline_safety.py` | 27 | ✅ All PASSED |
| `test_pose_estimation.py` | 4 | ✅ All PASSED |
| `test_presentation_backend.py` | 39 | ✅ All PASSED |
| `test_report_and_evaluation.py` | 25 | ✅ All PASSED |
| `test_research_comparison.py` | 13 | ✅ All PASSED |
| `test_runtime.py` | 16 | ✅ All PASSED |
| `test_template_store.py` | 30 | ✅ All PASSED |
| `test_webcam_pipeline.py` | 6 | ✅ All PASSED |
| **TOTAL** | **305** | ✅ **0 failures** |

### Key Regression Categories Covered

- ✅ Authentication always returns `false` — all paths
- ✅ Statistical demo never auto-selects when trained model missing
- ✅ Missing model → `MODEL_UNAVAILABLE`, no synthetic fallback
- ✅ Exact one-face requirement for real comparisons
- ✅ No shared raw-image file writes
- ✅ Inspector reports only operations actually completed
- ✅ Static-image heuristic results do not claim live/genuine status or an unrun transform
- ✅ Calibration and physical-PAD tooling requires supplied real-data manifests; no synthetic fixtures are substituted
- ✅ Webcam comparisons remain experimental; heuristic scores are advisory and authorization remains false
- ✅ Timing reports do not label local measurements as deployment guarantees
- ✅ SLERP transform rejects unsafe/invalid parameters
- ✅ Ephemeral context isolation (no global RNG mutation)
- ✅ CLI exit codes: 0 (demo/research done), 1 (rejection), 2 (error)
- ✅ Launcher path resolution independent of caller directory
- ✅ Image input bounds and pixel range validation
- ✅ Dataset annotation validation + calibration/test leakage prevention
- ✅ DPAPI template store: enroll, rotate, revoke, tamper-rejection
- ✅ Pose estimation: frontal pass, extreme yaw rejection
- ✅ Webcam pipeline: headless mode, similarity calibration
- ✅ Keyless PIN: deterministic derivation, unlinkability, cancelability

---

## 3. PAD Heuristic Evaluation (Synthetic Benchmark)

**Command:** `.\.venv\Scripts\python.exe -B evaluate_antispoof.py --samples-per-class 30 --seed 42`
**Result:** ✅ PASS — 0 disagreements across all 3 classes

| Generated Class | Samples | Class Disagreements | Rate (%) | Mean Heuristic Score |
|---|---|---|---|---|
| Synthetic smooth baseline (bona fide) | 30 | 0 | **0.00** | 0.9553 |
| Synthetic print pattern (attack) | 30 | 0 | **0.00** | 0.4929 |
| Synthetic screen replay pattern (attack) | 30 | 0 | **0.00** | 0.0000 |

**Threshold:** 0.60 | **Seed:** 42 | **Total images generated:** 90

> ⚠️ **Scope:** These are procedurally generated synthetic patterns (halftone grid, moiré
> interference, skin gradients). This does NOT measure real PAD accuracy on physical
> print/display attacks captured by a real camera. Physical attack trials remain outstanding.

---

## 4. SLERP Transform Validation (Demo Mode)

**Command:** `.\.venv\Scripts\python.exe -B verify_with_antispoof.py --demo --img1 <img1> --img2 <img2>`
Both runs used samples from `samples/` directory. `--demo` explicitly selects untrained
statistical features; face presence checks disabled for synthetic images.

### 4a. Same-Identity Pair (personA_1 vs personA_2)

| Field | Value |
|---|---|
| Status | `DEMO_ONLY` |
| Processed | `true` |
| Backend | `statistical_demo` |
| Authenticated | `false` (hardcoded) |
| is_match | `null` (no threshold decision in demo) |
| Tier-1 status | `HEURISTIC_PASS` |
| Tier-2 (SLERP) | `alpha=0.90, 50% dimensions shredded` |
| Raw similarity | `0.99993` |
| Encrypted similarity | `0.999998` |
| Key source | Ephemeral random context |
| Template shape | [49, 16] |
| Zeroed dimensions | 392 / 784 (50%) |
| Heuristic score (img1) | 0.70 (threshold 0.60; not a liveness probability) |
| Heuristic score (img2) | 0.70 (threshold 0.60; not a liveness probability) |
| Heuristic classification | `UNVERIFIED` (heuristic pass only; live capture not established) |

### 4b. Cross-Identity Pair (personA_1 vs personB_1)

| Field | Value |
|---|---|
| Status | `DEMO_ONLY` |
| Authenticated | `false` |
| Raw similarity | `0.996066` |
| Encrypted similarity | `0.999904` |
| Heuristic score (img1) | 0.70 (not a liveness probability) |
| Heuristic score (img2) | 0.70 (not a liveness probability) |

> ⚠️ **Note:** In demo mode, raw similarity reflects statistical (untrained) image noise,
> not real SFace identity features. Both pairs show high similarity because the
> synthetic sample images share the same base texture parameters. This does NOT
> represent real identity discrimination. No identity threshold decision is made.

---

## 5. Face Backend Models — Integrity Verification

**Verified by:** `test_face_backend.py::test_real_models_produce_stable_aligned_normalized_features`
**Verified by:** `test_face_backend.py::test_real_pretrained_pair_smoke_distinguishes_two_public_portraits`

| Model | File | SHA-256 | Present | Verified |
|---|---|---|---|---|
| YuNet (detection) | `face_detection_yunet_2023mar.onnx` | `8f2383e4dd3c…` | ✅ | ✅ |
| SFace (recognition) | `face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5…` | ✅ | ✅ |

- SFace cosine threshold: `0.363` (±0.03 uncertainty band)
- Alignment: 5-landmark YuNet → 112×112 crop on original pixels
- Normalization: byte-to-[-1,1] matching upstream official specification

---

## 6. MiniFASNet PAD Models — Integrity Verification

**Verified by:** `test_presentation_backend.py::test_real_official_pretrained_models_load_and_produce_repeatable_estimates`

| Model | SHA-256 | Source Commit | Present | Verified |
|---|---|---|---|---|
| MiniFASNetV2 (2.7_80x80) | `a5eb02e1…` | `b6d5f04a` | ✅ | ✅ |
| MiniFASNetV1SE (4_0_0_80x80) | `84ee1d37…` | `b6d5f04a` | ✅ | ✅ |

- Source: MiniVision open-source repository (pinned commit)
- Inference: CPU-only, no network calls during inference
- Output: Advisory bona-fide class softmax — NOT a calibrated liveness probability

---

## 7. Third-Party Source Integrity — TFace Backbone

**Verified by:** `test_runtime.py::test_pinned_upstream_source_and_notices`

| File | SHA-256 |
|---|---|
| `torchkit/backbone/common.py` | `53e2025f2534…` |
| `License.txt` | `e534cd4a5cd8…` |
| `LICENSE.face-evoLVe.txt` | `b5ee08d1d7d7…` |

- IR-50 architecture: API-compatible (verified by random-weight roundtrip test)
- Trained IR-50 checkpoint: **not supplied** (external; rights/provenance unresolved)

---

## 8. Security / Safety Properties — All Verified

| Property | Test | Result |
|---|---|---|
| Authentication always `false` | `test_pipeline_safety`, `comprehensive_test_suite` | ✅ |
| No statistical fallback when model missing | `test_pipeline_safety::test_missing_model_never_selects_statistics` | ✅ |
| Demo mode cannot authenticate different identity | `test_pipeline_safety::test_demo_never_authenticates_a_different_identity` | ✅ |
| Transform rejects unsafe boundaries | `test_pipeline_safety::test_transform_rejects_unsafe_boundaries[*]` (6 variants) | ✅ |
| No temp image writes / no key material in public output | `test_pipeline_safety::test_no_temp_input_writes_or_key_reports` | ✅ |
| SLERP stable for parallel and antipodal groups | `test_pipeline_safety::test_slerp_stable_for_parallel_and_antipodal_groups` | ✅ |
| No leakage of subject/context in public results | `test_template_store::test_enroll_load_copies_and_never_exposes_subject` | ✅ |
| Corrupt/tampered ciphertext rejected | `test_template_store::test_corrupt_ciphertext_and_swapped_rows_are_rejected` | ✅ |
| Cross-store payload rejected | `test_template_store::test_namespace_change_and_cross_store_payload_are_rejected` | ✅ |
| Keyless PIN: unlinkability and cancelability | `test_keyless_pin::test_pin_unlinkability_and_cancelability` | ✅ |
| Spoof aborts before extraction | `comprehensive_test_suite::test_spoof_aborts_before_extraction` | ✅ |
| Extreme pose (yaw) rejection | `test_pose_estimation::test_extreme_yaw_rejection` | ✅ |

---

## 9. Outstanding Items — Cannot Be Automated

The following validation items require physical equipment, independent datasets, or
owner decisions. They cannot be completed by automated software execution.

| # | Item | Reason Blocked |
|---|---|---|
| 1 | **Real SFace accuracy on live camera captures** | Requires independent face dataset with verified identities and a real camera |
| 2 | **Physical PAD trials** (print/display attacks) | Requires physical printer, screen, multiple cameras, lighting conditions |
| 3 | **IR-50 trained checkpoint accuracy** | Requires obtaining external checkpoint with verified provenance and rights |
| 4 | **Protected matching calibration** | Requires genuine/impostor score distribution from real enrolments |
| 5 | **Inversion / reconstruction attack testing** | Requires adversary model definition and experiment protocol |
| 6 | **Unlinkability across deployments** | Requires multi-deployment test environment |
| 7 | **Deployment & accessibility testing** | Physical devices, screen readers, load testing required |
| 8 | **Project LICENSE declaration** | Requires owner decision on project and third-party asset rights |

---

## 10. Summary

| Category | Automatable | Result |
|---|---|---|
| Environment / runtime check | ✅ | **PASS** |
| Full regression suite (305 tests) | ✅ | **305 / 305 PASSED** |
| Synthetic PAD heuristic benchmark | ✅ | **0 / 90 disagreements (0%)** |
| SLERP transform pipeline (demo) | ✅ | **Executes correctly** |
| SFace + YuNet model integrity | ✅ | **SHA-256 verified** |
| MiniFASNet PAD model integrity | ✅ | **SHA-256 verified** |
| TFace backbone source integrity | ✅ | **SHA-256 verified** |
| Security / safety properties | ✅ | **All 13 properties confirmed** |
| Real camera / physical PAD trials | ❌ Requires hardware | **Pending** |
| Real identity accuracy (TAR/FAR) | ❌ Requires dataset | **Pending** |
| IR-50 trained accuracy | ❌ Requires external weights | **Pending** |
| Protected matching calibration | ❌ Requires experiment | **Pending** |
| Inversion/unlinkability proof | ❌ Requires adversary model | **Pending** |
| Project LICENSE | ❌ Requires owner decision | **Pending** |

**All software checks possible with the supplied local fixtures: COMPLETE.**
**Outstanding items require physical hardware, external data, or owner decisions.**

---

*Generated automatically on 2026-11-01 by validation run against commit `683b1e1c` (branch: main).*
