# Audit remediation and validation

This is the initial remediation checkpoint. Later authorized dependency provisioning, tensor/browser validation, offline evaluation and encrypted local research storage are recorded in [AUTONOMOUS_FOLLOWUP.md](AUTONOMOUS_FOLLOWUP.md). Statements below about absent backends, skipped tensors or unperformed scans describe this earlier checkpoint.

Completed local code pass: October 16, 2026 (Asia/Calcutta), workspace `C:\Documents\AI_Project`. Based on the October 15 audit of checkout `4c42099` and its existing working tree. No commits, dependency installations/updates, credential requests, or production actions were performed.

## Result and operating contract

The application remains a research demonstration. False authentication is prevented by a closed decision boundary: every pipeline decision is false, every report keeps `authenticated=false`, and `is_match` is unset. Existing cosine values are now explicitly diagnostic, because this checkout has no validated PAD or calibrated protected matcher. Replacing the threshold with another guessed value would not repair the underlying assurance problem.

The statistical extractor is available only through explicit demo selection. The default path never silently substitutes it for a missing model. Normal PAD requires exactly one detectable face and reports missing backend support as unavailable. Explicit synthetic mode can process drawings, but cannot verify a live human or an identity. A single-image demonstration does not enroll or persist a credential.

This is an implemented safety restriction and API behavior change, not a claim that genuine/impostor recognition, physical PAD, inversion resistance, or a production enrollment product has been completed.

## Finding disposition and per-fix checks

“Fixed” below refers to the investigated local code defect. “Mitigated” means the unsafe decision or misleading claim is removed while a genuine biometric capability still needs external validation.

| Audit finding | Disposition and concrete change | Checks after the fix |
| --- | --- | --- |
| GF-01 Critical: common-noise false matches | **Mitigated.** No cosine/threshold authorizes identity. Alpha 1 is rejected. Fresh explicit ephemeral contexts replace universal seeded operational material; contexts are shared only within a diagnostic comparison. Local RNG and per-group dropout replace global reseeding. No claim of a calibrated matcher or unlinkability. | Different-identity demo never authenticates; mocked available-checkpoint path remains unverified with no extraction; extreme alpha, fresh-context, mask, seeded-research reproducibility and parallel/antipodal numerical tests pass. |
| GF-02 High: silent untrained fallback | **Fixed.** Trained extraction raises an actionable unavailable error; statistics require `demo=True` / `--demo`. Backend status is explicit. Optional model loading is cached and keyed by checkpoint metadata, uses weights-only loading, and handles expected incompatible-weight errors. | Missing checkpoint cannot call the statistics path; explicit demo shapes are correct; unsafe/incompatible weights-only rejection maps to `FeatureExtractorUnavailable`; unavailable UI/CLI cases pass. Real model inference is still untested. |
| GF-03 High: resampled/no-face PAD bypass | **Mitigated.** Default face-presence filter rejects no/multiple detected faces before heuristics; fixed analysis resolution and shared decoding are enforced. Static-image heuristics are no longer described as liveness evidence, and no pass can authenticate. | No/multiple-face gate spies prevent spectral/extraction work; all four audited print-resize variants cannot authorize; missing/incomplete face support is unavailable rather than bypassed. Real physical attack resistance is not established. |
| GF-04 High: shared retained biometric files | **Fixed.** Dashboard/core extraction uses in-memory decoded images, with no shared JPEG paths. PAD and extraction receive the same decoded snapshot. Report tools also avoid temporary JPEGs. | Image-save spy rejects any core raw-input write; dashboard passes ndarray inputs and retains original file hashes; report tests prohibit JPEG intermediates. The shared-file race is removed rather than serialized. |
| GF-05 High: empty upload/camera authentication | **Fixed.** Missing input is `None`; required actions are disabled. Presets appear only when explicitly selected. | Every empty/one-empty Upload/Camera pairing tested through AppTest; no core call or success outcome. |
| GF-06 Medium: false inspector encryption | **Fixed.** Inspector states that generation has not run and requires an explicit operation; it displays the actual result. Configuration alone no longer signals encryption. | Inspector state and explicit-call behavior tested through AppTest; successful demo and unavailable backend handled separately. |
| GF-07 Medium: unbounded full-image PAD work | **Fixed for local work bounds.** Decoded inputs are limited to 16 MP / files to 20 MiB. Spectral/texture analysis is always 112 by 112; face search is bounded to 640 on its longest side. Radial accumulation processes pixels once. Latest inspector diagnostics are cached per session; unrelated reruns do not redraw samples. | Oversize and dimension validation tests; spectral input-shape spy for 1080p; stable sample/local RNG/session-cache tests. Local 1920 by 1080 heuristic-only preparation/analysis measured 13.6–14.7 ms after warmup (three calls), versus seconds in the prior audit. Not a general timing guarantee. |
| GF-08 Medium: report plotting crash | **Fixed.** All removed colormap calls replaced; each figure closes even on save failure; Windows help/printed messages are compatible with the inspected consoles. | Colormap/save-failure regression; plain system/conda interpreter help; full installed Matplotlib 3.11.1 run produces all six stages plus summary. |
| GF-09 Medium: ignored report gate | **Fixed.** Stage 2 returns its result and `main` stops before any extraction for rejection. No temporary image is left behind. | Real rejected bundled print produces only stages 1–2; extraction spies confirm no call; temporary-file failure checks pass. Figures intentionally requested by the user are still exported. |
| GF-10 Medium: incomplete dependency declaration | **Fixed declaration; environment recreation pending.** Runtime imports are declared with inspected demo versions; development tools have a separate manifest. Unused `dareblopy` removed. Model provisioning/provenance is explicit; an unrelated similarly named package is not installed. | Manifests inspected against imports; installed demo `pip check` passes. No clean installation, transitive lock recreation or full advisory scan was performed. Installed OpenCV has incomplete face support, which is handled safely and documented. |
| GF-11 Medium: absent legacy packages | **Fixed supported entry points; unavailable stages explicit.** Local imports corrected, help is independent of optional Torch, legacy own-image tool delegates to the single CLI. Old multi-stage launcher exits 2 before side effects, naming absent enrollment/matching stages. | CLI/help, absent checkpoint/data/backend, no-output-on-error tests; launcher subprocess stubs from outside workspace preserve 0/1/2 for all supported modes. No missing original stages were fabricated. |
| GF-12 Medium: ignored dataset image folder | **Fixed.** Bare and dataset-prefixed names resolve inside the named folder; labels/row widths/path traversal/missing files are validated; empty datasets have stable shapes; files close promptly. | Nine initial folder/annotation regressions, then broader parameterized data/launcher suite; correct pair/label alignment, confinement and empty cases all pass. |
| GF-13 Medium: denied CLI exits success | **Fixed.** Exit 0 means processed explicit demo only; 1 means rejected/unverified; 2 means invalid input or unavailable prerequisites. Launchers propagate native exit status and resolve script paths independently of caller CWD. | Real demo exit 0, rejected print exit 1, unavailable face backend exit 2, missing image/invalid alpha exit 2; subprocess wrapper and launcher status tests pass. |
| GF-14 Medium: unsupported evaluation/security claims | **Fixed reporting.** Removed active production, standards-certification, mathematical inversion, cross-system unlinkability and timing guarantees. Synthetic metrics contain only measured generated-class disagreements. No hardcoded comparative baseline or uninstrumented privacy assertion. | Evaluation wording/metadata tests, generated-case CLI smoke, report titles/labels and four documentation files reviewed. Independent real evaluation remains required. |
| GF-15 Medium: discarded “enrollment” credential | **Fixed misleading operation contract.** Operation is named/described as transient demonstration and never claims enrollment; reports contain no template/key/mask or persistent credential claim. | Single-image status/report tests require `DEMO_ONLY`, false decision, and explicit no-enrollment wording. A real credential lifecycle remains a separate product requirement. |
| GF-16 Low: image/parameter errors | **Fixed tested boundaries.** Shared decoding validates dimensions/channels/ranges/finiteness before pixel processing; EXIF orientation is applied. Configuration validates finite safe bounds. Caller streams remain open/position-preserved. Byte limit and decoding use the same open file. | Fourteen input tests plus transform/CLI bounds and UI corrupt/oversize tests. Independent review also found huge-finite vector overflow, complex keys, and weights-only checkpoint exceptions; repaired and regression-tested. |
| S-01: deep model channels | **Statically repaired, backend validation pending.** Group convolution uses the selected backbone output channels; exact square input sizes and group arguments validated. | Parser/structural tests pass; actual tensor case skips without Torch/compatible backbone. |
| S-02: zero/constant normalization | **Statically repaired, backend validation pending.** Zero embeddings remain finite; constant attention uses uniform weights; attention initialization handles nested layers and missing convolution bias. | Focused optional tensor regressions exist; three PyTorch-dependent tests skip in this environment. |

## Additional practical improvements

- Default pytest discovers the intended tests and the comprehensive suite. `test_section` is no longer a fake fixture-consuming test. The legacy root CLI is no longer imported as a test.
- The comprehensive suite now checks the closed authentication boundary, different identities, direct extraction gates, missing models, JSON secrecy and explicit non-enrollment. Syntax checking is labeled syntax-only.
- Ruff configuration is explicit and the complete tree passes it. No blanket lint suppression or formatting-only large rewrite was used to hide defects.
- Added a Windows/Python 3.11 GitHub workflow for declared environment installation, lint, regression tests, synthetic evaluation and report smoke. It uses read-only token permissions and does not persist checkout credentials. Structure was checked locally; the workflow has not run remotely. Action usage was checked against the [official checkout](https://github.com/actions/checkout) and [setup-python](https://github.com/actions/setup-python) documentation.
- Sample generation has a main guard, no import-time writes, local reproducible RNG, configurable output and explicit overwrite protection. Demo inputs remain stable across unrelated UI reruns.
- Added `.gitignore` for bytecode, caches, runtime outputs, checkpoints and local environment files. Existing tracked generated files and user image changes were preserved; no Git-index cleanup was attempted.
- Model-backed extraction reuses an eval model and shares the backbone output for vector/group extraction. Dataset output arrays now use float32. No trained-backend performance claim is made.
- Removed the assertion that a missing LICENSE file grants MIT rights. Project/external-code licensing remains an owner decision; a license was not invented.

## Final validation

All commands ran locally on Windows using existing interpreters. Runtime dependencies were not installed or upgraded. The system Python 3.13 has pytest and basic scientific packages but no Streamlit or Torch; the existing `slerpface` Python 3.11 environment has the UI/report stack but no pytest or Torch. Tests were therefore run in both environments without mixing or copying installed packages.

| Command/check | Outcome |
| --- | --- |
| System `python -B -m pytest -p no:cacheprovider --basetemp <external-validation-dir>/full-pytest -q`, plugin autoload disabled | **116 passed, 13 skipped**, 13.86 s. Ten skips are UI tests because Streamlit is absent from that interpreter; they were all executed separately below. The other three need Torch. |
| Python 3.11 `python -B tests/test_dashboard.py`, writable external TEMP/TMP | **10/10 pass**, including mocked and real-core integration, missing/invalid input, unavailable backend, privacy, cache and synthetic benchmark flows. |
| Python 3.11 `python -B tests/comprehensive_test_suite.py` | **16/16 pass**, also collected/passed in the system pytest run. |
| System `python -B tests/test_antispoof.py` | Five subsystem checks pass, also included in pytest. |
| Focused sets after changes | Image input initially 13 pass, then 14 after open-file byte check; core safety progressed through 23/26/27 pass as review edges were added; data/launchers 31 pass and 3 skips; report/evaluation 23 pass; dashboard 10 pass. These overlap the final totals. |
| Combined distinct collected cases across available environments | **126 pass, 3 model-dependent cases remain skipped** out of 129 collected. This is combined coverage across interpreters, not a single pytest invocation or a biometric validation result. |
| `ruff check --no-cache .` | Pass. |
| `ast.parse` all project Python files | All 22 parse. |
| PowerShell native parser, both scripts | Zero syntax errors. |
| `pyproject.toml` and workflow YAML | Parse; workflow read-only permissions/step structure checked. |
| `git diff --check` | Pass after removing an extra documentation EOF blank line. |
| Verification CLI real process statuses | Explicit demo 0; rejected print 1; unavailable default backend 2. Different-identity demo keeps authentication false. |
| Legacy wrapper, core/extraction/report help | Pass; plain Windows console report help no longer needs an encoding override. |
| Python 3.11 synthetic evaluation smoke | Nine generated patterns processed; synthetic-only measured output. |
| Python 3.11 full report smoke | Seven PNGs written to an external fresh directory; selected Stage 2/6 visuals inspected. No intermediate raw JPEG. |
| Python 3.11 rejected report smoke | Only Stage 1/2 outputs; no extraction or subsequent stages. |
| Existing runtime `pip check` | No broken installed requirements; not a CVE scan or clean-environment reconstruction. |

The existing runtime remains Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1, Pillow 12.3.0, Streamlit 1.63.0, Matplotlib 3.11.1 and OpenCV 5.0.0.93. Runtime pins reflect this inspected demo environment. The installed OpenCV package lacks `CascadeClassifier`; the new guard reports that limitation rather than assuming its distribution version proves full capability. The system test interpreter uses Python 3.13.7 with NumPy 2.2.0, SciPy 1.17.0, Pillow 12.2.0, pytest 9.1.1 and Ruff 0.15.20. No production deployment is pointed at that system test environment by these changes.

Validation logs and generated smoke artifacts are outside the repository at `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\fix-validation` and sibling report/test directories. The earlier audit report is preserved separately and has not been rewritten to imply its findings were never present.

## Preservation and remaining work

The two pre-existing modified `temp_img1.jpg` / `temp_img2.jpg` files and the pre-existing untracked evaluation bytecode were SHA-256 checked before and after remediation: all three are unchanged. Existing sample files were not regenerated. Git metadata was not modified by a commit, staging operation, branch change or cache-removal operation. New ignore rules hide the untracked bytecode from ordinary status without deleting it.

The following cannot be completed or represented as validated by local code changes alone:

1. **Biometric accuracy/security:** trusted model implementation/weights, real bona fide/impostor identities, physical print/screen attacks, supported cameras, and protected-match calibration. The common-noise mathematical compression still exists in illustrative SLERP; it is explicitly excluded from identity decisions. Authentication must remain disabled until a validated decision method exists.
2. **Model/runtime provisioning:** install a compatible full face-presence backend and intended Torch/torchkit stack, recreate the declared environment cleanly, run the three skipped tensor checks and real checkpoint forward passes. Checkpoint/API/ABI/provenance compatibility is not established by a file path.
3. **Persistent credentials and privacy claims:** identity binding, access/retention/key lifecycle, storage/revocation policy and inversion/unlinkability evaluation require an explicit product/threat model and data. No credential database or cryptographic guarantee was fabricated.
4. **Licensing:** the owner must establish the project license and external code/weight rights before a license file/grant is added.
5. **Operational validation:** remote CI execution, browser/device accessibility and responsive testing, physical camera operation, comprehensive deployment/load testing and a complete transitive vulnerability/SBOM review remain outstanding. The configured CI has only been parsed/inspected locally. Existing package installation state was preserved.

These limits are recorded without blocking any safe local remediation above. No approval was requested for routine code changes or tests.
