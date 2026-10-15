# Functional camera and upload prototype

Updated October 16, 2026, in `C:\Documents\AI_Project`. This implementation follows the user's clarification that GuardFace is a prototype and the request to repair its missing real-camera functionality. It supersedes the earlier state where non-demo extraction was deliberately stopped. Earlier audit/tooling reports remain dated evidence, not a description of the current default face backend.

## What now works

Camera/upload images use **actual pretrained local inference**: YuNet detects exactly one face and five landmarks; SFace aligns original pixels to 112 by 112 and returns a normalized 128-value embedding. A pair produces original cosine plus an explicitly experimental `similar`, `different` or `uncertain` label. A single image completes a trained feature/transform operation. Missing, multiple or too-small faces return recapture guidance. Missing or unverified model bytes produce setup guidance without a statistical fallback.

The previous real-image texture gate could reject a bona fide camera capture before useful processing. Texture/frequency measurements are now advisory for real images. A learned two-model MiniFASNet estimate is also shown separately. Neither one blocks a research comparison or claims physical liveness. Synthetic mode remains explicit and retains its illustrative texture gate; generated textures are not passed to the trained backend as real faces.

The pipeline still returns `authenticated=false`, `is_match=null` and an authorization value of false. A successful *operation* is different from authorization. `experimental_match` and `comparison_label` describe the experiment only. This is appropriate for the user's local prototype; it does not establish a deployed biometric authentication service.

The default experimental threshold is the official OpenCV SFace example cosine point, **0.363**. Scores within **+/-0.03** are shown as uncertain. The uncertainty band is a transparent display rule, not a statistical confidence interval. Neither point is calibrated to this camera/population. The UI displays three decimal places. Original trained cosine alone determines the label; common-noise transformed cosine never does.

SFace's eight 16-value groups are an explicitly labelled adaptation of a global embedding. They are **not** the official Tencent SlerpFace network's learned spatial groups. That original optional IR-50 checkpoint still has separate provenance/provisioning requirements. An explicitly supplied compatible checkpoint now receives aligned input and actually runs; it cannot inherit the default SFace operating point without a separately chosen/calibrated research setting.

## UI and supporting tools

- Camera widgets display each capture once, removing the redundant second preview.
- Pair, inspector and benchmark results persist across unrelated reruns and invalidate when inputs/settings change.
- The real inspector uses the same aligned model analysis, shows face/quality metadata and reports texture assessment as inconclusive/advisory.
- The sidebar shows actual model readiness. Default trained operation requires no checkpoint textbox entry.
- The CLI returns 0 for a completed trained research operation or explicit synthetic demo, 1 for rejected/no-operation input, and 2 for invalid/unavailable operation. No exit code means authenticated.
- Real report figures now use aligned SFace features and dynamic eight-group layouts, without inventing a 7x7 spatial map for a global embedding. Synthetic reports preserve their existing gate and 49-group illustration.
- Offline identity evaluation supports default CLI `--backend sface` and optional `--backend ir50 --model <path>`, with aligned inputs, calibration/test separation and model-artifact binding. The evaluator's presentation rates remain explicitly the legacy static-image heuristic; they do not validate MiniFASNet.
- CI now explicitly provisions and verifies the pinned face and presentation models before its regression/browser checks. Hosted CI still requires publication; no commit, push or remote dispatch was performed.

## Model provisioning and provenance

The existing `.venv` and installed Python distribution versions were retained. No further package update was required. Four pretrained artifacts were fetched from official upstream repositories at exact commits, checked against exact sizes/SHA-256 hashes and installed in ignored local directories. Inference and readiness checks never download or upload photographs.

| Backend | Artifacts / source | Local location |
| --- | --- | --- |
| Detection/alignment/recognition | YuNet 2023mar and SFace 2021dec; OpenCV Zoo commit `47534e27c9851bb1128ccc0102f1145e27f23f98` | `models/opencv/` |
| Advisory presentation estimate | MiniFASNetV2 and MiniFASNetV1SE; MiniVision commit `b6d5f04ad78778917853b25c778acef6d5626d15` | `ckpt/presentation/` |

[PRETRAINED_FACE_BACKEND.md](PRETRAINED_FACE_BACKEND.md) and [PRESENTATION_MODEL.md](PRESENTATION_MODEL.md) record primary sources, full hashes, licenses, preprocessing, class semantics and limitations. Complete applicable upstream notices are retained under `third_party/opencv_zoo` and `third_party/silent_face`. Official source-directory terms support this local research integration; training-data/commercial-right questions are documented rather than assumed resolved. No root project license was invented.

Setup from a fresh checkout after installing the hash-locked environment:

```powershell
.\.venv\Scripts\python.exe -B scripts/provision_face_models.py
.\.venv\Scripts\python.exe -B scripts/provision_pad_model.py
.\.venv\Scripts\python.exe -B scripts/check_runtime.py --require-prototype --require-presentation
```

Verified files are reused. Failed/truncated/oversized/wrong-hash downloads never install unverified bytes. Face provisioning requires explicit `--replace` for an existing unverified artifact. Presentation provisioning uses atomic exclusive publication and rejects existing invalid artifacts. Restricted weights-only Torch loading replaces the upstream unrestricted loader for MiniFASNet. Readiness now reports `prototype_runtime_ready=true` and learned presentation availability; a missing optional IR-50 `trained_checkpoint` does not mean the default model is missing.

## Validation evidence

Focused passes completed during implementation:

| Check | Result |
| --- | --- |
| Core safety and trained-comparison unit regressions | 40 passed |
| Evaluation/report regressions | 52 passed |
| Actual pretrained face backend, input contracts, model integrity and atomic provisioning | 26 passed, zero skips |
| Learned presentation backend, source/weights integrity, crop/scale/class semantics, failure states and actual weights | 39 passed, zero skips |
| Dashboard interactions and persistence | 14 passed plus seven subtests, zero skips; threshold display precision also passed its focused repeat |
| Ruff | All first-party checks pass |
| Runtime capability check | Both pretrained face and learned presentation models ready; authentication false |

Final consolidated validation:

| Check | Final outcome |
| --- | --- |
| Complete Python run | **286 passed, seven subtests passed, zero skips**, exit 0; 373.20 seconds under concurrent model/browser validation |
| Final additional IR-50 alignment regression | One passed in 2.70 seconds after the combined run's collection; current collection has 287 tests. No implementation change followed the passing combined run |
| Real browser suite | **10/10 passed**, zero page errors, Chrome 155.0.8059.41 / Playwright 1.62.1; desktop/narrow light/dark, labels/keyboard, missing/corrupt inputs, synthetic boundaries, sessions, benchmark, actual trained image uploads and persistent/invalidation behavior |
| Real browser model outputs | Mild transformed public source: original cosine 0.9779, experimental similar. Separate portrait: 0.1519, experimental different. Learned presentation output remains visible/advisory |
| Actual real-image report command | Exit 0; all seven PNG figures generated, including the corrected eight-vector-group diagram; that diagram visually inspected |
| Runtime requirements | `scripts/check_runtime.py --require-prototype --require-presentation` exits 0 with both backends ready and authentication disabled |
| Environment/source checks | `pip check`, complete first-party Ruff and whitespace checks pass; no Python distributions changed |
| CI definition | YAML parses with 14 steps, read-only contents and both model provisioners/capability requirements; no hosted execution claimed |
| Existing user server | Confirmed stale cached import on port 8501, restarted only its verified process tree; fresh browser check passes updated heading/backend-ready with zero app exceptions/page errors |
| Preservation | All seven pre-existing photograph/bytecode artifacts match their recorded starting SHA-256 hashes; no commit, push or remote workflow dispatch |

Focused counts above overlap and must not be added to these totals. Full test log: `pytest-full.txt`; final alignment test used the dedicated `alignment-regression` directory; browser evidence is under `browser/final-all/results.json`. A final focused real-upload/390-pixel light/dark screenshot repeat passed 1/1 after waiting for sidebar animation; its evidence is `browser/visual-final/results.json`. The first browser harness attempt had a tab-panel locator mismatch, corrected to named upload controls; no application error was recorded for it. Browser/keyboard checks do not establish complete WCAG, screen-reader or physical-device coverage.

An auxiliary collection-only command initially used the sandbox's default TEMP and printed a permission error during interpreter-shutdown cleanup; the actual full/focused test runs used explicit writable TEMP/TMP and were clean. Collection was repeated with those same settings (`collection-final.txt`). Exact vendored-byte preservation was checked with `git check-attr text`; relevant files report `unset`, preventing Windows newline conversion from invalidating their hashes.

Actual model smoke checks use two documented public NASA/U.S. Navy portraits, not the user's photographs. The source/hashes/rights evidence is in [tests/fixtures/README.md](../tests/fixtures/README.md). The original portrait against its mild brightness/rotation derivative produced cosine **0.971476**; the separate public portrait produced **0.151917**. The existing OpenCV threshold was not fitted to these fixtures. A transformed copy is not an independent camera recapture, and two portraits do not establish recognition accuracy, fairness or physical spoof resistance.

One combined local timing run measured first face extraction at 3.019 seconds and initial optional Torch/PAD loading plus estimation at 24.160 seconds. Subsequent full pair operations took 1.424 and 1.446 seconds. A separate process measured roughly 25 seconds cold and 3.8 seconds warm. Expect the first analysis to initialize models; these are local observations under concurrent validation, not a latency guarantee. OpenCV thread-pool reduction was tested and was slower, so it was not applied. MiniFASNet caps CPU intra-operation threads at two process-wide to avoid racing per-request thread changes; the effect is documented in its model notes.

The evidence root is `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\functional-prototype`. `model-smoke.json` records measured operations. Browser screenshots/logs and full Python/report logs are stored there separately.

## Run the updated prototype

```powershell
cd C:\Documents\AI_Project
.\.venv\Scripts\python.exe -B -m streamlit run .\app_demo.py
```

In **Pair Comparison**, select Camera/Upload for both inputs. Keep **Use synthetic feature demo** off and leave the optional checkpoint path blank. Supply one clear face per image and run comparison. Expect `RESEARCH_COMPARISON`, actual original cosine and an experimental label. Texture and presentation estimates remain separate. A first cold run may take around 30 seconds in this environment. Use the preset inputs only with explicit synthetic mode.

If an existing Streamlit process shows an import error after source changes, restart that specific server so its cached Python modules reload. Captures/results are in memory and are cleared by a restart; the app does not write them to shared JPEGs.

The existing user's port-8501 server was restarted during this pass after a read-only browser confirmed `ImportError: cannot import name inspect_real_image`. Its old verified tree was replaced by `.venv` Python with `-B`, loopback binding and telemetry disabled. The updated server intentionally remains available at `http://localhost:8501`; fresh-context validation confirms it loads. All isolated test browser/server trees were closed separately. Reload the existing tab and recapture images after this restart.

## Remaining limits

This pass completes a functional **local research comparison flow**, not a universal biometric guarantee. Real camera/pose/lighting recaptures, unrelated identities and physical print/display attacks remain necessary for measured accuracy/PAD claims. Users must evaluate the actual target camera, define acceptance criteria and calibrate against independent data before considering operational decisions. Learned presentation output is still a single-frame estimate and cannot establish trusted physical capture or resist every replay/injection attack.

Production account integration, identity-bound enrollment, recovery/retention policy, protected matcher calibration, inversion/unlinkability research, full accessibility/device coverage, sustained load and deployment validation remain separate future work. Model/source advisories, weights/data rights and a project-wide license require ongoing review for the intended distribution. The earlier 77-package advisory scan is an unchanged Python-distribution snapshot, not a scan of newly added model graphs, all native libraries or the OS/browser stack.

Existing user photographs and earlier modifications were preserved. Seven pre-existing photograph/bytecode artifacts were snapshotted outside the checkout before investigation. The old user server regenerated bytecode while watching module changes; after its controlled restart with `-B`, the pre-existing modified artifact was restored from its verified snapshot, and initially clean app bytecode was restored to its original repository bytes. All seven starting snapshot hashes match. `.gitattributes` preserves exact vendored-source/license and fixture bytes across Windows checkouts so Git newline conversion cannot invalidate source hashes. Our validation uses `-B`/`PYTHONDONTWRITEBYTECODE` and private writable temporary directories. No user camera was activated by automation and no user image was sent to an external service.
