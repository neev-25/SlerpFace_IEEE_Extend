# GuardFace product requirements

Version: research-boundary revision, October 16, 2026.

GuardFace is a local face-comparison prototype combining pretrained YuNet/SFace alignment/embeddings, advisory learned MiniFASNet presentation estimates, texture diagnostics, and SLERP/dropout visualization. It must not be presented as an authentication service or a validated liveness detector. Earlier security and performance claims exceeded the available evidence and are retired.

## Current scope and architecture

Inputs are explicit synthetic presets, uploaded images, camera still captures, or CLI paths. Shared validation bounds decoded dimensions, channels, finite values and ranges. Real operation requires exactly one YuNet-detected face, maps five landmarks to the original pixels, aligns to 112 by 112, and extracts a normalized pretrained SFace embedding. FFT/texture and MiniFASNet scores are advisory on real images; a low texture score cannot prevent research comparison. Explicit synthetic demonstrations use a separate illustrative gate and untrained statistics. No result authorizes identity.

The Streamlit UI calls the shared pipeline and keeps input images in memory. The CLI returns structured status reports. Optional dataset extraction exports research features using a compatible externally supplied checkpoint and the bundled pinned backbone. Offline dataset evaluation separates calibration/test subjects and captures. Report generation deliberately writes requested figures and enforces its gate before feature extraction. A separate local Windows DPAPI/SQLite research store supports encrypted payloads, rotation and revocation; it is not connected to authentication or an account service. There is no production deployment definition.

## Implemented requirements

| ID | Required behavior |
| --- | --- |
| R-01 | Every authentication result remains false. Original trained cosine may yield an explicitly experimental similar/different/uncertain label; protected cosine never determines it. |
| R-02 | Untrained statistical extraction requires explicit demo selection and is visibly labeled. Missing trained prerequisites cannot select it automatically. |
| R-03 | Real comparison requires exactly one YuNet face and five-landmark alignment. Missing or unverified models yield actionable unavailable status without synthetic fallback. |
| R-04 | Static-image scores are described as unvalidated heuristics; neither a score nor face detection is called proof of liveness. |
| R-05 | Missing uploads/captures remain empty. The UI cannot process a required pair until both inputs are supplied. Invalid images produce controlled errors. |
| R-06 | PAD and extraction use the same decoded image. The dashboard creates no shared raw-image files. Existing user files are preserved. |
| R-07 | Inspector text describes operations actually completed. A configuration choice alone cannot claim a transformed template. |
| R-08 | Decoded-image work is bounded and heuristic analysis is standardized to 112 by 112. UI reruns reuse unchanged inputs/results where applicable. |
| R-09 | Research protection rejects identity-destroying/invalid parameters, uses explicit ephemeral contexts and local random generators, and carries no encryption or privacy guarantee. |
| R-10 | Single-image transforms are transient demonstrations, with no claim of enrollment, credential persistence, or revocation. |
| R-11 | Real reports use aligned pretrained features and advisory texture analysis. Synthetic reports retain their illustrative rejection gate. Reports use supported plotting APIs and avoid intermediate biometric files; requested figures are explicit sensitive outputs. |
| R-12 | Dataset annotations preserve pair/label alignment, honor the image folder, confine image paths, validate rows, and handle empty input consistently. |
| R-13 | CLI statuses distinguish completed demos (0), rejection/unverified (1), and operational failure (2). Launchers root paths at their script directory and propagate native statuses. |
| R-14 | Missing legacy encryption/matching stages produce an actionable error before any partial extraction output. |
| R-15 | Dependency declarations separate runtime and development requirements. Model implementation/weight provisioning remains explicit. |
| R-16 | Evaluation output identifies generated inputs and measured class disagreements, without fabricated baseline results or standards/accuracy claims. |

`AntiSpoofResult.is_live` is retained for compatibility and means only a heuristic pass. `authentication_eligible` remains false. Normal detector defaults use `require_face=True`; the isolated `.venv` now has a working OpenCV classifier, while incomplete backends still report unavailable. Synthetic mode is not an alternative authentication policy.

The real prototype uses `modules.face_backend` independently of that legacy Haar heuristic. Missing/multiple/small faces produce recapture guidance. Learned presentation estimates have no live/spoof threshold and cannot block an explicitly labelled research comparison. The original 0.363 SFace example threshold has a +/-0.03 experimental uncertainty band; a custom IR-50 model does not inherit that operating point. Model downloads are explicit setup commands, hash-verified and absent from inference.

## Supported workflows

The dashboard has an aligned-face inspector, trained pair comparison and synthetic evaluation. Results persist until their inputs/settings change; captured camera previews are not duplicated. The supported CLI is `verify_with_antispoof.py`; `test_my_image.py` delegates to it. `--demo` explicitly selects untrained illustrations; otherwise the provisioned SFace backend is used. `--model_path` is an optional original IR-50 override with separate threshold requirements. The legacy dataset extractor still requires its own compatible TFace checkpoint. `evaluate_dataset.py` supports default SFace or explicit IR-50 calibration/test reports. `manage_templates.py` manages synthetic research transforms in a separate encrypted local store.

`run_guardface.ps1` supports `demo`, `cli`, `eval`, and `test` plus explicit `-PythonExe`; by default it prefers the workspace `.venv`. Its paths are independent of caller working directory. `run_pipeline.ps1` exits 2 because the original enrollment and matching packages are absent. Setup, command examples, input conventions, and output retention details are in [README.md](README.md).

## Verification requirements

Every corrected boundary needs a regression that fails on the audited behavior: false authentication, silent statistical fallback, missing-input success, retained shared inputs, misleading inspector state, ignored report gates, invalid annotations, and lost exit codes. Numerical tests must check finite zero embeddings, uniform constant attention, valid temperatures, and supported model input/channel configurations when their backend is available.

Run `python -B -m pytest -q` and applicable command/tool smoke checks. Record interpreter/dependency versions, outcomes, skipped checks, and environment limitations. Test counts describe regression coverage rather than certification or biometric performance. Exact audit-remediation validation belongs in [docs/AUDIT_FIXES.md](docs/AUDIT_FIXES.md), not a permanent readiness badge.

## Requirements before enabling authentication

These are outstanding research and product requirements, not implemented assurances:

1. Independently measure the default pretrained SFace backend on supported real captures. Its source-pinned weights, alignment and inference are implemented. Obtain original IR-50 weights only for that optional architecture's evaluation; random-checkpoint tests do not establish its recognition accuracy.
2. Evaluate PAD using independent real bona fide captures and physical print/display attacks across cameras, lighting, resolutions, and resampling conditions. Measure each attack class and detection failure separately. Haar face presence alone does not satisfy this requirement.
3. Implement and calibrate an appropriate protected matching method against held-out genuine/impostor identities. Shared-noise cosine is not an identity policy. Define the operational threshold only from an explicit validation protocol.
4. Define enrollment identity binding, key/context handling, access control, retention, revocation, and recovery before integrating persistent credentials into a product. The separate DPAPI prototype provides local storage/transaction checks, not identity or anti-rollback guarantees. Validate revocation and cross-deployment unlinkability rather than inferring them from randomization.
5. Test inversion/reconstruction attacks and privacy claims under an explicit adversary model. A dimension count or dropout rate is not a proof of resistance.
6. Extend completed local browser, keyboard/tab, responsive, session-isolation and dependency-inventory checks to physical supported devices, screen readers, load and deployment. The Python advisory scan excludes embedded native libraries and the OS/browser stack. Hosted CI requires publication of the reviewed workflow.

The functional pretrained implementation and validation record is [docs/FUNCTIONAL_PROTOTYPE.md](docs/FUNCTIONAL_PROTOTYPE.md). [docs/AUTONOMOUS_FOLLOWUP.md](docs/AUTONOMOUS_FOLLOWUP.md) and the initial audit-fix record remain dated earlier checkpoints.

No target accuracy, latency, throughput, standards certification, or privacy guarantee is established by the current implementation. Product owners must choose acceptance criteria and substantiate them through the above protocols before enabling an authentication decision. The intended project license also needs owner confirmation; no missing-license assertion is treated as a grant.
