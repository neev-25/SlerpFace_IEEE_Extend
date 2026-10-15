# GuardFace status

Updated October 16, 2026. Workspace: `C:\Documents\AI_Project`.

**Current state: functional local face-comparison prototype; authentication disabled.** The project now runs pretrained YuNet/SFace detection, alignment and embeddings for camera/upload inputs, displays experimental similarity outcomes, and supplies optional learned MiniFASNet presentation estimates. Texture scores are advisory on real images. It does not establish live-human presence, population accuracy, protected-template privacy or deployment readiness. See [docs/FUNCTIONAL_PROTOTYPE.md](docs/FUNCTIONAL_PROTOTYPE.md) for the current implementation and validation.

## Implemented audit remediation

| Area | Current behavior | Validation limit |
| --- | --- | --- |
| Identity comparison | Trained original cosine produces explicit experimental similar/different/uncertain labels. Authentication remains false. Transformed cosine cannot determine the label. | The 0.363 SFace starting point and +/-0.03 uncertainty band are not calibrated on the user's camera/data. |
| Extractor selection | Default YuNet/SFace weights are provisioned with pinned hashes; statistical features require explicit demo selection and never replace missing models. Optional original Tencent code remains available. | The original IR-50 checkpoint is still optional/external; independent accuracy evidence is absent. |
| Face/PAD checks | Real comparisons require exactly one landmark-aligned YuNet face. Learned MiniFASNet and texture estimates remain visible as advisory; texture does not veto the comparison. | A static estimate does not prove liveness. Physical camera/attack trials remain needed. |
| UI input/privacy | Missing uploads/captures stay empty; paired processing requires both inputs. Processing uses decoded images in memory and accurate operation labels. | No claim of general application/privacy certification follows from removing shared image writes. |
| Research transforms | Ephemeral contexts and local generators replace the universal seeded context; invalid/identity-destroying parameters are rejected. | Randomness and dropout do not prove inversion resistance, unlinkability, or revocation. |
| Reports | Supported colormaps, enforced heuristic gate, no intermediate raw-image file, explicit fresh output directory. | Requested report figures intentionally contain input/feature visualizations. |
| Dataset tooling | Named folders and annotations are validated. Offline evaluation separates calibration/test subjects and image contents, freezes thresholds, and reports class errors with uncertainty. | Real identity evaluation needs an explicit trained checkpoint and independent data. Supplied labels are not certified ground truth. |
| Model implementation | Deep-backbone channels, exact input checks, normalization, and constant attention handling were corrected; CPU tensor and random-checkpoint compatibility tests run. | Random weights establish compatibility, not recognition accuracy. |
| Local research storage | A separate Windows user-scope DPAPI/SQLite prototype supports encrypted payloads, atomic rotation and revocation. | It has no identity binding, matcher or product account integration; same-account processes can decrypt. |
| Launchers | Script-root paths, explicit interpreter option, automatic local `.venv` selection, validated modes, and native statuses. | The old multi-stage enrollment/matching workflow is absent and exits 2 before output. |
| Evaluation/docs | Generated-pattern results and historical limitations are labeled explicitly; unsupported claims are withdrawn. | Independent physical presentations and identities have not been evaluated. |

The initial finding dispositions are preserved in [docs/AUDIT_FIXES.md](docs/AUDIT_FIXES.md). The subsequent environment, browser, storage and evaluation work is consolidated in [docs/AUTONOMOUS_FOLLOWUP.md](docs/AUTONOMOUS_FOLLOWUP.md). This table is not a declaration that every security or product requirement has been completed.

## Current checks

Use a provisioned runtime/development environment and run:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_runtime.py
.\.venv\Scripts\python.exe -B -m pytest -q
.\.venv\Scripts\python.exe -B .\verify_with_antispoof.py --help
.\.venv\Scripts\python.exe -B -m extract_template.verification --help
.\.venv\Scripts\python.exe -B .\generate_report_images.py --help
```

The launcher provides equivalent test dispatch through `run_guardface.ps1 -Mode test -PythonExe <interpreter>`. An unavailable default pytest temporary directory can be replaced with a dedicated writable `--basetemp`. Do not reinterpret historical positive-only suite results as evidence of biometric security. Current test outcomes, including skips, belong to the dated audit-fix record.

## Outstanding external work

- Default trained SFace inference now runs. Obtain original IR-50 weights only if reproducing that specific architecture, with provenance, integrity and applicable rights.
- Validate YuNet/SFace and learned presentation estimates on independent supported camera captures; detection and a still-image estimate do not prove liveness.
- Collect or obtain independent bona fide identities and physical presentation attacks across cameras, lighting, print/display media, and resolution changes.
- Design and calibrate protected matching; measure genuine/impostor distributions before enabling any authentication threshold.
- Define product enrollment identity binding, access control, retention, recovery and lifecycle policy; validate the local research-store prototype against those requirements before integration.
- Evaluate inversion, reconstruction, unlinkability, and revocation under a documented threat model.
- Extend the completed local desktop/mobile-width browser checks to physical camera/device trials, screen readers, load and deployment testing. Publish reviewed changes before hosted CI can run.
- Refresh the completed 77-distribution dependency scan before deployment and investigate native/OS/browser components outside its scope.
- Confirm project and external-asset licensing with the owner. No LICENSE file currently grants project rights.

These items require models, evaluation data, physical equipment, or owner decisions. No completion date or performance guarantee is inferred from the current local fixes.
