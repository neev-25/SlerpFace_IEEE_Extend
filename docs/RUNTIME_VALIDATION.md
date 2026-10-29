# Isolated runtime, model integration, and dependency review

Current dependency security refresh: [SECURITY_UPDATE_2026-10-25.md](SECURITY_UPDATE_2026-10-25.md). The version table and scan narrative below preserve the October 16 checkpoint; the current lock, inventory and audit/SBOM artifacts now reflect the October 25 patched environment.

The later [functional prototype](FUNCTIONAL_PROTOTYPE.md) provisions pretrained OpenCV SFace/YuNet and MiniFASNet weights. The original `trained_checkpoint` field below concerns optional Tencent IR-50 weights; current `prototype_runtime_ready` and `--require-prototype --require-presentation` check the functional default models. Earlier counts/results remain dated checkpoints.

Reviewed October 16, 2026 (Asia/Calcutta), workspace `C:\Documents\AI_Project`.

## Completed

A fresh Python 3.11.16 virtual environment now exists at `.venv`. It was created from the existing Python 3.11 interpreter without enabling system site packages. No shared conda/system installation was modified. Runtime and developer packages were resolved from official PyPI, installed with exact wheel hashes, and tested together. CPU PyTorch came from the [official PyTorch CPU index](https://download.pytorch.org/whl/cpu), following the [official version installation instructions](https://pytorch.org/get-started/previous-versions/).

| Component | Tested version / result |
| --- | --- |
| NumPy / SciPy | 2.2.6 / 1.17.1 |
| Pillow / Streamlit / Matplotlib | 12.3.0 / 1.63.0 / 3.11.1 |
| OpenCV Python | 4.12.0.88; `CascadeClassifier` exists, bundled frontal-face cascade loads, blank-image detection executes and returns zero faces |
| PyTorch | 2.13.0+cpu; real CPU tensor and model tests execute |
| pytest / Ruff / pip-audit | 9.1.1 / 0.15.20 / 2.10.1 |
| Bootstrap tools after advisory remediation | pip 26.2.1 / setuptools 84.0.0 |

The OpenCV/NumPy choice repairs an observed compatibility failure, rather than upgrading solely for a newer version. The previously installed OpenCV 5 build lacked the required classifier. OpenCV 4.12 provides it and declares NumPy >=2,<2.3 for this Python version; NumPy 2.2.6 satisfies that constraint and passes the complete local application suite. Other direct application pins were retained. Package metadata and artifact hashes are recorded in [dependency-inventory.json](dependency-inventory.json).

The required TFace backbone code is now a minimal, deterministic local import from a pinned official Tencent source commit. It does not depend on a similarly named PyPI `torchkit` distribution or mutate global package-search paths. Complete upstream license/third-party texts, original Git blob IDs, SHA-256 hashes and the namespace-only import modification are recorded under [third_party/tface](../third_party/tface/README.md). Source-integrity tests verify those bytes, reconstructing the documented import change before hashing `model_irse.py`.

Trained-image and dataset preprocessing now both use `(pixel - 127.5) / 127.5`, matching the official implementation. Tests verify exact black/white endpoints of -1 and +1. Statistical demo feature scaling remains separate.

## Reproduce this environment

The exact hash lock targets **Windows x64 / CPython 3.11**, and includes runtime, development/audit tools, bootstrap tools and the optional CPU model backend. Each hash identifies the actual wheel resolved for this platform; the file is intentionally not a universal platform lock.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-bootstrap.txt
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-windows-py311.lock.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_runtime.py
.\.venv\Scripts\python.exe -m pytest -q
```

For a smaller demo environment, install `requirements.txt` and `requirements-dev.txt` after the bootstrap step. Install `requirements-model.txt` separately if CPU tensor support is wanted; it selects the official CPU index and does not install weights. The hash lock is the reproducible version of the complete tested environment.

Creation initially encountered a sandbox-specific permission error in the Windows sandbox TEMP folder. Setting process-local TEMP/TMP to a writable validation directory resolved it. Ordinary sandbox DNS also blocked external package metadata; supported network approval allowed official package downloads. These were environment limitations, not application failures. No security control was bypassed.

## Commands and outcomes

| Check | Outcome |
| --- | --- |
| Clean installation from exact resolved/hash-verified wheels | 77 installed Python distributions, including all transitive/developer/bootstrap packages; inventory matches all installed names and versions |
| `python -m pip check` in `.venv`, after CPU Torch and patched bootstrap tools | No broken requirements |
| `python scripts/check_runtime.py` in `.venv` | Exit 0; demo runtime ready, face-detection capability ready, pinned model backend ready; trained checkpoint missing; authentication remains false |
| Focused `tests/test_runtime.py tests/test_data_and_launchers.py` | 46 passed, one missing-PyTorch-only boundary test skipped because Torch is now installed; 83.43 s |
| Full suite at this validation checkpoint | 193 passed, one missing-PyTorch-only boundary test skipped, four subtests passed; 76.23 s. That boundary test was subsequently changed to use an isolated child-process import block, so the root completion report records the later consolidated total. |
| Ruff on runtime checker, new runtime tests, model imports and normalization changes | All checks passed |
| Final runtime-diagnostic error regressions | Two passed: broken OpenCV classifier assets and optional extractor import failures return unavailable statuses without tracebacks |
| `pip install --dry-run --require-hashes --no-deps -r requirements-windows-py311.lock.txt` | Exit 0; lock syntax, exact pins and cached Torch artifact hash resolve without changing installed packages |

All three tensor cases skipped in the earlier environment now run successfully: zero-feature normalization, constant attention and attention softmax. Additional tests exercise the actual pinned IR-18 forward/flip/attention path and a real IR-50 state-dict save/load/extraction round trip. The IR-50 checkpoint in this test is **randomly initialized test material**, stored only in pytest's temporary output. It proves architecture/API compatibility, not pretrained recognition quality. Real PyTorch meta operators also check the full IR-152/IR-SE-200 shape graphs, including the repaired 2048-channel group projection and 224-pixel input, without allocating gigabytes of weights.

## Current vulnerability scan and SBOM

`pip-audit 2.10.1 --local --vulnerability-service osv --strict` queried current [OSV](https://osv.dev/) advisory data. No `--fix` or ignore-vulnerability option was used. All **77 Python distributions** were included, with **zero collection skips**. The final result is **zero known matched vulnerabilities**, not a claim that the application, computer, model, or dependencies are vulnerability-free.

The first scan found seven unique advisories in the fresh environment's old ensurepip bootstrap tools: six for pip 24.0 and one for setuptools 79.0.1. OSV returned each twice, so raw pip-audit output reported 14 vulnerability records. These were packaging-tool findings; no application runtime dependency was flagged. The [raw before report](dependency-audit-before-bootstrap.json) is retained. Only the isolated environment's bootstrap tools were updated, to official hash-verified pip 26.2.1/setuptools 84.0.0. The [pip changelog](https://pip.pypa.io/en/stable/news/) and [setuptools history](https://setuptools.pypa.io/en/latest/history.html) were checked; the latter documents the MANIFEST exclusion Unicode-normalization fix referenced by [GHSA-h35f-9h28-mq5c](https://github.com/pypa/setuptools/security/advisories/GHSA-h35f-9h28-mq5c). The original setuptools advisory page still listed no patched release while OSV and the later changelog recorded the fix; the changelog was used to resolve that discrepancy. This manifest/source-build issue is not evidence of a remotely exploitable GuardFace endpoint on Windows.

After those updates, a fresh-cache complete OSV scan returned zero matched findings. A supplementary query checked upstream `torch==2.13.0`, in addition to the installed `2.13.0+cpu` local version, and also returned zero matched findings. Artifacts:

- [After scan: all 77 installed distributions](dependency-audit-osv.json).
- [Supplementary upstream Torch version scan](dependency-audit-torch-base.json).
- [CycloneDX 1.4 SBOM: 77 Python components](dependency-sbom.cdx.json).
- [Exact component inventory, dependency declarations, source URLs and wheel SHA-256 values](dependency-inventory.json).
- [Reproducible Windows/Python 3.11 wheel hash lock](../requirements-windows-py311.lock.txt).

To refresh the full scan/SBOM from this environment:

```powershell
.\.venv\Scripts\python.exe -m pip_audit --local --vulnerability-service osv --strict --progress-spinner off --format json --output docs/dependency-audit-osv.json
.\.venv\Scripts\python.exe -m pip_audit --local --vulnerability-service osv --strict --progress-spinner off --format cyclonedx-json --output docs/dependency-sbom.cdx.json
```

This scan covers advisory matching for installed Python distribution versions, including transitive, optional model, development/audit and bootstrap distributions. It does not inventory every embedded native library in wheels, scan the Windows OS/driver stack, browser/Node tooling, or establish provenance/licensing/security of absent trained weights. Vendored TFace source is attested separately by the checked file hashes/notices; it is not an additional pip distribution in the 77-component SBOM. Advisory services can contain duplicate, delayed, missing or version-range-inaccurate records. Refresh the scan after dependency changes and before deployment.

## Remaining model boundary

The final consolidated run after all follow-up changes passed **199 tests and four subtests, with zero skips**. See [AUTONOMOUS_FOLLOWUP.md](AUTONOMOUS_FOLLOWUP.md) for the exact final command environment and combined browser/storage/evaluation results. Earlier counts above are preserved as dated intermediate checkpoints.

No trained checkpoint was downloaded or substituted. The available environment now resolves the prior missing-Torch and missing-face-classifier blockers, but cannot establish pretrained accuracy, PAD attack resistance, calibration, enrollment identity binding, weight/data rights or privacy guarantees. `--checkpoint <reviewed-path> --require-model` makes the runtime checker fail unless an explicitly supplied checkpoint completes a finite compatible forward pass; even that forward pass grants no authentication capability. Real licensed model weights, independent identity data and physical camera/print/screen trials remain separate evidence requirements.
