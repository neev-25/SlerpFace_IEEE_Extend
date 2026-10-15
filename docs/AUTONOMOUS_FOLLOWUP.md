# Autonomous remediation follow-up

This is the earlier safety/tooling checkpoint. The subsequent functional pretrained camera/upload implementation is recorded in [FUNCTIONAL_PROTOTYPE.md](FUNCTIONAL_PROTOTYPE.md); statements below about absent trained weights or stopped extraction refer to the earlier state and optional original IR-50 model.

Completed October 16, 2026 (Asia/Calcutta), in `C:\Documents\AI_Project`. The user authorized locally actionable fixes and testing after the initial investigation. This record supersedes environment and validation limitations in the earlier [AUDIT_FIXES.md](AUDIT_FIXES.md) checkpoint; that checkpoint preserves the original finding IDs and per-fix evidence.

## Result and remaining boundary

The research application now has a reproducible, tested local runtime; working face-presence and tensor backends; pinned upstream source and notices; real browser checks; offline evaluation tooling; and a separate encrypted local research-store prototype. The original critical false-authentication behavior is **mitigated by disabling authentication**, not resolved by establishing an accurate matcher. Every pipeline decision remains false and diagnostic cosine values cannot authorize identity. Static-image heuristics remain unvalidated for physical liveness.

No trained weights or real biometric dataset were supplied or substituted. Random-weight checkpoint tests prove compatibility only. The remaining research/product requirements below are material; passing local tests does not make the project ready to authenticate people.

| Workstream | Completed locally | Still required |
| --- | --- | --- |
| Runtime and model code | Isolated Python 3.11 `.venv`, working OpenCV 4.12 classifier, official CPU Torch, minimal pinned Tencent backbone, source hashes/notices, corrected trained preprocessing, executed tensor/shape/checkpoint tests | Trusted compatible pretrained weights, verified weight rights, pretrained accuracy measurements |
| Evaluation | Bounded offline manifest evaluation, subject/image separation, calibration-only threshold selection, held-out class counts/uncertainty, checkpoint/image digest checks, explicit synthetic labels | Independent real identities, physical print/display attacks, supported sensors and acquisition conditions, justified acceptance criteria |
| Research storage | Windows user-scope DPAPI/SQLite payload encryption, context binding, strict validation, transactional rotation/revocation, real DPAPI and CLI tests | Product identity binding, access control, retention/recovery, compromise and anti-rollback policy, privacy/adversarial evidence |
| Browser, dependencies and CI | Nine real browser flows, contrast repair, isolated-session check, 77-component Python inventory/hash lock/SBOM, strict advisory scan, local thirteen-step CI workflow | Reviewed publication before hosted CI can execute; physical devices/cameras, screen readers, sustained load and deployment validation; native/OS/browser inventory |
| Provenance and ownership | Official source commit/hash checks, complete relevant upstream license notices, documented official checkpoint location and unresolved weight terms | Owner choice of root project license and product policy; verified permission for actual weights/data |

## Confirmed follow-up defects and repairs

The initial sixteen confirmed findings and two suspected issues are tracked in [AUDIT_FIXES.md](AUDIT_FIXES.md). Closed authentication is a mitigation for the unsafe identity/PAD claims, while several other findings are implementation repairs. Subsequent evidence removes the earlier missing-backend, skipped-tensor, clean-install and unscanned-dependency limitations; it does not remove biometric evidence requirements.

- **GF-F01 — Low, confirmed and fixed: inspector contrast.** `app_demo.py` rendered diagnostic numbers through Streamlit's JSON widget; the light-theme foreground measured 3.204:1 against white. Four labelled plain-text values now inherit the normal theme color. All four measured 12.531:1 in light and 18.106:1 in dark. The browser regression requires at least 4.5:1. See [OPERATIONAL_VALIDATION.md](OPERATIONAL_VALIDATION.md).
- **GF-F02 — Low, confirmed and fixed: malformed split traceback.** `modules/dataset_validation.py::load_manifest` tested a supplied split against a dictionary before checking its type. A JSON list/object therefore raised an uncaught `TypeError`. The parser now requires a string first. Three CLI regressions verify a controlled exit 2 and no output file for malformed split values.
- **Trained preprocessing mismatch — confirmed and fixed.** The extractor/data loader used a divisor of 128, while the pinned source expects 127.5. Both trained paths now map byte endpoints exactly to -1 and +1; regression checks cover those endpoints. Statistical-demo scaling remains separate. This correction has compatibility evidence, not an accuracy result.
- **Environment defects — resolved in isolation.** The old shared OpenCV build lacked the required classifier. A fresh compatible OpenCV/NumPy pair restores it. The first scan also identified seven unique advisories in the new environment's old bootstrap pip/setuptools versions; only the isolated bootstrap tools were patched using official artifacts/advisory evidence. The final strict scan includes 77 distributions, zero collection skips and zero known matched vulnerabilities. These are environment findings, not evidence of a remotely exploitable application endpoint. See [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md).

New evaluator/storage tooling also received peer review and boundary tests: renamed image copies cannot inflate pair counts or cross splits, a capture cannot have conflicting subjects, real identity evaluation needs an explicit checkpoint, changing weights/images aborts reports, and failed/concurrent storage mutations preserve transaction integrity.

## Consolidated validation

The final Python run used `.venv\Scripts\python.exe` (Python 3.11.16), disabled unrelated pytest plugin autoload, prevented bytecode writes, and assigned process-local TEMP/TMP plus pytest's basetemp to a dedicated writable evidence directory.

| Check | Final outcome |
| --- | --- |
| Complete existing/new Python suite | **199 passed, four subtests passed, zero skips**, exit 0, 51.67 seconds |
| Offline evaluator coverage within that run | 25 cases, including malformed-manifest CLI errors, leakage, labels, calibration ties, mutation, backend boundaries and overwrite protection |
| Local research-store coverage within that run | 30 cases, including actual Windows DPAPI, tampering, rollback, concurrent rotation and the real CLI lifecycle |
| Tensor/model checks | Previously skipped cases execute; real IR-18 forward/attention, random IR-50 checkpoint round trip, IR-152/200 meta shape graphs and normalization endpoints pass |
| Real browser run | **9/9 passed** on Chrome 155 / Playwright 1.62.1; no page errors or visible app exceptions; desktop and 390-pixel light/dark views, labelled controls, keyboard tabs, missing/corrupt inputs, session isolation and 105-pattern UI evaluation |
| `python -B -m ruff check .` | All checks passed |
| `python -B -m pip check` | No broken requirements |
| `python -B scripts/check_runtime.py` | Exit 0: demo, classifier and pinned model backend available; checkpoint missing; authentication false |
| Dependency inventory/advisory scan | 77 Python distributions, zero skips, zero known matched OSV findings; supplemental upstream Torch-version query also clear at scan time |
| Hash lock | Exact platform-specific wheel hashes, complete installed inventory match and hash-verified dry run pass |
| Launcher and script checks | Launcher status/CWD tests pass; default launcher selects `.venv` and synthetic evaluation exits 0; PowerShell/JavaScript parsing passes |
| Reports/CLI | Earlier per-fix checks produced all seven requested illustration figures, stopped rejected inputs before extraction and verified 0/1/2 exit meanings; these regressions remain in the consolidated suite |
| Local CI definition | YAML parses with thirteen steps and read-only contents; native failure propagation checked. Hosted workflow **not executed**: reviewed changes remain unpublished |
| Working tree | Existing user artifacts retain their original SHA-256 hashes; no commit, push or remote workflow dispatch |

Final Python log: `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\autonomous-final\pytest-final.txt`. Browser results/screenshots: the same evidence root's `operational-validation\browser-final` directory. Earlier failed harness/setup attempts and dated intermediate counts are retained separately rather than counted as successful final checks.

The earlier 194-test run passed but emitted a Windows sandbox TEMP cleanup traceback at interpreter shutdown. A process-local writable TEMP/TMP resolved that environment issue. A clean 196-test run and the final 199-test run have no such traceback. Network and loopback checks used supported automatic approval review; no rejection was bypassed. Shared Python/conda environments were not changed.

To run the Python checks from the workspace:

```powershell
.\.venv\Scripts\python.exe -B scripts/check_runtime.py
.\.venv\Scripts\python.exe -B -m pip check
.\.venv\Scripts\python.exe -B -m ruff check .
.\.venv\Scripts\python.exe -B -m pytest -q
```

If sandbox TEMP is unavailable, set process-local TEMP/TMP to an existing private writable directory and give pytest a new `--basetemp` there. [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md) records reproducible installation and scan commands; [OPERATIONAL_VALIDATION.md](OPERATIONAL_VALIDATION.md) records browser commands and [TEMPLATE_STORE.md](TEMPLATE_STORE.md) records the explicit store lifecycle.

## Coverage and exclusions

The original audit inventoried 39 non-Git files and reviewed all 24 first-party text files, including all sixteen then-existing Python files. Its directory/file checklist remains in the original audit evidence. The follow-up reviewed changed first-party code, added helpers/CLIs/tests, launchers, manifests, hidden configuration, CI and documentation. The new evaluation module/CLI and store received separate peer review. Python tests exercise real code across image decoding, PAD preparation, transforms, UI state, report generation, extraction, launcher processes, DPAPI and SQLite.

| Review status | Scope |
| --- | --- |
| Detailed | `verify_with_antispoof.py`, `app_demo.py`, `modules/image_input.py`, `modules/antispoof/*.py`, `modules/model.py`, `extract_template/*.py`, report/sample/evaluation entry points, both launchers, new `modules/dataset_validation.py`, `evaluate_dataset.py`, `modules/template_store.py`, `manage_templates.py`, `scripts/check_runtime.py`, `scripts/browser_checks.cjs`, first-party tests and CI/manifests/configuration |
| Inspected with a narrower purpose | Vendored TFace files for exact source integrity, license scope, import changes and executed compatibility; generated inventory/audit/SBOM JSON for schema, component counts, finding scope and reproducibility; browser screenshots for the specified layouts and operation controls |
| Not reviewed line by line | `.git` internals, `.venv`/dependency caches, installed dependency source, generated images/reports, bytecode, binary/PDF/image assets and native wheel libraries. Bundled images were exercised as relevant inputs; the included paper supplies research context, not code certification |
| Not exercised | Trained recognition with official weights; real held-out identities; physical spoof/camera trials; complete screen-reader/mobile-device/cross-browser accessibility; sustained resource/load tests; hosted CI and deployment; inversion/reconstruction/unlinkability research |

No complete semantic/security audit of PyTorch, OpenCV, Streamlit or upstream TFace is implied. Advisory matching does not cover every embedded native library or the OS/browser/Node stack. Windows-only DPAPI behavior was exercised here; non-Windows closed behavior is tested, not a portable encrypted-store implementation.

The pre-existing modified `temp_img1.jpg`, `temp_img2.jpg` and untracked `__pycache__\evaluate_antispoof.cpython-311.pyc` were preserved byte for byte. Existing modifications were not reset. Generated databases are ignored, no user store was initialized automatically, and browser/server processes created for validation were closed.

## Prioritized continuation

1. **Before any authentication use:** keep the closed decision boundary. Obtain the actual reviewed/compatible trained checkpoint and applicable rights; verify its digest and run `scripts/check_runtime.py --checkpoint <path> --require-model`. This proves a finite forward pass only.
2. **Next evaluation stage:** acquire independent identity and physical presentation data with known provenance, define acceptance criteria, and supply a schema-1 manifest to `evaluate_dataset.py`. Measure held-out performance and detection failures; investigate common-noise distortion and correlated trials. Do not install an empirical threshold into the live pipeline automatically.
3. **Owner/product decisions:** choose the root project license, intended authentication/enrollment policy, retention, recovery, supported platforms and adversary model. Upstream source notices are included, but actual weight/data rights and a project-wide license were not invented.
4. **After reviewed publication:** run the prepared hosted workflow and inspect actual runner results. Extend device/accessibility, deployment and sustained load coverage. Refresh advisories and expand the native/browser/OS inventory for the deployment target.
5. **Longer-term research:** test inversion/reconstruction, linkage, compromise recovery and context rotation against explicit threats. The DPAPI prototype and randomized transforms are tools for these experiments, not evidence that their privacy goals have been achieved.

The project is substantially more reproducible and its local failure boundaries are tested. Its health is appropriate for a clearly labelled research demonstration. The remaining checkpoint/data, scientific validation, owner-policy and deployment requirements prevent a claim of production biometric readiness.
