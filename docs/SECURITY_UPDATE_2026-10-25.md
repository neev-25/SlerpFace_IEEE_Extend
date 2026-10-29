# Dependency security update — October 25, 2026

The [failed hosted run](https://github.com/neev-25/SlerpFace_IEEE_Extend/actions/runs/40194224633) used commit `91402cf848d003abd7a9c6e5484003d6d63326c4`. Installation, model provisioning and runtime checks passed. Strict OSV auditing then reported **50 advisory records across four packages**; Certifi's record was duplicated. These records are not 50 demonstrated exploitable paths in GuardFace. Later CI stages were skipped. The earlier audit-service correction worked: [the preceding hosted run](https://github.com/neev-25/SlerpFace_IEEE_Extend/actions/runs/38809119589) succeeded.

## Patched versions and compatibility

| Distribution | Previous | Updated | Reason |
| --- | --- | --- | --- |
| Certifi | 2026.7.22 | **2026.10.19** | Removes the emSign root certificate identified by [the maintainer advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2), also reported as PYSEC-2026-4320. |
| Filelock | 4.2.0 | **4.2.4** | The six reported findings are fixed from 4.2.1; subsequent maintenance also corrects relative-path lock hold state. See [maintainer advisories](https://github.com/tox-dev/filelock/security/advisories) and [release notes](https://github.com/tox-dev/filelock/releases). |
| Pillow | 12.3.0 | **13.0.0** | Resolves the reported decoder, memory-safety and decompression issues; see [upstream security and breaking changes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html). Requires Python 3.11+, which matches this project's target. |
| Setuptools | 84.0.0 | **84.1.0** | Fixes the Windows legacy wheel-installation namespace path escape, [GHSA-xmj3-9gf7-h52w](https://github.com/pypa/setuptools/security/advisories/GHSA-xmj3-9gf7-h52w). This is packaging-tool behavior, not evidence of an application RCE. |
| Streamlit | 1.63.0 | **1.66.1** | Compatibility update: 1.63.0 requires Pillow below 13, while [the selected release metadata](https://pypi.org/pypi/streamlit/1.66.1/json) permits Pillow below 14. Updating Pillow alone would create an inconsistent environment. |

The five Windows/Python 3.11 wheels came from official PyPI release metadata and were checked against published SHA-256 values before installation. Runtime pins, the Setuptools bootstrap pin and all five full-lock hashes were updated together. The other 72 distributions, including NumPy/OpenCV/Torch, retain their tested versions. No shared conda/system environment was changed. No vulnerabilities or packages were excluded from the audit.

The complete candidate lock resolved successfully before applying updates. One initial wheel transfer had a TLS record error; the verified retry succeeded. An attempted offline installation of the full lock still tried to fetch its explicit Torch URL and hit sandbox DNS restrictions. It changed no packages. The already validated five-wheel update subset was then installed locally with hashes and no dependency resolution; `pip check` confirms the complete installed environment remains consistent. All 77 installed versions match the refreshed inventory and lock.

`docs/dependency-audit-osv.json`, `docs/dependency-audit-torch-base.json`, `docs/dependency-inventory.json` and the CycloneDX SBOM are refreshed to the patched environment. Prior snapshots and raw hosted findings are retained in the external evidence directory. The scan covers Python distribution advisories; it does not audit every embedded native library, model graph, operating-system component or browser dependency.

## Later CI blockers addressed

The newly added scripts/tests had 40 Ruff errors that the failed audit prevented CI from reaching. Unused imports/bindings were removed, an ambiguous local name was clarified, and test constants were moved below imports. Direct-script imports that intentionally follow project-path initialization have narrowly documented `E402` annotations, following existing project conventions. Lint rules and the CI lint step remain enabled.

Offline webcam validation also constructed `VideoCapture` before checking `headless`, potentially opening a device during tests. Headless mode now uses no camera handle, and a regression explicitly forbids camera construction. Interactive camera behavior is unchanged. The regression suite uses synthetic/public fixtures; its "physical" helper names do not establish real physical attack or camera accuracy.

The existing real-upload browser test also assumed advisory comparison while the new dual-tier commit enables its optional strict heuristic gate by default. The captured page showed `ZERO_LEAKAGE_TRIGGERED` on the second public portrait, not an application crash or dependency incompatibility. The comparison test now explicitly selects advisory mode; a separate browser case verifies that the default configured gate still reports interception without producing a comparison cosine. The app's default policy was not relaxed to make the test pass. Failure screenshots/body text are now saved by the real-comparison check for future diagnosis. Neither test claims the heuristic proves physical spoofing or absence of intermediate feature computation.

## Verification

Evidence root: `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\security-update-2026-10-25`.

| Check | Outcome |
| --- | --- |
| Full candidate dependency resolution and exact wheel hashes | Passed; only the five intended versions change |
| Installed environment / `pip check` | 77 distributions; no broken requirements |
| Fresh-cache strict OSV audit | **77 scanned, zero skipped, zero known matched vulnerabilities** |
| Supplementary public Torch version audit | Passed, zero known matched findings; official CPU build remains included in the installed scan |
| Refreshed CycloneDX SBOM | Passed strict generation |
| Ruff | All first-party checks passed |
| Full patched Python suite | **304 passed, seven subtests passed, zero skips**, 79.30 seconds |
| Actual trained-model smoke comparison | `RESEARCH_COMPARISON`, processed true; public source/derivative cosine 0.970822, experimental similar |
| Runtime, synthetic evaluation, report-generation smoke checks | All passed; prototype/presentation backends ready and all seven demo PNG figures generated |
| Browser compatibility | **11/11 passed**, zero page errors; advisory real comparison and configured strict interception tested separately |
| Post-cleanup checks | **15 focused tests passed**; Ruff, `pip check`, diff whitespace and installed/inventory/lock/SBOM consistency passed; model paths and both retained images verified |

Streamlit 1.66's AppTest runner creates an asyncio loopback socket. The restricted sandbox stalled inside Python's socket-pair creation, rather than an application function. The stalled validation process was stopped after verifying its identity; rerunning with supported loopback access passed the complete suite. No application workaround or skip was introduced. Tests use dedicated writable TEMP/TMP directories to avoid unrelated Windows sandbox cleanup errors.

Full-suite log: `pytest-final.log`. Advisory reports: `audit-before.json`, `audit-after.json`, `torch-audit-after.json`. Actual model smoke: `real-pipeline.log`. Network/setup retry logs are retained separately. Existing application source behavior was preserved apart from lint-only edits and preventing physical camera access during offline tests.

The complete browser result is `browser-final/results.json`. It verifies responsive light/dark layouts, labelled controls/keyboard tabs, missing/corrupt inputs, session isolation, generated evaluation, real pretrained upload comparisons, result retention/invalidation, and configured strict rejection. Earlier real-upload timeout logs are retained as diagnosis of the stale policy assumption; they are not counted as passing checks. Browser validation uses public fixtures and does not measure camera, identity or physical presentation accuracy.

## Requested cleanup

After validation, the user requested removal of unused files. Removed **71 disposable files (1,386,210 bytes)** from 12 bytecode/lint cache directories and four obsolete test-output folders, plus 12 pytest-created links verified to resolve inside those disposable folders. Empty `output/` was removed. Tracked bytecode is now marked for deletion so the next commit stops versioning it; existing ignore rules already prevent newly generated caches from being added.

The first cleanup request including the old `temp_img1.jpg`/`temp_img2.jpg` files was rejected by automatic approval review because those images contain previously preserved user changes. The user explicitly chose **Keep both images**; both remain untouched. Runtime environments, verified model weights, first-party/vendor source and notices, tests/fixtures, paper/report assets and validation documentation were retained. No unverified linked target outside the intended disposable subtree was removed. The exact plan, hashes and outcome are in `cleanup-manifest.json` and `cleanup-result.json` under the evidence root.

## Publish and verify

The new hosted commit has not yet been run. Commit/push the updated manifests, lock, refreshed audit/inventory artifacts, lint fixes, headless regression and this report. GitHub will install the new pins and execute the unchanged strict security gate before later checks. Re-running the old commit retains its vulnerable pins. Restart any already running Python dashboard after updating its environment so it loads the patched libraries.
