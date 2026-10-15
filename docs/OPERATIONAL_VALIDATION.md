# Operational validation

This is the earlier nine-flow browser checkpoint. Later real pretrained upload comparisons and the updated UI are recorded in [FUNCTIONAL_PROTOTYPE.md](FUNCTIONAL_PROTOTYPE.md); the workflow now also provisions/verifies the pinned model assets before running checks.

## First hosted run: CPU Torch audit compatibility correction

The [first hosted run](https://github.com/neev-25/SlerpFace_IEEE_Extend/actions/runs/38807712946), on commit `5ce2e5da278e9860b0fff7dd51fa2057c82ddcc0`, completed installation, both model provisioners and runtime checks successfully. It stopped at dependency auditing: pip-audit's default PyPI service requested the exact `torch==2.13.0+cpu` release, which is hosted on the official PyTorch index rather than PyPI. The strict collection error was a service/packaging compatibility problem; the later lint/test/browser steps were skipped, not failed or validated by that run.

The workflow now explicitly uses strict OSV auditing for the installed environment, matching the service used in earlier local validation. It also derives Torch's public upstream version from installed metadata (`packaging.version.Version(...).public`) and audits that exact version through OSV using `--no-deps --disable-pip`. This preserves the CPU wheel and checks upstream advisories without installing/resolving packages, excluding Torch or disabling strict collection checks. The temporary query requirement is not an installation manifest; wheel integrity remains covered by the existing hash lock. The supplementary no-hash informational warning is expected for a query-only input.

The exact extracted PowerShell workflow block passed live locally with exit 0 and no known matched findings for either audit. YAML and PowerShell syntax, immediate failure propagation (injected exits 23, 17 and 19), and whitespace checks passed. PowerShell terminating errors also stop the step if writing the temporary query file fails. No package versions or application code changed. Evidence is under `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\ci-audit-fix`.

These corrected commands have not yet run on a new hosted commit. Commit/push the updated workflow to trigger that run; rerunning the old commit alone retains its old workflow. Historical zero-remote-workflow statements below describe the earlier inspection, before the user's publication.

Validated on October 16, 2026 (Asia/Calcutta), in `C:\Documents\AI_Project`. This is local operational evidence for the research demonstration. It does not establish identity accuracy, physical presentation-attack resistance, or deployment readiness.

## Real browser results

`scripts/browser_checks.cjs` launches a Streamlit process bound to `127.0.0.1` on an ephemeral port, runs headless Playwright checks, writes screenshots and JSON results to a selected local directory, and closes its browser/server. Windows helpers are hidden. It refuses a non-localhost `--base-url`. It uses generated preset patterns and one intentionally invalid PNG payload. No camera permission is granted and no camera is activated.

The final full run against the freshly provisioned Python 3.11 `.venv` passed **9/9 checks**, with **zero browser `pageerror` events** and no visible Streamlit application exceptions. Browser: installed Chrome `155.0.8059.41`; Playwright `1.62.1` from the bundled tooling. The fresh runtime contains Streamlit `1.63.0`, NumPy `2.2.6`, and full OpenCV `4.12.0`. Runtime/dependency verification is recorded separately in `RUNTIME_VALIDATION.md`.

| Check | Observed result |
| --- | --- |
| Desktop: 1440 × 1050, light | Research warning, inspector, inputs, diagnostics and explicit operation render. Document/body width stays 1440; no page-wide horizontal overflow. |
| Narrow: 390 × 844, light | Content stacks and text wraps; document/body width stays 390. Every tab is reachable through the scrollable tab bar. |
| Narrow: 390 × 844, dark | Same layout bounds; warning, controls and labels visually inspected against the dark theme. |
| Accessible labels and keyboard | All four sliders, checkbox, checkpoint textbox and inspector radio group have names. Focusing the first tab, pressing ArrowRight and Enter selects Pair Comparison. |
| Empty inputs | Inspector Upload/Camera and three mixed pair Upload/Camera/Preset combinations disable the relevant action while an input is missing. |
| Corrupt upload | A browser upload request containing `corrupt.png` with invalid bytes displays the recoverable image error and leaves processing disabled. |
| Unavailable and demo behavior | The default operation remains unverified. Explicit synthetic features yield `DEMO_ONLY`, two labelled diagnostic cosines, and “Identity remains unverified.” No identity-success claim appears. |
| Session isolation | Three independent browser contexts preserve different input modes and synthetic-feature selections without propagating state. This is a focused concurrency smoke test, not a load benchmark. |
| Synthetic benchmark | The real UI processes 105 generated patterns and renders BPCER/APCER/ACER with the research caveat. |

Browser inspection found one confirmed UI issue: inspector JSON numeric text used `rgb(133,153,0)` on white, a measured contrast ratio of **3.204:1**. `app_demo.py` now renders those four diagnostics as labelled plain text with four decimals, inheriting the normal theme color. The final measured ratios are **12.531:1 in light theme** (`rgb(49,51,63)` on white) and **18.106:1 in dark theme** (`rgb(250,250,250)` on `rgb(14,17,23)`), for all four values. The browser suite requires at least 4.5:1 for each diagnostic value in both tested themes. This is a focused contrast check, not a claim of full accessibility conformance. Input images and authentication behavior were unchanged. The existing dashboard suite passed all ten tests and four subtests after this change; focused Ruff and JavaScript syntax checks also pass.

Initial browser-harness attempts failed because Streamlit's visually hidden native inputs are covered by their clickable labels, slider accessible names currently repeat their label, and immediate assertions raced server reruns. The final harness clicks visible labels, waits for actual outcomes, accepts the observed accessible name, and allows up to 60 seconds for a new session to initialize. A separate keyboard reproduction confirmed normal ArrowRight/Enter behavior. Those initial harness failures are preserved in external logs rather than presented as application defects.

## Reproduction and artifacts

With a normal local Playwright install and downloaded Chromium:

```powershell
node scripts/browser_checks.cjs --python .venv/Scripts/python.exe --out C:\research\fresh-browser-results
```

The local audit used the bundled Node/package paths, without installing browser tooling into this repository:

```powershell
$env:NODE_PATH = 'C:/Users/meetk/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'
$env:GUARDFACE_BROWSER_CHANNEL = 'chrome'
& C:/Users/meetk/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe scripts/browser_checks.cjs --python C:/Documents/AI_Project/.venv/Scripts/python.exe --out C:/Users/meetk/.codex/visualizations/2026/10/15/01a140ab-599e-7733-8a65-904dbf848f56/operational-validation/browser-final
```

Add `--case parallel` to repeat only the session-isolation check. `--base-url http://127.0.0.1:<port>` can target an already running local server; the script then owns only its browser, not that external server. Unset `GUARDFACE_BROWSER_CHANNEL` to use Playwright's installed Chromium.

Evidence directory: `C:\Users\meetk\.codex\visualizations\2026\10\15\01a140ab-599e-7733-8a65-904dbf848f56\operational-validation\browser-final`. The preceding `browser-clean-runtime` directory preserves the complete passing run before the focused contrast fix; `browser-cleanup-check` repeats session isolation after Windows process-tree cleanup was added.

The directory contains `results.json`, `server.log`, top/operation screenshots for each of the three viewport/theme cases, and screenshots of missing inputs, corrupt upload, unverified pair demonstration, and the synthetic benchmark. Streamlit scrolls inside its application container, so a “fullPage” screenshot alone does not capture every off-screen element; the operation screenshots explicitly scroll to the lower controls. Top and lower narrow light/dark screenshots were visually inspected. Source/result inspection covers the remaining screens.

The sandbox initially denied localhost socket access. The automatic approval reviewer permitted the necessary loopback server/browser commands; there was no rejected approval and no attempt to bypass a rejected boundary. Early Windows server logs include a connection-reset callback when browser contexts close; this did not produce a page exception or fail the completed flows. Test subprocess cleanup was checked separately.

## CI checks and remote status

The local `.github/workflows/checks.yml` retains read-only repository permissions and disabled persisted checkout credentials. It first installs `requirements-bootstrap.txt` with `--require-hashes`, then the complete `requirements-windows-py311.lock.txt` with `--require-hashes`. The platform-specific lock records the 77-package Windows x64/Python 3.11 environment, including CPU Torch, development tools and the audit tool; installation does not provision trained weights. CI checks concrete runtime capabilities with `python -B scripts/check_runtime.py` and separately requires Torch and the SlerpFace module to import, so missing optional imports cannot quietly skip the tensor tests. No checkpoint is required for those code/capability checks.

The workflow runs strict installed-dependency auditing with the OSV service (`python -m pip_audit --local --vulnerability-service osv --strict --progress-spinner off`), Ruff, pytest, synthetic evaluation, report generation, and the same browser script. CPU Torch's `+cpu` wheel is hosted by PyTorch rather than published as that exact PyPI release, so the default PyPI audit service cannot look it up. The audit step additionally derives Torch's public version from installed metadata and queries an exact temporary requirement with `--no-deps --disable-pip`, retaining strict failure behavior without installing packages or excluding Torch. Browser tooling installs into `runner.temp` with Playwright pinned to `1.62.1`, followed by Chromium installation. Node 24 is selected explicitly; automatic npm caching is disabled because this project has no npm application lockfile. Each native command propagates failure; multi-command PowerShell steps explicitly check `$LASTEXITCODE` before proceeding.

`node --check scripts/browser_checks.cjs` passed. The final workflow parses as YAML with thirteen steps and `contents: read`; its PowerShell blocks parse with no syntax errors, and native failure propagation was checked locally. The actual local browser script is exercised above; the setup actions and hosted runner have not run for these uncommitted changes. Lock installation, full local regressions, capability and vulnerability results are recorded in `RUNTIME_VALIDATION.md`. The repository currently has **zero remote workflows and zero workflow runs** according to the read-only GitHub API:

```powershell
gh run list --limit 5 --json databaseId,name,status,conclusion,headSha,url
gh api repos/neev-25/SlerpFace_IEEE_Extend/actions/workflows --jq '{total_count,workflows:[.workflows[]|{name,path,state}]}'
```

Results were `[]` and `{"total_count":0,"workflows":[]}`. No commit, push, workflow dispatch, rerun, or remote publication was performed. A later authorized publication of the reviewed changes is required before hosted CI can provide evidence about them.

The current action tags were checked against primary sources: [checkout releases](https://github.com/actions/checkout/releases), [setup-python releases](https://github.com/actions/setup-python/releases), and [setup-node usage](https://github.com/actions/setup-node). Browser installation follows [Playwright's CI documentation](https://playwright.dev/docs/ci).

## Remaining operational limits

This covers Chromium-based headless desktop rendering and a narrow simulated viewport. It does not substitute for real mobile devices, Firefox/WebKit coverage, a screen-reader audit, a comprehensive contrast assessment, all keyboard paths, touchscreen behavior, or zoom/reflow checks. The framework currently repeats slider names in the accessibility tree; controls remain labelled and operable, but screen-reader phrasing was not assessed. Sidebar interaction on narrow screens and physical camera capture remain outside this pass.

Three isolated sessions do not establish throughput, saturation, memory stability under prolonged use, multi-host behavior, or deployment limits. These browser/CI checks introduced no public server or hosted service. A separate local research-store prototype is documented in [TEMPLATE_STORE.md](TEMPLATE_STORE.md); it is not connected to the dashboard or authentication. Independent identity/PAD data, physical hardware trials, trained-checkpoint validation, and owner-controlled license/product decisions remain separate work.
