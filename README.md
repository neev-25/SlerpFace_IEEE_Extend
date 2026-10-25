# GuardFace face comparison prototype

GuardFace performs local experimental face comparison with pretrained YuNet detection, five-landmark alignment and OpenCV SFace embeddings. It also displays a learned MiniFASNet presentation estimate, texture/frequency diagnostics and SLERP/dropout transformations inspired by the included SlerpFace paper. **Authentication is disabled in every mode.** Similarity labels are prototype measurements, not permission to access an account or proof of live capture.

The webcam/upload path now extracts real trained features and reports `similar`, `different` or `uncertain`. A low texture score is advisory and no longer prevents a real face comparison. Synthetic presets remain a separate explicitly selected illustration. The functional implementation and latest validation are recorded in [docs/FUNCTIONAL_PROTOTYPE.md](docs/FUNCTIONAL_PROTOTYPE.md).

The previous production-readiness, standards-compliance, inversion-resistance, accuracy, and throughput claims were not substantiated by this checkout and have been withdrawn. The bundled samples are generated illustrations, not independent biometric subjects or physical presentation trials.

## Setup

The October 25 security refresh updates Pillow/Streamlit and three affected supporting packages together. See [the security update and validation record](docs/SECURITY_UPDATE_2026-10-25.md); use the current manifests/hash lock rather than the historical version tables in earlier audit checkpoints.

Use a dedicated Python 3.11 environment. This workspace already has the tested `.venv` and verified pretrained face/presentation models. For a fresh checkout, create the environment and explicitly provision the models:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --require-hashes -r requirements-bootstrap.txt
python -m pip install --require-hashes -r requirements-windows-py311.lock.txt
python -B scripts/provision_face_models.py
python -B scripts/provision_pad_model.py
python -B scripts/check_runtime.py --require-prototype --require-presentation
```

The hash lock targets Windows/Python 3.11 and includes runtime, development, audit and CPU tensor packages. The smaller manifests describe dependency groups, not a cross-platform hash lock. The setup scripts download exact pinned official model bytes with SHA-256 checks; inference never downloads anything or sends photographs. Model binaries are ignored under `models/opencv/` and `ckpt/presentation/`. No GPU is required. Source and model notices are retained under `third_party/`; see [pretrained face provenance](docs/PRETRAINED_FACE_BACKEND.md) and [presentation model provenance](docs/PRESENTATION_MODEL.md).

## Run the dashboard and CLI

From the project directory:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run .\app_demo.py
.\.venv\Scripts\python.exe -B .\verify_with_antispoof.py --img1 C:\data\reference.jpg --img2 C:\data\probe.jpg
python -B .\verify_with_antispoof.py --demo --img1 .\samples\genuine_personA_1.jpg --img2 .\samples\genuine_personA_2.jpg
python -B .\verify_with_antispoof.py --help
```

For an advisory real-face comparison, leave **Use synthetic feature demo** unchecked and the optional checkpoint path blank, and turn off **Enforce Zero-Leakage Protocol**. That optional strict heuristic policy is enabled by the dual-tier extension and can reject a photo before a comparison result; its texture score is not a validated live/spoof decision. In **Pair Comparison**, choose Upload or Camera for both inputs, supply one clear face per image, and click **Run pair comparison**. The provisioned SFace backend requires no checkpoint entry. It shows original trained cosine and an experimental label; a +/-0.03 band around the adjustable 0.363 OpenCV example threshold is labelled uncertain. This operating point is not calibrated to the user's camera or population. Transformed cosine is displayed separately and never determines the label.

The inspector also shows aligned-face metadata and an optional learned presentation estimate. Those softmax values are not calibrated liveness probabilities; a still image cannot establish physical presence. Missing/multiple/small faces produce recapture guidance. Low FFT/texture scores do not declare a real photo spoofed. Results persist across unrelated reruns and invalidate when their inputs/settings change. Camera captures appear once. Images remain in memory; the old shared `temp_img1.jpg` and `temp_img2.jpg` paths are not written, and existing files are preserved.

`--demo` explicitly selects untrained image-statistics features and disables face-presence checks for drawings. Output is labeled `DEMO_ONLY`, `authenticated` remains false, and `is_match` remains unset. Cosine scores are diagnostic values, with no identity threshold decision. A single-image run is a transient transform demonstration; it neither enrolls nor persists a credential.

Without `--demo`, the pipeline requires exactly one YuNet-detected face, aligns original pixels and extracts trained SFace features. Missing/incompatible/unverified model bytes return `MODEL_UNAVAILABLE` with setup guidance, never statistical fallback. The original `AntiSpoofDetector` Haar mode remains available to legacy heuristic experiments, but it is not the real comparison's face backend. Neither detector establishes liveness. Older shared Python environments were left unchanged.

The supported verification CLI uses these process statuses:

| Exit | Meaning |
| --- | --- |
| 0 | Trained research operation or explicit synthetic demo completed; no identity authenticated |
| 1 | Face/input rejected or no operation completed |
| 2 | Invalid input, unavailable dependency/model, or another operational error |

`test_my_image.py` forwards to this CLI and has the same arguments and status contract.

## PowerShell launchers

```powershell
.\run_guardface.ps1 -Mode demo -PythonExe .\.venv\Scripts\python.exe
.\run_guardface.ps1 -Mode cli -PythonExe .\.venv\Scripts\python.exe
.\run_guardface.ps1 -Mode eval -PythonExe .\.venv\Scripts\python.exe
.\run_guardface.ps1 -Mode test -PythonExe .\.venv\Scripts\python.exe
```

The launcher resolves project paths from `$PSScriptRoot`, so it can be invoked by absolute path from another working directory. It prefers this workspace's `.venv/Scripts/python.exe`; if absent, it uses `python` on PATH. An explicit `-PythonExe` overrides that selection; use an absolute path when invoking it elsewhere. Modes are dashboard (`demo`), two bundled diagnostic comparisons (`cli`), synthetic evaluation (`eval`), and pytest (`test`). Native command failures are propagated; `cli` preserves rejection status across its two runs. Samples are never generated automatically.

`run_pipeline.ps1` is a compatibility notice for the absent original encryption/matching stages. It exits 2 before extraction or output writes and prints supported alternatives. It cannot restore the missing stages.

## Optional original SlerpFace model research

The default functional prototype uses trained SFace, whose 128-value vector is reshaped into eight groups of 16 for experimental transforms. This is **not** the official SlerpFace network's learned spatial grouping. The original Tencent backbone subset is also bundled and tensor-tested, but its IR-50 trained checkpoint remains optional and is not supplied. To reproduce that architecture's research path:

1. Obtain the intended model implementation and weights from their trusted source, retaining provenance, license, and file hashes. A similarly named package is not sufficient evidence of compatibility.
2. Use the verified CPU environment, or install a PyTorch build compatible with the intended platform/runtime. Check the checkpoint's architecture and preprocessing. The trained path now matches the official byte-to-[-1,1] normalization.
3. Use the pinned `third_party.tface.torchkit` implementation, not a similarly named external package. The shared extractor expects IR-50 with 112 by 112 RGB input and 16 group channels; extraction tooling also exposes other explicitly selected backbone depths.
4. Supply an explicit checkpoint path when checking the non-demo CLI:

```powershell
python -B .\verify_with_antispoof.py --img1 C:\data\reference.jpg --img2 C:\data\probe.jpg --model_path C:\models\compatible-ir50.pth
python -B -m extract_template.verification --help
python -B -m extract_template.verification --model_path C:\models\compatible-ir50.pth --backbone IR_50 --data_root C:\data\verification --output_dir C:\research\templates
```

The first command detects/aligns the face and runs the explicitly selected compatible checkpoint. The SFace default threshold does not classify an IR-50 model: without a separately chosen operating point its experimental label remains uncertain. Authentication stays disabled. Direct legacy feature exports expect prepared/aligned research images and sensitive output handling. Dataset input is `data_root/val/` with named image folders and annotations `0|1 image1 image2`; malformed rows and missing images are errors. See [MODEL_PROVENANCE.md](docs/MODEL_PROVENANCE.md) for the original architecture's source/checkpoint link and unresolved weight rights.

## Independent evaluation and local template experiments

`evaluate_dataset.py` evaluates an explicit manifest, separates calibration/test subjects and captures, selects thresholds on calibration only, and reports held-out error counts with uncertainty. CLI identity extraction defaults to the provisioned SFace backend; use `--backend ir50 --model <checkpoint>` for the original architecture. Real feature inputs are aligned consistently with the dashboard. Synthetic fixtures require `--demo`. Presentation rates from this tool are explicitly the legacy static-image heuristic, not a validation of MiniFASNet's advisory estimates. See [DATASET_VALIDATION.md](docs/DATASET_VALIDATION.md).

`manage_templates.py` provides a separate Windows research store with user-scope DPAPI encryption, opaque IDs, atomic rotation and revocation. Its CLI deliberately enrolls only explicit untrained demonstration material and never authenticates. It stores neither raw images nor original feature vectors. See [TEMPLATE_STORE.md](docs/TEMPLATE_STORE.md) for commands, tests, same-account/backup limitations and the local-owner threat model.

## Evaluation and report figures

```powershell
python -B .\evaluate_antispoof.py --samples-per-class 30 --seed 42
python -B .\generate_report_images.py --demo --out C:\research\fresh-report
python -B .\generate_report_images.py --help
```

Evaluation measures class disagreements on generated smooth, print-like, and screen-like patterns. It does not measure real PAD accuracy, identity matching, inversion resistance, standards compliance, or executed comparison baselines. Timing from one environment is not a performance guarantee.

Real image/camera reports use the same aligned pretrained SFace backend and treat texture as advisory; `--model` optionally selects IR-50. The figure layout distinguishes eight global vector groups from the 49 spatial/demo groups. Explicit synthetic reports retain their illustrative rejection gate. Choose a fresh private `--out` directory: requested figures deliberately export input-image and feature visualizations. They do not prove privacy or authentication. Existing sample illustrations require explicit `--overwrite` to replace.

## Tests and architecture

```powershell
python -B -m pytest -q
```

The regression suite covers input boundaries, closed authentication decisions, explicit demo selection, missing inputs, in-memory processing, report gates, dataset leakage/calibration boundaries, local encrypted-store transactions, model tensor behavior, and launcher exit codes. Tensor tests run in the new `.venv`; optional tests can skip in an unprovisioned interpreter. Passing regressions do not establish biometric accuracy. Real browser checks cover desktop/narrow layouts, light/dark themes, keyboard tabs, invalid inputs and session isolation; [OPERATIONAL_VALIDATION.md](docs/OPERATIONAL_VALIDATION.md) records the evidence and limits. The CI workflow is prepared locally; no changes were published to run it remotely. In restricted environments where pytest cannot create its default temporary directory, add `--basetemp` pointing to a dedicated writable directory.

| Area | Responsibility |
| --- | --- |
| `app_demo.py` | Streamlit inspector, paired diagnostics, synthetic benchmark |
| `verify_with_antispoof.py` | Real aligned comparisons, experimental labels, advisory PAD, ephemeral transforms, JSON CLI |
| `modules/face_backend.py` | Offline pretrained YuNet/SFace detection, alignment, embeddings and integrity |
| `modules/presentation_backend.py` | Optional learned MiniFASNet estimates with verified checkpoints |
| `modules/image_input.py` | Decoded-image limits, range/channel validation |
| `modules/antispoof/` | Face presence and bounded FFT/texture heuristics |
| `modules/model.py` | Optional SlerpFace backbone and attention implementation |
| `extract_template/` | Validated image-pair loading and optional trained research feature exports |
| `evaluate_antispoof.py`, `create_demo_samples.py` | Synthetic evaluation and explicit sample generation |
| `generate_report_images.py` | Gated research visualizations |
| `tests/` | Regression, integration, and optional tensor checks |

The new local research registry is separate from the dashboard and is not a biometric account service. There is no trainer or validated authentication policy in this checkout. `train.yaml` is a historical configuration referencing external training infrastructure, not a runnable training entry point.

## Remaining work and licensing

The local prototype can now perform real pretrained inference and comparison. Independent identity recaptures, physical print/display trials across sensors, protected-match calibration, product credential policy, unlinkability/inversion testing and deployment/accessibility validation remain future evidence requirements. Public fixture and transformed-photo smoke checks prove functional execution only. Original IR-50 random-weight round trips prove architecture compatibility, not trained accuracy.

The included reference paper describes the research inspiration; its experimental results must not be attributed to this implementation. This checkout contains no project LICENSE file. The owner must confirm the intended project license and external-code/weight rights before a license declaration or redistribution decision is made.
