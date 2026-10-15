# Offline dataset validation

`evaluate_dataset.py` adds a reproducible path for evaluating supplied images and identity labels. It does not turn the research dashboard into an authentication service. No biometric accuracy result is claimed until actual compatible weights and independent data are supplied.

## Dataset manifest

The manifest is JSON stored beside the image tree. Image paths are relative to that directory, confined to it, and must already exist. File bytes are bounded and hashed; a change after validation aborts evaluation. Image paths, subject identifiers, templates and protection material are omitted from the public report.

```json
{
  "schema_version": 1,
  "data_kind": "real",
  "source": "Dataset source and applicable use/provenance record",
  "pairs": [
    {"left": "cal/a1.jpg", "right": "cal/a2.jpg", "left_subject": "a", "right_subject": "a", "same_person": true, "split": "calibration"},
    {"left": "cal/a1.jpg", "right": "cal/b1.jpg", "left_subject": "a", "right_subject": "b", "same_person": false, "split": "calibration"},
    {"left": "test/c1.jpg", "right": "test/c2.jpg", "left_subject": "c", "right_subject": "c", "same_person": true, "split": "test"},
    {"left": "test/c1.jpg", "right": "test/d1.jpg", "left_subject": "c", "right_subject": "d", "same_person": false, "split": "test"}
  ],
  "presentations": [
    {"image": "pad/live1.jpg", "subject": "pad-a", "kind": "bona_fide", "capture_kind": "camera"},
    {"image": "pad/print1.jpg", "subject": "pad-a", "kind": "print", "capture_kind": "physical_attack"},
    {"image": "pad/screen1.jpg", "subject": "pad-a", "kind": "screen", "capture_kind": "physical_attack"}
  ]
}
```

This is a schema illustration, not a dataset supplied with the repository. A manifest may supply identity pairs, presentations, or both. Both identity splits must contain genuine and impostor pairs. Subjects and exact image bytes cannot overlap calibration/test splits. Self-pairs (including byte-identical renamed copies), duplicate/reversed pairs, conflicting subject assignments for a capture, inconsistent identity labels, path escapes and missing files are rejected. Renaming an image does not defeat these checks.

Presentation kinds are `bona_fide`, `print`, `screen`, `mask`, and `other_attack`. Capture kinds are `camera`, `physical_attack`, `digital_replay`, and `synthetic`. Declarations are supplied ground truth; this software cannot independently establish how an image was captured. Synthetic records cannot be labeled as a real dataset. A source declaration records provenance supplied by the operator, not a license grant or certification.

## Run

```powershell
python -B evaluate_dataset.py --manifest C:\research\dataset\manifest.json --backend sface --out C:\research\fresh-result.json
python -B evaluate_dataset.py --manifest C:\research\dataset\manifest.json --backend ir50 --model C:\research\compatible-ir50.pth --out C:\research\fresh-ir50-result.json
```

The CLI defaults to provisioned pretrained SFace. `--backend ir50` requires an explicit compatible checkpoint. Both real paths detect and align inputs consistently with the dashboard before extraction. The Python API retains `backend="ir50"` by default for backward compatibility; pass `backend="sface"` for the provisioned default model. SFace model artifact hashes/readiness are checked before and after evaluation. `--demo` requires a synthetic declaration and cannot be reported as biometric evidence. Existing reports are never overwritten. Changed checkpoint/model bytes abort report creation.

Optional parameters are `--alpha`, `--drop`, `--seed`, `--target-fmr`, and `--threshold`. The first four concern numerical identity experiments; `--threshold` applies to static-image presentation heuristics. The seed is deliberately reproducible research material, never a credential key or a privacy guarantee. Input limits are 100,000 records, a 20 MiB manifest, and the shared per-image byte/pixel bounds.

## Meaning of results

Identity evaluation compares the original vector cosine with a mean of per-group protected cosines. One context is shared for each reference/query comparison. The common-noise distortion still exists; using group scores is an explicit research comparison rather than a claim that the original matching vulnerability has been solved mathematically.

The threshold is selected using calibration observations only: choose the lowest threshold meeting the requested *empirical* false-match target. It is then frozen while scoring independent test subjects. Equal genuine/impostor scores may yield a reject-all threshold and a 100% false-nonmatch rate; this is surfaced rather than relabeled as success. Nothing copies the threshold into the live pipeline or enables authentication.

Reports include false-match and false-nonmatch counts/rates, trial counts and nominal 95% exact binomial intervals, using [SciPy's exact proportion interval](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html). These intervals assume independent Bernoulli trials. Reusing identities/captures can violate that assumption; subject/cluster-aware confidence analysis is needed for a real scientific claim. Zero observed false matches in a small sample does not imply zero population risk. A target below what the data can support must not be represented as established merely because its empirical count is zero.

Presentation evaluation reports each class separately, including face-detection failures. This evaluator currently measures the **legacy Haar/static-image heuristic**, explicitly recorded as `backend: static_image_heuristic`; it does not evaluate MiniFASNet's advisory estimates. A missing detector is an operational error. Rejection rates include the configured presence/heuristic gate. Real learned-model PAD validation needs independently acquired physical trials and a separate defined evaluation; transformed drawings cannot establish it.

The report binds its manifest digest and optional checkpoint digest, records algorithm parameters, labels its data kind, and always sets `authentication_enabled=false`. Checkpoint hashes describe retrieved bytes, not publisher authenticity or permission to use weights. Dataset access, capture/identity labels and applicable rights must be verified outside this evaluator.

## Regression coverage

`tests/test_dataset_validation.py` covers threshold separation/ties, interval uncertainty, malformed labels, distinct captures, subject and renamed-image leakage, confined paths, mutation after validation, explicit synthetic/real backend boundaries, group scoring, output overwrite protection and separate face-detection failures. Passing these checks validates the evaluation code's boundaries, not a face recognition model's accuracy.
