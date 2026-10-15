# Optional pretrained presentation model

The webcam prototype now has an optional **learned single-frame presentation
estimate**. It is independent of the face comparison and never prevents a
comparison. Its three numbers are model softmax outputs, not calibrated
probabilities that a person is physically present. No authentication or liveness
decision is enabled.

## Model source and terms

The two pretrained models come directly from MiniVision's
[Silent-Face-Anti-Spoofing repository](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/tree/b6d5f04ad78778917853b25c778acef6d5626d15),
pinned to commit `b6d5f04ad78778917853b25c778acef6d5626d15` (checked 2026-10-16).
The [upstream README](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/README_EN.md)
describes releasing the model and identifies `resources/anti_spoof_models/` as
the fusion models. The repository's
[Apache-2.0 license](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/LICENSE)
includes Copyright 2020 Minivision. No separate weight restriction was found at
that commit. This is the upstream repository licensing evidence, not an
independent audit of upstream training-data rights. Retain the included license
and attribution when redistributing these materials.

Only the [unmodified architecture](../third_party/silent_face/MiniFASNet.py) and
[upstream license](../third_party/silent_face/LICENSE) are vendored. The old
upstream Python/Torch requirements, face detector, and executable scripts are
not installed. The local wrapper uses the project's existing CPU PyTorch 2.13
environment with restricted `weights_only=True` deserialization, strict tensor
state loading, and no online operation during inference.

Before the first learned estimate, the adapter caps PyTorch intra-operation CPU
threads at two under a lock (an existing lower limit is retained). This is an
intentional **process-wide** runtime setting to avoid oversubscribing tiny
depthwise convolutions on machines with many cores. It is not changed back
between requests, because transient thread-pool changes would race with other
requests. Inter-operation threading and the shared/system Python environments
are not changed. The cap does not modify weights or preprocessing.

| Official checkpoint | Bytes | SHA-256 |
| --- | ---: | --- |
| `2.7_80x80_MiniFASNetV2.pth` | 1,849,453 | `a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0` |
| `4_0_0_80x80_MiniFASNetV1SE.pth` | 1,856,130 | `84ee1d37d96894d5e82de5a57df044ef80a58be2b218b5ed7cdfd875ec2f5990` |

The hashes above were calculated from the official, commit-pinned HTTPS files;
they are integrity pins, not signatures from the model publisher. Every
inference call checks the length and SHA-256 of both local files before using
cached models. Both are required; there is no one-model or heuristic fallback.
The architecture file is checked against its pinned SHA-256 before first load.

## Provisioning

Already provisioned for this workspace under ignored `ckpt/presentation/`.
To reproduce from the project root:

```powershell
.\.venv\Scripts\python.exe -B .\scripts\provision_pad_model.py
```

The provisioner downloads only the two commit-pinned official URLs, caps each
read at its expected size plus one byte, verifies both length and hash, and
atomically publishes verified files. It leaves existing mismatched files alone
and reports an error. Verified existing files require no network. For another
directory, pass `--model-dir C:\path\to\presentation`; callers must also pass
that directory to the backend. The application never downloads models itself.

## Preprocessing and output semantics

`evaluate_presentation(image, face_box, model_dir=None)` accepts an RGB image
through the bounded shared image loader and an original-image `(x1,y1,x2,y2)`
box with exclusive upper bounds from the separate face detector. It rejects
invalid boxes and boxes smaller than 32 pixels per side as `INCONCLUSIVE`; the
32-pixel minimum is a local engineering guard, not a validated model threshold.

The adapter follows the pinned upstream
[context crop](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/generate_patches.py):
2.7x and 4x contextual rectangles are moved inside the image, cropped with
inclusive pixel endpoints, and resized by OpenCV bilinear interpolation to
80x80. RGB is converted to BGR, then CHW float32 in **[0,255] with no division by
255**, matching upstream's
[tensor conversion](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/data_io/functional.py)
and [inference path](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/anti_spoof_predict.py).
The average of the two softmax vectors is returned. Upstream's
[test code](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/test.py)
identifies **class 1 as bona fide**. This project preserves the other labels as
`attack_class_0` and `attack_class_2`; it does not invent print/replay meanings.

`presentation_backend_status(model_dir=None)` separately verifies local hashes,
architecture, and required imports without running inference on fabricated
images. Its `available` and `ready` fields indicate installation readiness, not
measured accuracy.

The result's `to_dict()` contains `status`, `bona_fide_score`, `model_name`,
`reason`, `class_scores`, and `metadata`. Status is `MODEL_ESTIMATE` after valid
inference, `UNAVAILABLE` for missing/incompatible/tampered models or backend
errors, and `INCONCLUSIVE` for invalid input/face bounds. No status represents a
pass, a spoof verdict, or permission to authenticate. No threshold is used.

## Practical limits and validation

The upstream README explicitly cautions that results depend on camera, scene,
complete face framing and pose. Its demonstration script constrains frames to
3:4 portrait orientation. This prototype accepts landscape webcam frames through
the same context-crop operation, and uses its separate face detector; these are
domain changes requiring camera-specific evaluation. Static texture scoring
cannot establish trusted live capture, resist all replay/injection attacks, or
prove that a capture is physically present. Published upstream performance
numbers are **not** claimed for this application.

Focused checks cover input bounds, corner crops, color order, byte scaling,
class semantics, integrity failures, offline inference, missing-backend
behavior, bounded downloads and atomic publication. A real official-weight
smoke check loads both checkpoints with restricted deserialization and verifies
repeatable finite outputs. Its uniform square is an execution fixture, not a
biometric or spoof-accuracy test. Real webcam/print/screen accuracy remains
unmeasured and must be evaluated separately using consented camera captures.

A local timing check on 2026-10-16, after Python/Torch imports, measured
0.671 seconds for the first two-model estimate and 0.266/0.281 seconds for two
cached estimates on a 160x160 execution fixture. The thread cap reduced the
process's reported intra-operation thread count from 20 to two. These are local
measurements, not a latency guarantee or accuracy evaluation; full application
startup, face detection and embedding computation are additional work.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q tests\test_presentation_backend.py
```
