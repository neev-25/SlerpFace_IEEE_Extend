# Functional local face comparison backend

The webcam/upload prototype now has a pretrained CPU inference route:
YuNet detects one face and five landmarks; OpenCV SFace aligns it to 112 x 112,
then extracts a normalized 128-dimensional face embedding. Processing remains
local. Inference and status checks never download models or send photographs.

This backend supports experimental similarity comparisons. Face recognition
does not establish physical liveness or confer authentication assurance.

## Installation and integrity

Use the existing isolated runtime:

```powershell
.\.venv\Scripts\python.exe -B scripts/provision_face_models.py
```

This explicit command downloads from official OpenCV Zoo GitHub content using
HTTPS, exact sizes and Git LFS SHA-256 identifiers. Each completed verified file
replaces its temporary download atomically. Wrong hashes, oversized/truncated
responses, untrusted redirects and timeouts fail without installing those bytes.
Already valid files are reused. Existing unverified files require an explicit
`--replace`. Both artifacts are ignored under `models/opencv/` rather than
committed as generated binary files.

Source commit: [`47534e27c9851bb1128ccc0102f1145e27f23f98`](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98).

| Artifact | Bytes | SHA-256 | Upstream directory license |
| --- | ---: | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | MIT |
| `face_recognition_sface_2021dec.onnx` | 38696353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` | Apache-2.0 |

The hashes and sizes are the official versioned Git LFS pointers:
[YuNet](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx),
[SFace](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx).
Unmodified license texts are saved under `third_party/opencv_zoo/`.

The chosen YuNet 2023 model works with the project's pinned OpenCV 4.12 runtime.
The newer 2026 dynamic model addresses the OpenCV 5 ONNX Runtime engine and is
not needed for this runtime. See the [upstream YuNet notes](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/README.md).

## API and operational bounds

`modules.face_backend.extract_pretrained_face(image, model_dir=None)` accepts
the existing bounded RGB image inputs and returns `FaceFeatures`:

- `vector`: a finite nonzero unit-length 128-value embedding;
- `groups`: that vector reshaped to eight groups of 16 values;
- `face_box`: original-image `(x1, y1, x2, y2)` pixel coordinates;
- `landmarks`: five original-image x/y landmark coordinates;
- `detection_score`, `model_name`, public scalar `quality` metadata;
- `aligned_rgb`: the 112 x 112 RGB aligned face for an optional alternate model.

**The vector grouping is an experimental adaptation.** It is not the learned
spatial groups from the official Tencent SlerpFace architecture. Similarity and
template comparisons across different embedding backends are not interchangeable.

`FaceInputError.code` distinguishes `INVALID_IMAGE`, `NO_FACE`, `MULTIPLE_FACES`
and `LOW_QUALITY`. No face, multiple faces, faces smaller than 40 pixels or eye
separation below 10 pixels fail with capture guidance. No arbitrary texture or
blur threshold is used as proof of spoofing. Detection confidence is 0.75, an
operational prototype setting rather than a calibrated biometric threshold.

Detector work is bounded to a 640-pixel maximum side; landmarks map back to the
original image before `alignCrop`, so recognition uses original pixels. Shared
image decoding bounds remain 20 MiB and 16 million pixels. One pair of mutable
OpenCV networks is kept per thread. SHA-256 verification is memoized by resolved
path, size, timestamps and inode; changed files invalidate readiness and the
thread-local model cache. Model files must remain locally trusted; this is not
an operating-system file-integrity boundary against a malicious local account.

`face_backend_status(model_dir=None)` returns `ready`, `model_name`, `backend`,
`device`, `model_dir`, model presence/hash verification, OpenCV version, and an
actionable public `reason`. `FaceBackendUnavailable` reports runtime or model
provisioning failures without substituting synthetic embeddings.

## License and model limitations

The official [SFace directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface)
states Apache-2.0 for files in that directory; the official
[YuNet directory](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet)
states MIT. We preserve those notices and use the artifacts for the user's
explicitly stated local research prototype. These notices do not resolve
every possible training-image, personality-right or commercial-deployment
question. Upstream [issue 313](https://github.com/opencv/opencv_zoo/issues/313)
asks about SFace training data and commercial terms. Review the answer and
applicable data rights before any commercial or public biometric deployment.
No project-wide license is invented by this integration.

OpenCV describes the required
[detection, five-point alignment and recognition sequence](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
The implementation follows this sequence but reports experimental results;
upstream published benchmark accuracy does not establish this application's
camera accuracy, population performance, or liveness resistance.

## Validation scope

`tests/test_face_backend.py` covers bounded image handling, scaling/coordinate
mapping, RGB/BGR alignment, normalized vector contracts, invalid network
outputs, missing/corrupt models, hash cache invalidation, and downloader
integrity/failure cleanup. Its actual pretrained integration tests run offline
using the public NASA and U.S. Navy fixtures documented in
`tests/fixtures/README.md`: one-face extraction/repetition, blank/multiple-face
rejection, a same-source mild transformation and a separate-portrait comparison.
The transformed photograph is not an independent recapture. These are functional
smoke tests, not recognition or presentation-attack accuracy estimates.
