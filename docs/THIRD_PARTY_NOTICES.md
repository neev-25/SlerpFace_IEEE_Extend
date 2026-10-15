# Third-party source and rights inventory

Checked 2026-10-16. This inventory identifies verified upstream notices and unresolved rights. It does not apply a new license to GuardFace and does not replace complete license texts included with distributed dependencies or vendored source.

## Functional prototype model additions

The default trained face backend uses OpenCV Zoo YuNet and SFace models at pinned commit `47534e27c9851bb1128ccc0102f1145e27f23f98`. Their directory-specific MIT and Apache-2.0 notices are retained in [third_party/opencv_zoo](../third_party/opencv_zoo/). Exact sizes, Git LFS SHA-256 identifiers, source links and unresolved commercial/training-data questions are documented in [PRETRAINED_FACE_BACKEND.md](PRETRAINED_FACE_BACKEND.md). The local purpose is the user's research prototype; these notices do not invent additional rights.

The advisory presentation backend uses MiniVision's two MiniFASNet checkpoints at commit `b6d5f04ad78778917853b25c778acef6d5626d15`, with the unmodified architecture and Apache-2.0 notice retained in [third_party/silent_face](../third_party/silent_face/README.md). See [PRESENTATION_MODEL.md](PRESENTATION_MODEL.md) for source, hashes, released-weight evidence and training-data/operational limits. Model weights are provisioned explicitly into ignored directories, not embedded into the first-party source.

The two public inference fixtures have original NASA/U.S. Navy credits, source URLs, public-domain evidence and hashes in [tests/fixtures/README.md](../tests/fixtures/README.md). They are test material, not evidence of independent capture accuracy, liveness, consent for deployment or endorsement by the subjects/agencies.

## Tencent source details

The local SlerpFace architecture is derived from the authors' [Tencent TFace implementation](https://github.com/Tencent/TFace/tree/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface). The required backbone components originate from `recognition/torchkit/backbone/common.py` and `model_irse.py` at the same pinned commit. Exact Git blob identifiers are recorded in [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

The [upstream License.txt](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/License.txt) identifies Tencent copyright 2025, Apache License 2.0, and exceptions for listed third-party components. Keep that complete file with vendored TFace code. The applicable Apache terms include retaining attribution and license text, identifying modified files, and preserving relevant NOTICE material when supplied. The pinned repository's complete recursive tree contained no separately named NOTICE file; its combined License.txt contains extensive dependency notices. This records the observed source declaration without claiming ownership of local changes or externally hosted model weights.

Local changes to the model include input/configuration validation, stable constant-map attention, initialization corrections, deep-backbone channel handling, and import/error handling. Minimal vendored package initializers and any adjusted imports are local packaging changes and must carry their own modification notices. They should not be represented as a verbatim complete TFace distribution.

## face.evoLVe ancestor code

Both the local architecture header and Tencent's backbone reference [ZhaoJ9014/face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe), historically named `face.evoLVe.PyTorch`. Its [LICENSE](https://github.com/ZhaoJ9014/face.evoLVe/blob/master/LICENSE) identifies the MIT license and copyright 2019 Jian Zhao. The fetched license has Git blob identifier `f7d642d16bcaf30c7938f14bb1289a1eb6989092`.

Retain the original copyright, complete MIT permission notice, and disclaimer with source derived from it. The TFace repository's Apache declaration does not erase this ancestor attribution. The exact original ancestor revision used by Tencent was not identified; the header's mutable URL alone cannot establish a historical commit.

## Paper, checkpoint, and datasets

Reference publication: Zhong et al., *SlerpFace: Face Template Protection via Spherical Linear Interpolation*, AAAI 2025, DOI [10.1609/aaai.v39i10.33162](https://doi.org/10.1609/aaai.v39i10.33162). Citation acknowledges the research; it does not supply code, checkpoint, or biometric-data rights. The bundled PDF's local hash is in MODEL_PROVENANCE.md. No independent redistribution assessment of that PDF was completed.

The authors' [readme](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/readme.md) links an external Drive folder containing the intended checkpoint and pair lists. Checkpoint-specific terms, publisher hash, and training-data rights were not verified. No blanket code-license grant is asserted for these external files. Verification datasets have separate access/use terms; obtain and retain their applicable terms rather than assuming that a code link grants dataset rights.

## Installed libraries and project ownership

Package wheels contain their own license files and can bundle native components with additional notices. A transitive dependency security scan does not establish license compliance. Preserve the notices from the exact versions distributed; this source inventory is not a complete software bill of materials or legal clearance of a packaged application.

No root project LICENSE was present at the start of this work. The project owner must select a license they have authority to grant for original contributions and resolve intended distribution/use where upstream, model, or data rights remain uncertain. No project license was selected on the owner's behalf.
