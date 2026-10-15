# Trained-model provenance and compatibility

This document concerns the original optional Tencent SlerpFace checkpoint. The functional camera/upload prototype now uses separately provisioned OpenCV SFace weights by default; see [PRETRAINED_FACE_BACKEND.md](PRETRAINED_FACE_BACKEND.md) and [FUNCTIONAL_PROTOTYPE.md](FUNCTIONAL_PROTOTYPE.md). Missing optional IR-50 weights no longer prevent the default pretrained comparison.

Research checked on 2026-10-16. This records verified sources and the remaining evidence needed for trained extraction. It does not authorize biometric authentication, establish model accuracy, or assign a license to this project.

## Repository identity

The selected workspace is `C:\Documents\AI_Project`. Its configured Git remote is [neev-25/SlerpFace_IEEE_Extend](https://github.com/neev-25/SlerpFace_IEEE_Extend). This is distinct from the paper authors' implementation.

The [AAAI publication](https://ojs.aaai.org/index.php/AAAI/article/view/33162) identifies *SlerpFace: Face Template Protection via Spherical Linear Interpolation*, DOI `10.1609/aaai.v39i10.33162`, published in 2025. The [Tencent implementation](https://github.com/Tencent/TFace/tree/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface) explicitly identifies itself as the official code and links the paper. Local research results must be reported independently of the publication's results.

The bundled reference PDF has SHA-256 `2a395b0519be86c66e91cd53efe82209f22d44ce05202e9ffcc3d727a5a6f9b1`. This is a local integrity record; byte identity with the publisher's download was not independently established.

## Verified code snapshot

The GitHub API resolved Tencent/TFace `master` to commit `171ec1434aca0972e111663ac646c9c1e3e44b5b` during this review. The commit date is 2025-12-11 and the commit is unsigned. Its content can be pinned for reproducibility; that does not provide a cryptographic author signature. The original SlerpFace addition is commit `6a4f9a85135d4b3afdeb31b540a3c24b45de2263`.

The following files were fetched through the GitHub connector at the pinned commit. These identifiers are **Git blob SHA-1 identifiers**, not raw-file SHA-256 digests.

| Upstream path | Git blob identifier | Purpose |
| --- | --- | --- |
| `recognition/torchkit/backbone/common.py` | `cfc943b3487838d86e8965dfee161fe51346a59a` | Initialization, flattening, and SE components |
| `recognition/torchkit/backbone/model_irse.py` | `1ceb7c28f222a251a2ba0b2d22173f0f99ad4e6c` | IR/IR-SE residual blocks and layer definitions |
| `recognition/tasks/slerpface/modules/model.py` | `d1425f0a79b9f7acd159ae832fc2c82ddabc1b8c` | SlerpFace backbone and attention architecture |
| `recognition/tasks/slerpface/extract_template.sh` | `bcbc06548fed95427876a392c77d023e0d239284` | Official IR-50 extraction command |
| `recognition/test/utils.py` | `c59abbb7becb78c58a18729d226cc9bca2b9d1dd` | Original verification data preprocessing |
| `recognition/tasks/slerpface/requirements.txt` | `6da7862e75910d134cec9a424c0b83ec54fea4e4` | Historical broad dependency lower bounds |
| `recognition/tasks/slerpface/readme.md` | `26276c0018fb87d047a4a4b9a9886ffcb901787a` | Official model and dataset instructions |
| `License.txt` | `e94b5ac45782eec7a7ee6803146acc7296835d24` | Apache-2.0 terms and third-party exceptions |

Source links: [common.py](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/torchkit/backbone/common.py), [model_irse.py](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/torchkit/backbone/model_irse.py), [SlerpFace model](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/modules/model.py), [extraction command](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/extract_template.sh), [preprocessing](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/test/utils.py), [historical requirements](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/requirements.txt), [readme](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/readme.md), [license](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/License.txt).

Use this implementation's backbone components. Installing an unrelated PyPI distribution named `torchkit` does not establish compatibility. Upstream `torchkit.backbone.__init__` imports many other architectures; a minimal, attributed package containing the required modules avoids unnecessary architecture/dependency imports. Any import-path modifications and new minimal initializers must be identified as local changes and tested against the pinned module definitions.

## Intended inference contract

The [official shell command](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/extract_template.sh) explicitly selects IR-50. The upstream Python parser defaults to IR-18, so relying on its default would change the architecture. The supported shared local extractor selects IR-50 explicitly.

The configuration is RGB, NCHW float32, 112 by 112 pixels, IR mode, 512-dimensional vector output, and 16 group channels. IR-50 group features have shape `(N, 16, 7, 7)` before the local layout conversion. The [original input loader](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/test/utils.py) maps byte values with `(x / 255 - 0.5) / 0.5`; endpoints are exactly -1 and 1. Dividing `(x - 127.5)` by 128 is a different preprocessing protocol and should not be attributed to this loader. Face alignment/cropping and sensor conditions also need a documented evaluation protocol: resizing a whole scene to 112 pixels does not establish equivalence with aligned verification data.

Use `eval()` and inference mode. Checkpoint keys and tensor shapes must load strictly, including `group_cov` and `group_attention`; a generic ArcFace/IR-50 checkpoint is not enough when these trained branches are absent. Do not use `strict=False` to conceal incompatible weights or fabricate a checkpoint from random initialization. Validate both forward and horizontally flipped feature paths, finite outputs, and attention normalization. A successful random-weight shape test demonstrates code execution only.

The original dependency lower bounds are historical compatibility declarations, not safe modern environment pins. The [PyTorch-published advisory for CVE-2025-32434](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6) identifies versions before 2.6.0 as affected even with `weights_only=True`; 2.6.0 fixes that specific issue. Select a currently patched build with compatible Python/platform support and audit the resolved environment. Keep weights-only loading and never switch it off merely to load an unverified file. This single advisory is not a complete vulnerability audit. See [RUNTIME_VALIDATION.md](RUNTIME_VALIDATION.md) for the separate installation/test record.

## Official checkpoint availability

The [pinned authors' readme](https://github.com/Tencent/TFace/blob/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/tasks/slerpface/readme.md) links the public [SlerpFace Google Drive folder](https://drive.google.com/drive/folders/12O9Ql-GhlOqSAEsACcOarVDN22gss6ma) and names `Backbone_Epoch_24_checkpoint.pth` and five pair-list files. The trusted link is established; checkpoint bytes, size, contents, publisher checksum, and checkpoint-specific terms were not verified in this research pass. The text browser did not enumerate the folder. No checkpoint was downloaded by this provenance investigation.

When retrieving the file from that exact author-linked location, record the resolved file URL, retrieval date, byte length, and SHA-256 before loading. Retain model-specific terms or an explicit rights statement when available. If no publisher hash exists, a locally computed hash records the retrieved bytes but is not independent evidence of authenticity. Validate the IR-50 state dictionary in an isolated process using a patched Torch build. Reject mismatched keys/shapes and nonfinite outputs. Exercise direct research extraction; authentication must remain disabled until independent PAD and matching validation is complete.

The source repository's license does not, by itself, prove what rights attach to an externally hosted checkpoint or its training data. Weight redistribution, dataset access, and intended use remain unresolved where no applicable terms were verified. No author was contacted and no license grant was invented. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Completed and outstanding

Completed: configured remote identification; official paper/code chain; pinned source/blob inventory; exact expected backbone APIs; source license and ancestor-code attribution; official checkpoint link; input-contract comparison; safe-loading advisory check. Direct shell GitHub metadata retrieval failed on DNS; read-only GitHub connector calls succeeded.

Outstanding: official checkpoint retrieval and hash; applicable weight/data terms; real-checkpoint strict load and forward pass; independent identities and physical presentation trials; matching calibration, revocation/unlinkability, and privacy evaluation. Runtime installation and vendoring progress are recorded separately. These evidence gaps cannot be filled by passing numerical unit tests or by choosing a project license on the owner's behalf.
