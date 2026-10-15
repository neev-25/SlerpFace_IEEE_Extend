# Pinned Tencent TFace backbone subset

Source: [Tencent/TFace](https://github.com/Tencent/TFace/tree/171ec1434aca0972e111663ac646c9c1e3e44b5b/recognition/torchkit/backbone), commit `171ec1434aca0972e111663ac646c9c1e3e44b5b`.

Only `common.py` and `model_irse.py` are included. Their tensor implementation is the upstream implementation, not a substitute model or pretrained weights. The local package initializers are minimal and locally authored. The one code modification changes `model_irse.py`'s `torchkit.backbone.common` import to a package-relative `.common` import and marks that modification in the file. This avoids changing global module-search paths or loading an unrelated installed package named `torchkit`.

The downloaded bytes were checked against Git blob IDs before writing. Original upstream SHA-256 values:

| File | Git blob | SHA-256 |
| --- | --- | --- |
| common.py | cfc943b3487838d86e8965dfee161fe51346a59a | 53e2025f253406e1d98b0ad86a764ca9f9753a9afbe31ba49ff418f6778e7dbb |
| model_irse.py (before the import change) | 1ceb7c28f222a251a2ba0b2d22173f0f99ad4e6c | 40582defaff7812e102942f46e3fb2d8eaf30e3beed9775875aebaba2347fb2f |
| License.txt | e94b5ac45782eec7a7ee6803146acc7296835d24 | e534cd4a5cd86d462e842be79a306a52a45d01062a2d295367a21ccb4175845b |
| LICENSE.face-evoLVe.txt | f7d642d16bcaf30c7938f14bb1289a1eb6989092 | b5ee08d1d7d7165e5d5b7085b4b328703a52f94c7a9c21c5386bad17bd1f2d29 |

`License.txt` retains the complete upstream TFace license and third-party exceptions. `LICENSE.face-evoLVe.txt` retains the original MIT attribution, copyright (c) 2019 Jian Zhao, obtained from [face.evoLVe/LICENSE](https://github.com/ZhaoJ9014/face.evoLVe/blob/master/LICENSE). No upstream NOTICE file was present in the pinned TFace tree. These files establish notices for this code subset; they do not grant rights to project code, datasets, or pretrained model weights.

Tests use randomly initialized tensors solely to check implementation shapes, numerics and checkpoint loading. They do not measure trained biometric accuracy or permit authentication.
