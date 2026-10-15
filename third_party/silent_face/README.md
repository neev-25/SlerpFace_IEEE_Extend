# MiniVision MiniFASNet architecture

`MiniFASNet.py` is an **unmodified, byte-identical** copy of
[`src/model_lib/MiniFASNet.py`](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/model_lib/MiniFASNet.py)
from MiniVision Silent-Face-Anti-Spoofing commit
`b6d5f04ad78778917853b25c778acef6d5626d15`.

Upstream author: zhuying; company: Minivision. Copyright 2020 Minivision.
The upstream Apache-2.0 [LICENSE](LICENSE) is included verbatim. No upstream
training code, detector, environment, or dependency versions are installed.

| File | SHA-256 |
| --- | --- |
| MiniFASNet.py | `e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7` |
| LICENSE | `daf94bf1dc9cc5700fe5af2c7c0cbd1836e70d509ed78fd0bebef7432edee6fb` |

Pretrained checkpoints are provisioned separately into ignored
`ckpt/presentation/`; see [the presentation-model documentation](../../docs/PRESENTATION_MODEL.md).
The project adapter changes loading to verified bytes and PyTorch
`weights_only=True`, caches models, returns uncategorized probability estimates,
and adapts the upstream context crop. It does not execute upstream scripts.
