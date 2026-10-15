"""Create reproducible synthetic illustrations, never real identity/PAD evidence.

The historical image filenames are retained for existing demo selectors. Their
"genuine" and "person" labels describe generated classes, not known identities
or live human presentations. Importing this module performs no disk writes.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "samples"


def generate_demo_samples(seed: int = 42) -> dict[str, np.ndarray]:
    """Return five synthetic RGB arrays using an isolated random generator."""
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:112, :112]
    radial = np.exp(-((x - 56) ** 2 + (y - 56) ** 2) / (2 * 42 ** 2))
    person_a = np.empty((112, 112, 3), dtype=np.float32)
    for channel, value, sigma in ((0, 190, 3), (1, 145, 3), (2, 115, 2.5)):
        person_a[:, :, channel] = value * (0.65 + 0.35 * radial) + rng.normal(0, sigma, (112, 112))
    person_a2 = person_a * 0.96 + rng.normal(0, 2, (112, 112, 3))
    person_b = np.empty_like(person_a)
    for channel, value, sigma in ((0, 160, 4), (1, 110, 3), (2, 85, 2)):
        person_b[:, :, channel] = value * (0.60 + 0.40 * radial) + rng.normal(0, sigma, (112, 112))

    print_pattern = person_a * 0.82 + 25
    print_pattern += (np.sin(x * 1.5) * np.cos(y * 1.5) * 14)[:, :, None]
    luminance = np.mean(print_pattern, axis=2, keepdims=True)
    print_pattern = print_pattern * 0.72 + luminance * 0.28

    screen_pattern = person_a.copy()
    moire = (np.sin(x * 0.8 + y * 0.4) + np.cos(x * 0.4 - y * 0.8)) * 22
    for channel, scale in ((0, 0.8), (1, 1.1), (2, 1.5)):
        screen_pattern[:, :, channel] += moire * scale
    glare = np.exp(-((x - 82) ** 2 + (y - 28) ** 2) / (2 * 12 ** 2)) > 0.6
    screen_pattern[glare] = 254

    return {
        name: np.clip(array, 0, 255).astype(np.uint8)
        for name, array in (
            ("genuine_personA_1.jpg", person_a),
            ("genuine_personA_2.jpg", person_a2),
            ("genuine_personB_1.jpg", person_b),
            ("spoof_print_attack.jpg", print_pattern),
            ("spoof_screen_replay.jpg", screen_pattern),
        )
    }


def write_demo_samples(out_dir=DEFAULT_OUTPUT_DIR, *, seed: int = 42,
                       overwrite: bool = False) -> list[Path]:
    """Write generated illustrations, protecting existing samples by default."""
    samples = generate_demo_samples(seed)
    output = Path(out_dir).expanduser().resolve()
    paths = [output / name for name in samples]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError("Output contains existing samples; use a fresh --out directory "
                              "or explicitly pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        Image.fromarray(samples[path.name]).save(path)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_DIR), help="Destination directory")
    parser.add_argument("--seed", type=int, default=42, help="Reproducible local RNG seed")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing generated samples")
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        paths = write_demo_samples(args.out, seed=args.seed, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Generated {len(paths)} synthetic demo illustrations in: {paths[0].parent}")
    print("These are not a real biometric dataset or evidence of identity/liveness accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
