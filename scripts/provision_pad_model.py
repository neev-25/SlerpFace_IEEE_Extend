"""Explicit, bounded provisioning of pinned official MiniVision checkpoints."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

# Make direct execution from any working directory use this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.presentation_backend import DEFAULT_MODEL_DIR, MODEL_SPECS, SOURCE_COMMIT, verify_model_bytes  # noqa: E402


def provision(model_dir=DEFAULT_MODEL_DIR):
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    completed = []
    for spec in MODEL_SPECS:
        destination = directory / spec.filename
        if destination.exists():
            with destination.open("rb") as source:
                verify_model_bytes(source.read(spec.size + 1), spec)
        else:
            request = urllib.request.Request(spec.url, headers={"User-Agent": "GuardFace-prototype-model-provisioner/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.url != spec.url:
                    raise ValueError("Unexpected presentation-model download redirect.")
                payload = response.read(spec.size + 1)
            verify_model_bytes(payload, spec)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=".pad-", suffix=".download", delete=False) as output:
                    temporary = Path(output.name)
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                # Atomic, exclusive publication on the same filesystem. A
                # partially written checkpoint is never visible at destination.
                try:
                    try:
                        os.link(temporary, destination)
                    except PermissionError:
                        if os.name != "nt":
                            raise
                        # Windows rename is atomic and refuses an existing
                        # destination; works where hard links are restricted.
                        os.rename(temporary, destination)
                except FileExistsError:
                    with destination.open("rb") as source:
                        verify_model_bytes(source.read(spec.size + 1), spec)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        completed.append({"filename": spec.filename, "sha256": spec.sha256, "source": spec.url})
    return {"source_commit": SOURCE_COMMIT, "models": completed, "authentication_enabled": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(provision(args.model_dir), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Presentation model provisioning failed: {exc}\n")


if __name__ == "__main__":
    main()
