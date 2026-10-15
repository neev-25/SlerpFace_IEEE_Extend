"""Explicitly provision pinned OpenCV Zoo weights; inference never downloads."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.face_backend import DEFAULT_MODEL_DIR, MODEL_COMMIT, MODEL_SPECS  # noqa: E402


class _TrustedRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlparse(newurl)
        if parsed.scheme != "https" or parsed.hostname not in {"media.githubusercontent.com", "raw.githubusercontent.com"}:
            raise ValueError("Refusing a model download redirect outside official GitHub content hosts.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download_verified(url, target, expected_sha256, expected_size, *, replace=False):
    """Bounded TLS download followed by atomic replace after exact verification."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"media.githubusercontent.com", "raw.githubusercontent.com"}:
        raise ValueError("Only official HTTPS GitHub model source URLs are permitted.")
    target = Path(target)
    if target.exists():
        if target.stat().st_size == expected_size:
            with target.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest == expected_sha256:
                return {"filename": target.name, "sha256": digest, "state": "already_verified"}
        if not replace:
            raise ValueError(f"{target.name} exists but is unverified. Inspect it, then use --replace if appropriate.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    deadline = time.monotonic() + 180
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "GuardFace-prototype-model-setup/1.0"})
        opener = urllib.request.build_opener(_TrustedRedirects())
        with opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise ValueError(f"Unexpected model download HTTP status: {response.status}")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) != expected_size:
                raise ValueError("Model response size differs from the pinned Git LFS size.")
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".download-", suffix=".part", delete=False) as output:
                temporary = Path(output.name)
                digest = hashlib.sha256()
                count = 0
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Model download exceeded its total time limit.")
                    chunk = response.read(min(1024 * 1024, expected_size - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > expected_size:
                        raise ValueError("Model response exceeds its pinned size.")
                    digest.update(chunk)
                    output.write(chunk)
                if count != expected_size or digest.hexdigest() != expected_sha256:
                    raise ValueError("Model integrity check failed. No model was installed.")
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, target)
        temporary = None
        return {"filename": target.name, "sha256": expected_sha256, "state": "installed"}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def provision(model_dir=None, *, replace=False):
    directory = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    directory = directory.resolve()

    def install(spec):
        url = (f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_COMMIT}/"
               f"models/{spec['directory']}/{spec['filename']}")
        return download_verified(url, directory / spec["filename"], spec["sha256"], spec["size"], replace=replace)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(install, MODEL_SPECS.values()))
    return {"model_dir": str(directory), "source_commit": MODEL_COMMIT, "models": outcomes,
            "purpose": "Local research prototype; no authentication or liveness assurance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--replace", action="store_true", help="Replace an existing model only after a trusted download verifies.")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(provision(args.model_dir, replace=args.replace), indent=2))
    except (OSError, ValueError, TimeoutError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
