"""Manage encrypted local research templates; identity authentication is disabled."""

import argparse
import json
import sqlite3
import sys

import numpy as np

from modules.antispoof.detector import AntiSpoofDetector
from modules.image_input import load_rgb_image
from modules.template_store import DPAPIProtector, TemplateStore, TemplateStoreError
from verify_with_antispoof import create_protection_context, extract_face_features, slerp_encrypt


def _demo_material(image):
    # Explicit synthetic research input. Gate before extraction, and decode once
    # so checking/extraction refer to the same bytes rather than a changing path.
    decoded = load_rgb_image(image)
    result = AntiSpoofDetector(require_face=False).evaluate(decoded)
    if not result.is_live:
        raise ValueError("Synthetic input failed its static-image heuristic; no template was extracted.")
    features, _ = extract_face_features(decoded, demo=True)
    context = create_protection_context(features.shape)
    transformed = slerp_encrypt(features, context=context)
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError("Synthetic protected groups cannot be normalized.")
    return transformed / norms, context


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="Explicit SQLite path in an existing private directory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Exclusively create a user-scope DPAPI research store")
    enroll = sub.add_parser("enroll-demo", help="Store an explicitly synthetic, untrained research transform")
    enroll.add_argument("--image", required=True, help="Synthetic illustration; no real biometric subject")
    enroll.add_argument("--subject", required=True, help="Research label encrypted inside the payload")
    sub.add_parser("list", help="List opaque credential IDs and status only")
    revoke = sub.add_parser("revoke", help="Remove an active credential's ciphertext")
    revoke.add_argument("credential_id")
    rotate = sub.add_parser("rotate", help="Atomically revoke and replace using a new explicit synthetic input")
    rotate.add_argument("credential_id")
    rotate.add_argument("--image", required=True, help="New synthetic illustration; no real biometric subject")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        # The CLI has no test-protector flag and no environment-selected fallback.
        protector = DPAPIProtector()
        store = TemplateStore(args.store, _test_protector=protector)
        if args.command == "init":
            result = store.initialize()
        elif args.command == "list":
            result = {"credentials": store.list_credentials()}
        elif args.command == "revoke":
            result = store.revoke(args.credential_id)
        else:
            material, context = _demo_material(args.image)
            if args.command == "enroll-demo":
                result = store.enroll(args.subject, material, context)
            else:
                result = store.rotate(args.credential_id, material, context)
        print(json.dumps({**result, "authenticated": False, "assurance": "RESEARCH_ONLY_UNTRAINED"}, allow_nan=False))
        return 0
    except (TemplateStoreError, sqlite3.Error, OSError, ValueError) as exc:
        # Errors never serialize templates, context, images, or subject labels.
        print(json.dumps({"status": "UNAVAILABLE", "authenticated": False,
                          "error": str(exc)}, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
