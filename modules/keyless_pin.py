"""
modules/keyless_pin.py
======================
Cryptographic Keyless PIN Binding for SlerpFace Template Protection.

Physical & Cryptographic Principle:
- Eliminates persistent transformation key storage on disk/database.
- Derives the SlerpFace hypersphere rotation key and feature dropout mask
  directly and deterministically from a user's memorized PIN using PBKDF2-HMAC-SHA256.
- Ephemeral in-memory lifecycle: derived on-demand, used during transform,
  and discarded.
- Cancelable biometrics: changing the PIN immediately revokes the old template
  and yields an uncorrelated pseudo-impostor template for the same face.
"""

import hashlib
from typing import Optional, Tuple
import numpy as np

from verify_with_antispoof import (
    GROUP_SIZE,
    DEFAULT_DROP_RATE,
    DEFAULT_SLERP_ALPHA,
    ProtectionContext,
    validate_protection_parameters,
)

DEFAULT_KDF_SALT = b"GuardFace.KeylessPIN.SlerpKey.v1\x00"
PBKDF2_ITERATIONS = 100_000


def derive_pin_seed(pin: str, salt: Optional[bytes] = None) -> bytes:
    """
    Derives a 64-byte cryptographic master pseudorandom seed from a user PIN
    using PBKDF2-HMAC-SHA256 with 100,000 iterations.
    """
    if not isinstance(pin, str) or not pin.strip():
        raise ValueError("PIN must be a non-empty string.")
    
    kdf_salt = salt if salt is not None else DEFAULT_KDF_SALT
    if not isinstance(kdf_salt, bytes) or len(kdf_salt) < 16:
        raise ValueError("Salt must be a bytes object of at least 16 bytes.")
        
    return hashlib.pbkdf2_hmac(
        hash_name="sha256",
        password=pin.strip().encode("utf-8"),
        salt=kdf_salt,
        iterations=PBKDF2_ITERATIONS,
        dklen=64,
    )


def create_pin_protection_context(
    pin: str,
    shape: Tuple[int, int],
    drop_rate: float = DEFAULT_DROP_RATE,
    salt: Optional[bytes] = None,
) -> ProtectionContext:
    """
    Deterministically creates an ephemeral ProtectionContext (unit rotation key
    and dropout mask) directly from a user PIN.
    
    Never persists keys to disk or external stores.
    """
    validate_protection_parameters(DEFAULT_SLERP_ALPHA, drop_rate)
    if len(shape) != 2 or shape[0] < 1 or shape[1] != GROUP_SIZE:
        raise ValueError(f"Features must contain nonempty groups of {GROUP_SIZE} dimensions.")

    seed_bytes = derive_pin_seed(pin, salt=salt)
    
    # Initialize a cryptographically seeded NumPy generator
    seed_int = int.from_bytes(seed_bytes, byteorder="big")
    rng = np.random.default_rng(np.random.SeedSequence(seed_int))

    # Deterministic Gaussian random vectors normalized to unit vectors on the S^15 hypersphere
    key = rng.standard_normal(shape)
    norms = np.linalg.norm(key, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Degenerate key group generated; retry with a different salt.")
    key = key / norms

    # Deterministic dropout mask per group
    keep = np.ones(shape, dtype=bool)
    num_drop = int(GROUP_SIZE * drop_rate)
    for group in range(shape[0]):
        drop_indices = rng.choice(GROUP_SIZE, num_drop, replace=False)
        keep[group, drop_indices] = False

    key.setflags(write=False)
    keep.setflags(write=False)
    return ProtectionContext(key, keep)


def evaluate_pin_unlinkability(
    pin1: str,
    pin2: str,
    shape: Tuple[int, int] = (32, GROUP_SIZE),
    drop_rate: float = DEFAULT_DROP_RATE,
) -> dict:
    """
    Measures the key orthogonality and template decorrelation between two different PINs.
    Validates revocability and cross-service unlinkability.
    """
    ctx1 = create_pin_protection_context(pin1, shape, drop_rate)
    ctx2 = create_pin_protection_context(pin2, shape, drop_rate)

    # Key cosine correlation across groups
    group_cosines = [
        float(np.dot(ctx1.key[g], ctx2.key[g]))
        for g in range(shape[0])
    ]
    mean_key_cosine = float(np.mean(group_cosines))
    
    # Mask overlap
    mask_overlap = float(np.mean(ctx1.keep_mask == ctx2.keep_mask))

    return {
        "mean_key_cosine": mean_key_cosine,
        "max_key_cosine": float(np.max(np.abs(group_cosines))),
        "mask_overlap": mask_overlap,
        "is_unlinkable": abs(mean_key_cosine) < 0.15,
    }
