"""
tests/test_keyless_pin.py
=========================
Tests for Cryptographic Keyless PIN Binding for SlerpFace.
Validates PBKDF2 derivation, unit vectors, dropout masks, and revocability.
"""

from pathlib import Path
import numpy as np

from modules.keyless_pin import (
    create_pin_protection_context,
    evaluate_pin_unlinkability,
)
from verify_with_antispoof import run_pipeline, GROUP_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_pin_derivation():
    """Confirms identical PINs produce the exact same transformation context."""
    shape = (32, GROUP_SIZE)
    ctx1 = create_pin_protection_context("4892", shape, drop_rate=0.50)
    ctx2 = create_pin_protection_context("4892", shape, drop_rate=0.50)

    np.testing.assert_array_equal(ctx1.key, ctx2.key)
    np.testing.assert_array_equal(ctx1.keep_mask, ctx2.keep_mask)


def test_unit_vector_normalization_and_dropout():
    """Confirms rotation keys are unit vectors and dropout matches 50%."""
    shape = (16, GROUP_SIZE)
    ctx = create_pin_protection_context("8123", shape, drop_rate=0.50)

    # Unit vector check
    norms = np.linalg.norm(ctx.key, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)

    # Dropout check: exactly 8 zeroed elements per 16-D group
    drop_count = np.sum(~ctx.keep_mask, axis=1)
    np.testing.assert_array_equal(drop_count, 8)


def test_pin_unlinkability_and_cancelability():
    """Changing PIN produces uncorrelated templates (cancelable biometrics)."""
    shape = (32, GROUP_SIZE)
    report = evaluate_pin_unlinkability("1234", "5678", shape=shape)

    assert report["is_unlinkable"] is True
    assert abs(report["mean_key_cosine"]) < 0.15
    assert report["mask_overlap"] < 0.80


def test_keyless_pin_pipeline_integration():
    """Verifies CLI/Pipeline accepts --pin and applies PBKDF2 transformation."""
    sample_genuine = PROJECT_ROOT / "samples" / "genuine_personA_1.jpg"
    decision, report = run_pipeline(sample_genuine, demo=True, pin="7391")

    assert report["processed"] is True
    assert report["key_source"] == "PBKDF2-HMAC-SHA256 Keyless PIN derivation"
    assert report["tier2_status"] == "SlerpFace Encrypted: alpha=0.90, 50% dimensions shredded."
    assert report["zeroed_dimensions"] == 392  # 49 groups * 8 dropped dims
