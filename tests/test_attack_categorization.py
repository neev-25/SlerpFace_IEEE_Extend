"""
tests/test_attack_categorization.py
===================================
Tests for Granular Presentation Attack Detection (PAD) & Zero-Leakage Protocol.
Validates exact compliance with user specifications and Image 2 badges.
"""

from pathlib import Path
import pytest

from modules.antispoof.detector import AntiSpoofDetector
from verify_with_antispoof import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def detector():
    return AntiSpoofDetector(threshold=0.60, require_face=False)


def test_genuine_classification_and_badge(detector):
    """Scenario 3: Live human face / genuine sample passes PAD."""
    sample_path = PROJECT_ROOT / "samples" / "genuine_personA_1.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is True
    assert res.attack_type == "GENUINE"
    assert res.badge == "GENUINE (Live Human Face)"
    assert res.liveness_score >= 0.60
    assert "Passed liveness check" in res.summary_text
    assert "Natural 3D skin texture and regular frequency decay detected" in res.summary_text
    assert res.tier2_status == "SlerpFace Encrypted: alpha=0.90, 50% dimensions shredded."


def test_print_attack_classification_and_badge(detector):
    """Scenario 1: Printed paper attack is intercepted with exact badges & metrics."""
    sample_path = PROJECT_ROOT / "samples" / "spoof_print_attack.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is False
    assert res.attack_type == "PRINT_ATTACK"
    assert res.badge == "PRINT_ATTACK (Printed Photo / Paper Replay)"
    assert res.liveness_score < 0.60
    assert "Spoof detected! Paper printing texture / low chrominance diversity found" in res.summary_text
    assert "LBP Entropy:" in res.summary_text
    assert "High-Freq:" in res.summary_text
    assert res.tier2_status == "ZERO-LEAKAGE ACTIVE: Feature extraction aborted..."


def test_screen_replay_classification_and_badge(detector):
    """Scenario 2: Screen monitor replay is intercepted with moire & glare metrics."""
    sample_path = PROJECT_ROOT / "samples" / "spoof_screen_replay.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is False
    assert res.attack_type == "SCREEN_REPLAY"
    assert res.badge == "SCREEN_REPLAY (Digital Monitor / Phone Screen)"
    assert res.liveness_score < 0.60
    assert "Spoof detected! Screen pixel grid interference / moiré pattern found" in res.summary_text
    assert "Moiré Score:" in res.summary_text
    assert "Glare:" in res.summary_text
    assert res.tier2_status == "ZERO-LEAKAGE ACTIVE: Feature extraction aborted..."


def test_zero_leakage_protocol_aborts_extraction():
    """Confirms that Zero-Leakage Protocol aborts Tier-2 before any templates are generated."""
    sample_spoof = PROJECT_ROOT / "samples" / "spoof_screen_replay.jpg"
    decision, report = run_pipeline(sample_spoof, demo=True, zero_leakage=True)
    
    assert decision is False
    assert report["processed"] is False
    assert report["status"] == "REJECTED"
    assert report["tier1_status"] == "REJECTED"
    assert report["attack_type"] == "SCREEN_REPLAY"
    assert report["badge"] == "SCREEN_REPLAY (Digital Monitor / Phone Screen)"
    assert report["tier2_status"] == "ZERO-LEAKAGE ACTIVE: Feature extraction aborted..."
    assert "img1_template_shape" not in report
    assert "raw_similarity" not in report
