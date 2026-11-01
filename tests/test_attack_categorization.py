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


def test_heuristic_pass_does_not_claim_genuine_or_live(detector):
    """A static-image heuristic pass is not evidence of a live human."""
    sample_path = PROJECT_ROOT / "samples" / "genuine_personA_1.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is True
    assert res.attack_type == "UNVERIFIED"
    assert res.badge == "UNVERIFIED (Heuristic Pass; Liveness Not Established)"
    assert res.liveness_score >= 0.60
    assert "does not establish a genuine person or live capture" in res.summary_text
    assert res.tier2_status == "NOT_RUN (heuristic analysis only)"


def test_print_attack_classification_and_badge(detector):
    """Scenario 1: Printed paper attack is intercepted with exact badges & metrics."""
    sample_path = PROJECT_ROOT / "samples" / "spoof_print_attack.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is False
    assert res.attack_type == "PRINT_ATTACK"
    assert res.badge == "POSSIBLE PRINT ATTACK (Heuristic)"
    assert res.liveness_score < 0.60
    assert "Possible print-like pattern" in res.summary_text
    assert "heuristics are inconclusive" in res.summary_text
    assert "LBP Entropy:" in res.summary_text
    assert "High-Freq:" in res.summary_text
    assert res.tier2_status == "NOT_RUN (heuristic rejection only)"


def test_screen_replay_classification_and_badge(detector):
    """Scenario 2: Screen monitor replay is intercepted with moire & glare metrics."""
    sample_path = PROJECT_ROOT / "samples" / "spoof_screen_replay.jpg"
    res = detector.evaluate(sample_path)
    
    assert res.is_live is False
    assert res.attack_type == "SCREEN_REPLAY"
    assert res.badge == "POSSIBLE SCREEN REPLAY (Heuristic)"
    assert res.liveness_score < 0.60
    assert "Possible screen-replay-like pattern" in res.summary_text
    assert "heuristics are inconclusive" in res.summary_text
    assert "Moiré Score:" in res.summary_text
    assert "Glare:" in res.summary_text
    assert res.tier2_status == "NOT_RUN (heuristic rejection only)"


def test_zero_leakage_protocol_aborts_extraction():
    """Confirms that Zero-Leakage Protocol aborts Tier-2 before any templates are generated."""
    sample_spoof = PROJECT_ROOT / "samples" / "spoof_screen_replay.jpg"
    decision, report = run_pipeline(sample_spoof, demo=True, zero_leakage=True)
    
    assert decision is False
    assert report["processed"] is False
    assert report["status"] == "REJECTED"
    assert report["tier1_status"] == "REJECTED"
    assert report["attack_type"] == "SCREEN_REPLAY"
    assert report["badge"] == "POSSIBLE SCREEN REPLAY (Heuristic)"
    assert report["tier2_status"] == "ZERO-LEAKAGE ACTIVE: Feature extraction aborted..."
    assert "img1_template_shape" not in report
    assert "raw_similarity" not in report
