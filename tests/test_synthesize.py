import pytest

from src.nodes.synthesize_report import _weighted_score, synthesize_report_node
from src.state import SecurityFinding, Severity

def _f(sev) -> SecurityFinding:
    return SecurityFinding(file_path="a.py", line_number=1, description="x",
                           cwe_id="CWE-1", severity=sev)

def test_clean_is_one():
    assert _weighted_score([]) == 1.0

@pytest.mark.parametrize("sev, expected",[
    (Severity.CRITICAL, 0.4),       # 1 - 0.6
    (Severity.HIGH,     0.7),       # 1 - 0.3
    (Severity.MEDIUM,  0.85),       # 1 - 0.15
    (Severity.LOW,     0.93),       # 1 - 0.07
    (Severity.INFO,    0.98),       # 1 - 0.02
    (Severity.NONE,     1.0)        # 1 - 0.0

])
def test_single_finding_penalty(sev, expected):
    assert _weighted_score([_f(sev)]) == pytest.approx(expected)

def test_two_mediums_land_borderline():
    # 1 - (0.15 + 0.15) = 0.7 -> top of the re3flect band
    assert _weighted_score([_f(Severity.MEDIUM), _f(Severity.MEDIUM)]) == pytest.approx(0.7)

def test_score_clamped_at_zero():
    # penalty far exceeds 1.0 -> clamp to 0.0, never negative
    assert _weighted_score([_f(Severity.CRITICAL)]*5) == 0.0

def test_synthesize_writes_all_three_scores():
    out = synthesize_report_node({
        "security_findings": [_f(Severity.CRITICAL)],
        "quality_findings": [],
        "test_findings": [],

    })
    assert out["security_score"] == pytest.approx(0.4)
    assert out["quality_score"] == 1.0
    assert out["test_score"] == 1.0