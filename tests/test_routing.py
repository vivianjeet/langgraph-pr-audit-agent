"""Tests in this file: the conditional-edge ROUTING predicates (pure functions, no LLM/DB).

Reflexion gate:
- test_reflect_on_borderline_score      : borderline scores trigger a reflexion loop.
- test_reflect_triggers_on_any_dimension: any one low dimension triggers reflexion.
- test_no_reflect_outside_band          : scores outside the band do NOT reflect.
- test_reflect_on_auth_silence          : auth-touching diff with no findings reflects (suspicious silence).
- test_reflect_capped_after_two_loops   : reflexion is capped (no infinite loop).

Human-review gate:
- test_human_review_on_any_low_score    : any low score escalates to human review.
- test_human_review_on_critical_finding : a critical finding escalates to human review.
- test_no_human_review_when_clean       : a clean run skips human review.

End-to-end routing decisions:
- test_route_clean_finalizes            : clean -> finalize.
- test_route_borderline_reflects        : borderline -> reflect.
- test_route_low_score_to_human         : low score -> human.
- test_route_critical_outranks_reflect  : critical finding outranks the reflect path.
"""
import pytest

from src.graph import should_reflect, needs_human_review, route_after_synthesis
from src.state import SecurityFinding, Severity

def _sec(sev: Severity) -> SecurityFinding:
    return SecurityFinding(file_path="auth/login.py",
                           line_number=1,
                           description="x",
                           cwe_id="CWE-89",
                           severity=sev, 
                           title="SQL Injection")

def _make_state(**overrides) -> dict:
    """
    Clean baseline; each test overrides only what it excercises (keep tests readable).
    Predicates read the audit substate, so the baseline is wrapped under `audit`.
    """
    base = {
        "security_score": 1.0,
        "quality_score" : 1.0,
        "test_score" : 1.0,
        "security_findings": [],
        "quality_findings": [],
        "test_findings": [],
        "files_changed": ["README.md"],
        "iteration_count": 0
    }
    base.update(overrides)
    return {"audit": base}

# --------------- should_reflect ---------------------
@pytest.mark.parametrize("score", [0.5, 0.6, 0.7])
def test_reflect_on_borderline_score(score):
    assert should_reflect(_make_state(security_score=score)) is True

@pytest.mark.parametrize("key", ["security_score", "quality_score", "test_score"])
def test_reflect_triggers_on_any_dimension(key):
    assert should_reflect(_make_state(**{key: 0.6})) is True

@pytest.mark.parametrize("score",[0.49, 0.71, 1.0])
def test_no_reflect_outside_band(score):
    assert should_reflect(_make_state(security_score=score)) is False

def test_reflect_on_auth_silence():
    # auth file changed but zero security findings = suspicious quiet
    assert should_reflect(_make_state(files_changed=["auth/login.py"])) is True

def test_reflect_capped_after_two_loops():
    assert should_reflect(_make_state(security_score=0.6, iteration_count=2)) is False

# --------------- needs_human_review ---------------------
@pytest.mark.parametrize("key", ["security_score", "quality_score", "test_score"])
def test_human_review_on_any_low_score(key):
    # FAILS untill need_human_review checks all 3 scores (dedent the `return any(...)`).
    assert needs_human_review(_make_state(**{key: 0.4})) is True

def test_human_review_on_critical_finding():
    assert needs_human_review(_make_state(security_findings=[_sec(Severity.CRITICAL)])) is True

def test_no_human_review_when_clean():
    assert needs_human_review(_make_state()) is False

# ----------- route_after_synthesis (precedence: human_review > reflect > finalize) --------
def test_route_clean_finalizes():
    assert route_after_synthesis(_make_state()) == "finalize"

def test_route_borderline_reflects():
    assert route_after_synthesis(_make_state(security_score=0.6, files_changed=["x.py"])) == "reflect"

def test_route_low_score_to_human():
    assert route_after_synthesis(_make_state(security_score=0.4)) == "human_review"

def test_route_critical_outranks_reflect():
    # CRITICAL + borderline 0.6: human review MUST win. Proves needs_human_review is
    # checked BEFORE should_reflect. FAILS if route still checks should_reflect first.
    state = _make_state(security_score=0.6, security_findings=[_sec(Severity.CRITICAL)])
    assert route_after_synthesis(state) == "human_review"