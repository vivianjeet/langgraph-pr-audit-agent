from unittest.mock import patch
import src.nodes.reflexion as refl_mod
from src.nodes.reflexion import ReflectionOutput   # note: ReflectionOutput, not Reflexion
import src.llm_retry as llm_retry

def test_reflection_increments_iteration_and_records_gaps():
    fake = ReflectionOutput(
        gaps_identified=["missed CSRF"],
        additional_checks_needed=["check token rotation"],
        confidence_score=0.6
    )
    with patch.object(llm_retry, "_raw_chat", return_value=fake):
        out = refl_mod.reflexion_node({"messages": ["prior"], "iteration_count": 0,
                                       "security_findings": [], "security_score": 0.6})
    assert out["iteration_count"] == 1
    assert out["confidence_score"] == 0.6
    assert "missed CSRF" in out["gaps_identified"]

def test_reflexion_failure_still_increments_and_keeps_findings():
    # Loop-guard regression: a persistent failure must STILL advance iteration_count
    # (else infinite loop) and must NOT return security_findings:[] (that wiped real findings).
    with patch.object(llm_retry, "_raw_chat", side_effect=RuntimeError("boom")):
        out = refl_mod.reflexion_node({"messages": [], "iteration_count": 1,
                                       "security_findings": [{"x": 1}], "security_score": 0.6})
    assert out["iteration_count"] == 2                 # guard advanced on failure
    assert "security_findings" not in out              # did NOT wipe existing findings


def test_reflexion_quota_exhaustion_propagates():
    # Daily-quota -> QuotaExhaustedError must NOT be swallowed (fail closed).
    import pytest
    with patch.object(llm_retry, "_raw_chat",
                      side_effect=Exception("429 RESOURCE_EXHAUSTED ... PerDay ...")), \
         patch.object(llm_retry, "_KEYS", ["only-key"]):     # single key -> no rotation escape
        with pytest.raises(llm_retry.QuotaExhaustedError):
            refl_mod.reflexion_node({"messages": [], "iteration_count": 0,
                                     "security_findings": [], "security_score": 0.6})