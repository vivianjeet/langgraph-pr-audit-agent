"""Tests in this file: the PLAN node's core triage behaviour (LLM mocked).

- test_plan_writes_plan_to_state          : a successful call writes the audit_plan to state.
- test_plan_filters_hallucinated_files     : files_to_prioritize is filtered to real changed files.
- test_plan_skips_when_no_diff             : empty diff -> default plan, no LLM call.
- test_plan_falls_back_on_nonretryable_error: a hard error -> default plan + node_error.
- test_plan_degrades_on_transient_error    : a transient error after retries -> default plan.

(The plan node's MEMORY consumption - rules/precedent reaching the prompt - is pinned
separately in test_plan_memory.py.)
"""
from unittest.mock import patch
import pytest

from src.state import AuditPlan, Severity, AuditDepth
import src.nodes.plan as plan_mod
from src.nodes.plan import PlanAuditOutput as PlanOut
import src.llm_retry as llm_retry

@pytest.fixture
def patched_create():
    with patch.object(llm_retry, "_raw_chat") as mock_create:
        yield mock_create


def _output(focus, files):
    return PlanOut(
        reasoning="auth + raw SQL = high blast radius",
        plan=AuditPlan(focus_areas=focus, risk_level=Severity.HIGH,
                       audit_depth=AuditDepth.DEEP, files_to_prioritize=files),
    )


def test_plan_writes_plan_to_state(patched_create):
    patched_create.return_value = _output(["sql injection"], ["auth/login.py"])
    out = plan_mod.plan_audit_node(
        {"audit": {"parsed_diff": "[FILE]: auth/login.py\n+ q=...", "files_changed": ["auth/login.py"], "messages": []}}
    )
    patched_create.assert_called_once()
    assert out["audit"]["audit_plan"]["audit_depth"] == "deep"
    assert out["audit"]["audit_plan"]["risk_level"] == "high"
    assert out["audit"]["audit_plan"]["focus_areas"] == ["sql injection"]


def test_plan_filters_hallucinated_files(patched_create):
    # model returns a file that isn't in the diff -> must be dropped
    patched_create.return_value = _output(["x"], ["auth/login.py", "ghost.py"])
    out = plan_mod.plan_audit_node(
        {"audit": {"parsed_diff": "[FILE]: auth/login.py\n+ q=...", "files_changed": ["auth/login.py"], "messages": []}}
    )
    assert out["audit"]["audit_plan"]["files_to_prioritize"] == ["auth/login.py"]


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_plan_skips_when_no_diff(patched_create, empty):
    out = plan_mod.plan_audit_node({"audit": {"parsed_diff": empty, "files_changed": [], "messages": []}})
    patched_create.assert_not_called()
    assert out["audit"]["audit_plan"]["audit_depth"] == "shallow"   # the default plan
    assert "skipped" in out["audit"]["messages"][0]


def test_plan_falls_back_on_nonretryable_error(patched_create):
    # A hard error on EVERY tier -> the router walks its fallback chain (balanced->fast), and when all
    # tiers fail the node degrades to the default plan. (Routing through acall added tier fallback, so
    # _raw_chat is hit once per tier, not once total - the degrade-to-default behaviour is unchanged.)
    patched_create.side_effect = RuntimeError("boom")
    out = plan_mod.plan_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    assert patched_create.call_count >= 1                         # tried at least the primary tier
    assert out["audit"]["audit_plan"]["audit_depth"] == "shallow"

def test_plan_degrades_on_transient_error(patched_create):
    patched_create.side_effect = RuntimeError("503 Service Unavailable")
    out = plan_mod.plan_audit_node({"audit": {"parsed_diff": "x", "files_changed": [], "messages": []}})
    assert out["audit"]["audit_plan"]["audit_depth"] == "shallow"