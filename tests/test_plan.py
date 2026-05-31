from unittest.mock import patch
import pytest

from src.state import AuditPlan, Severity, AuditDepth
import src.nodes.plan as plan_mod
from src.nodes.plan import PlanAuditOutput as PlanOut


@pytest.fixture
def patched_create():
    with patch.object(plan_mod.client.chat.completions, "create") as mock_create:
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
        {"parsed_diff": "[FILE]: auth/login.py\n+ q=...", "files_changed": ["auth/login.py"], "messages": []}
    )
    patched_create.assert_called_once()
    assert out["audit_plan"]["audit_depth"] == "deep"
    assert out["audit_plan"]["risk_level"] == "high"
    assert out["audit_plan"]["focus_areas"] == ["sql injection"]


def test_plan_filters_hallucinated_files(patched_create):
    # model returns a file that isn't in the diff -> must be dropped
    patched_create.return_value = _output(["x"], ["auth/login.py", "ghost.py"])
    out = plan_mod.plan_audit_node(
        {"parsed_diff": "[FILE]: auth/login.py\n+ q=...", "files_changed": ["auth/login.py"], "messages": []}
    )
    assert out["audit_plan"]["files_to_prioritize"] == ["auth/login.py"]


@pytest.mark.parametrize("empty", ["", "   ", "\n"])
def test_plan_skips_when_no_diff(patched_create, empty):
    out = plan_mod.plan_audit_node({"parsed_diff": empty, "files_changed": [], "messages": []})
    patched_create.assert_not_called()
    assert out["audit_plan"]["audit_depth"] == "shallow"   # the default plan
    assert "skipped" in out["messages"][0]


def test_plan_falls_back_on_api_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")
    with patch("time.sleep"):                              # skip tenacity's backoff waits
        out = plan_mod.plan_audit_node(
            {"parsed_diff": "[FILE]: x.py\n+ a=1", "files_changed": ["x.py"], "messages": []}
        )
    assert patched_create.call_count == 3                  # tenacity retried 3x
    assert out["audit_plan"]["audit_depth"] == "shallow"   # default plan
    assert "failed after retries" in out["messages"][0]