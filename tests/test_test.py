from unittest.mock import patch
import pytest
from src.state import CoverageFinding, Severity
import src.nodes.coverage_audit as coverage_mod
from src.nodes.coverage_audit import CoverageAuditOutput

@pytest.fixture
def patched_create():
    with patch.object(coverage_mod.client.chat.completions, "create") as mock_create:
        yield mock_create

def _finding(severity):
    return CoverageFinding(
        file_path="src/service.py",
        line_number=10,
        description= "Magic number must be a named constant.",
        severity = severity
    )

@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.MEDIUM, Severity.LOW])
def test_coverage_audit_returns_findings(patched_create, severity):
    patched_create.return_value = CoverageAuditOutput(reasoning="r", findings=[_finding(severity)])
    out = coverage_mod.coverage_audit_node({"parsed_diff": "[FILE]: x.py\n+ ttl = 86400", "messages" : []})
    patched_create.assert_called_once()
    assert out["test_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_coverage_audit_skips_when_no_diff(patched_create, empty):
    out = coverage_mod.coverage_audit_node({"parsed_diff": empty, "messages" : []})
    patched_create.assert_not_called()
    assert out["test_findings"] == []
    assert "skipped" in out["messages"][0]

def test_coverage_audit_falls_back_on_api_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")
    with patch("time.sleep"):                    # skip tenacity's backoff waits
        out = coverage_mod.coverage_audit_node({"parsed_diff": "x", "messages": []})
    assert patched_create.call_count == 3        # tenacity retried 3x
    assert out["test_findings"] == []
    assert "failed after retries" in out["messages"][0]
