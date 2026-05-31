from unittest.mock import patch
import pytest
from src.state import QualityFinding, Severity
import src.nodes.quality_audit as qual_mod
from src.nodes.quality_audit import QualityAuditOutput

@pytest.fixture
def patched_create():
    with patch.object(qual_mod.client.chat.completions, "create") as mock_create:
        yield mock_create

def _finding(severity):
    return QualityFinding(
        file_path="src/service.py",
        line_number=10,
        description= "Magic number must be a named constant.",
        severity = severity
    )

@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.MEDIUM, Severity.LOW])
def test_quality_audit_returns_findings(patched_create, severity):
    patched_create.return_value = QualityAuditOutput(reasoning="r", findings=[_finding(severity)])
    out = qual_mod.quality_audit_node({"parsed_diff": "[FILE]: x.py\n+ ttl = 86400", "messages" : []})
    patched_create.assert_called_once()
    assert out["quality_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_quality_audit_skips_when_no_diff(patched_create, empty):
    out = qual_mod.quality_audit_node({"parsed_diff": empty, "messages" : []})
    patched_create.assert_not_called()
    assert out["quality_findings"] == []
    assert "skipped" in out["messages"][0]

def test_quality_audit_falls_back_on_api_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        out = qual_mod.quality_audit_node({"parsed_diff": "x", "messages": []})
    assert patched_create.call_count == 3
    assert out["quality_findings"] == []
    assert "failed after retries" in out["messages"][0]
