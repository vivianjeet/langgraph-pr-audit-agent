"""Tests in this file: the COVERAGE (test-gap) audit node's core behaviour (LLM mocked).

- test_coverage_audit_returns_findings        : parsed findings reach state (parametrised by severity).
- test_coverage_audit_skips_when_no_diff       : empty diff -> skip, no findings, no LLM call.
- test_coverage_audit_falls_back_on_nonretryable_error: hard error -> empty findings + node_error.
- test_coverage_audit_degrades_on_transient_error     : transient error after retries -> empty findings.

(The coverage node's RULE injection into the prompt is pinned in test_audit_rules.py.)
"""
from unittest.mock import patch
import pytest
from src.state import CoverageFinding, Severity
import src.nodes.coverage_audit as coverage_mod
from src.nodes.coverage_audit import CoverageAuditOutput
import src.llm_retry as llm_retry

@pytest.fixture
def patched_create():
    with patch.object(llm_retry, "_raw_chat") as mock_create:
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
    out = coverage_mod.coverage_audit_node({"audit": {"parsed_diff": "[FILE]: x.py\n+ ttl = 86400", "messages" : []}})
    patched_create.assert_called_once()
    assert out["audit"]["test_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_coverage_audit_skips_when_no_diff(patched_create, empty):
    out = coverage_mod.coverage_audit_node({"audit": {"parsed_diff": empty, "messages" : []}})
    patched_create.assert_not_called()
    assert out["audit"]["test_findings"] == []
    assert "skipped" in out["audit"]["messages"][0]

def test_coverage_audit_falls_back_on_nonretryable_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")            # non-retryable -> called once
    out = coverage_mod.coverage_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    patched_create.assert_called_once()
    assert out["audit"]["test_findings"] == []

def test_coverage_audit_degrades_on_transient_error(patched_create):
    patched_create.side_effect = RuntimeError("503 Service Unavailable")
    out = coverage_mod.coverage_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    assert out["audit"]["test_findings"] == []
