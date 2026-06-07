"""Tests in this file: the QUALITY audit node's core behaviour (LLM mocked).

- test_quality_audit_returns_findings        : parsed findings reach state (parametrised by severity).
- test_quality_audit_skips_when_no_diff       : empty diff -> skip, no findings, no LLM call.
- test_quality_audit_falls_back_on_nonretryable_error: hard error -> empty findings + node_error.
- test_quality_audit_degrades_on_transient_error     : transient error after retries -> empty findings.

(The quality node's RULE injection into the prompt is pinned in test_audit_rules.py.)
"""
from unittest.mock import patch
import pytest
from src.state import QualityFinding, Severity
import src.nodes.quality_audit as qual_mod
from src.nodes.quality_audit import QualityAuditOutput
import src.llm_retry as llm_retry
import asyncio

@pytest.fixture
def patched_create():
    with patch.object(llm_retry, "_raw_chat") as mock_create:
        yield mock_create

def _finding(severity):
    return QualityFinding(
        file_path="src/service.py",
        line_number=10,
        description= "Magic number must be a named constant.",
        severity = severity,
        title="Magic Number"
    )

@pytest.mark.parametrize("severity", [Severity.HIGH, Severity.MEDIUM, Severity.LOW])
def test_quality_audit_returns_findings(patched_create, severity):
    patched_create.return_value = QualityAuditOutput(reasoning="r", findings=[_finding(severity)])
    out = asyncio.run(
        qual_mod.quality_audit_node({"audit": {"parsed_diff": "[FILE]: x.py\n+ ttl = 86400", "messages" : []}})
    )
    patched_create.assert_called_once()
    assert out["audit"]["quality_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_quality_audit_skips_when_no_diff(patched_create, empty):
    out = asyncio.run(
        qual_mod.quality_audit_node({"audit": {"parsed_diff": empty, "messages" : []}})
    )
    patched_create.assert_not_called()
    assert out["audit"]["quality_findings"] == []
    assert "skipped" in out["audit"]["messages"][0]

def test_quality_audit_falls_back_on_nonretryable_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")            # non-retryable -> called once
    out = asyncio.run(
        qual_mod.quality_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    )
    patched_create.assert_called_once()
    assert out["audit"]["quality_findings"] == []

def test_quality_audit_degrades_on_transient_error(patched_create):
    patched_create.side_effect = RuntimeError("503 Service Unavailable")
    out = asyncio.run(
        qual_mod.quality_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    )
    assert out["audit"]["quality_findings"] == []
