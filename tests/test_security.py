"""Tests in this file: the SECURITY audit node's core behaviour.

- test_security_audit_passes_through_findings: parsed findings reach state (parametrised by severity).
- test_security_audit_skips_when_no_diff      : empty diff -> skip, no findings, no LLM call.
- test_security_audit_hits_real_llm           : INTEGRATION - calls Gemini against a vuln diff.
- test_security_audit_falls_back_on_nonretryable_error: hard error -> empty findings + node_error.
- test_security_audit_degrades_on_transient_error     : transient error after retries -> empty findings.

(The security node's RULE injection into the prompt is pinned in test_security_audit_memory.py.)
"""
from unittest.mock import patch
import pytest

from src.state import SecurityFinding, Severity
import src.nodes.security_audit as sec_mod
from src.nodes.security_audit import SecurityAuditOutput
import src.llm_retry as llm_retry

@pytest.fixture
def patched_create():
    """
    Patch the LLM once, each test sets .return_value to the fake it needs"""
    with patch.object(llm_retry, "_raw_chat") as mock_create:
        yield mock_create

def _finding(severity):
    return SecurityFinding(file_path="auth/login.py",
                           line_number=42,
                           description="Raw SQl from user input.",
                           cwe_id="CWE-89",
                           severity=severity)

@pytest.mark.parametrize("severity", [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM])
def test_security_audit_passes_through_findings(patched_create, vuln_diff, severity):
    patched_create.return_value = SecurityAuditOutput(reasoning="r", findings=[_finding(severity)])
    out = sec_mod.security_audit_node({"audit": {"parsed_diff": vuln_diff, "messages" : []}})
    patched_create.assert_called_once()
    assert out["audit"]["security_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_security_audit_skips_when_no_diff(patched_create, empty):
    out = sec_mod.security_audit_node({"audit": {"parsed_diff": empty, "messages" : []}})
    patched_create.assert_not_called()
    assert out["audit"]["security_findings"] == []
    assert "skipped" in out["audit"]["messages"][0]

@pytest.mark.integration
def test_security_audit_hits_real_llm(vuln_diff):
    out = sec_mod.security_audit_node({"audit": {"parsed_diff": vuln_diff, "messages" : []}})
    assert isinstance(out["audit"]["security_findings"], list)

def test_security_audit_falls_back_on_nonretryable_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")            # non-retryable -> called once
    out = sec_mod.security_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    patched_create.assert_called_once()
    assert out["audit"]["security_findings"] == []

def test_security_audit_degrades_on_transient_error(patched_create):
    patched_create.side_effect = RuntimeError("503 Service Unavailable")
    out = sec_mod.security_audit_node({"audit": {"parsed_diff": "x", "messages": []}})
    assert out["audit"]["security_findings"] == []
