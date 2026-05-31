from unittest.mock import patch
import pytest

from src.state import SecurityFinding, Severity
import src.nodes.security_audit as sec_mod
from src.nodes.security_audit import SecurityAuditOutput

@pytest.fixture
def patched_create():
    """
    Patch the LLM once, each test sets .return_value to the fake it needs"""
    with patch.object(sec_mod.client.chat.completions, "create") as mock_create:
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
    out = sec_mod.security_audit_node({"parsed_diff": vuln_diff, "messages" : []})
    patched_create.assert_called_once()
    assert out["security_findings"][0].severity == severity

@pytest.mark.parametrize("empty", ["", "    ", "\n"])
def test_security_audit_skips_when_no_diff(patched_create, empty):
    out = sec_mod.security_audit_node({"parsed_diff": empty, "messages" : []})
    patched_create.assert_not_called()
    assert out["security_findings"] == []
    assert "skipped" in out["messages"][0]

@pytest.mark.integration
def test_security_audit_hits_real_llm(vuln_diff):
    out = sec_mod.security_audit_node({"parsed_diff": vuln_diff, "messages" : []})
    assert isinstance(out["security_findings"], list)

def test_security_audit_falls_back_on_api_error(patched_create):
    patched_create.side_effect = RuntimeError("boom")
    with patch("time.sleep"):
        out = sec_mod.security_audit_node({"parsed_diff": "x", "messages": []})
    assert patched_create.call_count == 3
    assert out["security_findings"] == []
    assert "failed after retries" in out["messages"][0]
