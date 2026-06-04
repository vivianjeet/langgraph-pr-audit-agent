# tests/test_compliance_node.py - the compliance node with MCP tools + the triage 
# LLM mocked.
from unittest.mock import patch, AsyncMock
import asyncio
import src.nodes.compliance as comp


def _state(diff):
    return {"audit": {"parsed_diff": diff}}


def test_skips_when_no_diff():
    out = asyncio.run(comp.compliance_node({"audit": {}}))
    assert out["audit"]["compliance_context"] == []
    assert "no parsed diff" in out["audit"]["messages"][0]


def test_unregulated_diff_does_no_lookup():
    async def _triage(*a, **k):
        return comp.ComplianceQuery(needs_lookup=False, queries=[])
    with patch.object(comp, "call_gemini_async", side_effect=_triage), \
         patch.object(comp, "load_mcp_tools", AsyncMock()) as load:
        out = asyncio.run(comp.compliance_node(_state("diff --git a/readme.md b/readme.md\n+typo")))
    assert out["audit"]["compliance_context"] == []
    load.assert_not_called()                    # we never even start the 
                                                # servers for a clean diff


def test_regulated_diff_runs_search_tool():
    async def _triage(*a, **k):
        return comp.ComplianceQuery(needs_lookup=True, queries=["PII logging"])
    tool = AsyncMock(); tool.name = "search_compliance_docs"
    tool.ainvoke.return_value = [{"text": "mask PII", "source": "GDPR Art. 32", "framework": "gdpr"}]
    with patch.object(comp, "call_gemini_async", side_effect=_triage), \
         patch.object(comp, "load_mcp_tools", AsyncMock(return_value=[tool])):
        out = asyncio.run(comp.compliance_node(_state("diff --git a/log.py b/log.py\n+log.info(user.pan)")))
    assert out["audit"]["compliance_context"] == [{"text": "mask PII", "source": "GDPR Art. 32", "framework": "gdpr"}]
    tool.ainvoke.assert_awaited_once_with({"query": "PII logging", "k": 3})


def test_tool_unavailable_is_fail_soft():
    async def _triage(*a, **k):
        return comp.ComplianceQuery(needs_lookup=True, queries=["auth change"])
    with patch.object(comp, "call_gemini_async", side_effect=_triage), \
         patch.object(comp, "load_mcp_tools", AsyncMock(return_value=[])):   # no tools loaded
        out = asyncio.run(comp.compliance_node(_state("diff --git a/auth.py b/auth.py\n+x")))
    assert out["audit"]["compliance_context"] == []
    assert "tool unavailable" in out["audit"]["messages"][0]   # visible, not a crash