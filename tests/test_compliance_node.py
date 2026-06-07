# tests/test_compliance_node.py - the compliance node with the MCP tools and the triage LLM mocked.
from unittest.mock import patch, AsyncMock
import asyncio
import src.nodes.compliance as comp


def _triage_result(query):
    """compliance now triages through audit_with_diff_cache (diff cached), which returns
    (parsed_query, cache_note). The fake returns the parsed ComplianceQuery + an empty note."""
    async def _diff_cache(diff, instructions, response_model, max_output_tokens):
        return query, ""
    return _diff_cache


def _state(diff):
    return {"audit": {"parsed_diff": diff}}


def test_skips_when_no_diff():
    out = asyncio.run(comp.compliance_node({"audit": {}}))
    assert out["audit"]["compliance_context"] == []
    assert "no parsed diff" in out["audit"]["messages"][0]


def test_unregulated_diff_does_no_lookup():
    with patch.object(comp, "audit_with_diff_cache",
                      side_effect=_triage_result(comp.ComplianceQuery(needs_lookup=False, queries=[]))), \
         patch.object(comp, "load_mcp_tools", AsyncMock()) as load:
        out = asyncio.run(comp.compliance_node(_state("diff --git a/readme.md b/readme.md\n+typo")))
    assert out["audit"]["compliance_context"] == []
    load.assert_not_called()                    # never start the servers for a clean diff


def test_regulated_diff_runs_search_tool():
    tool = AsyncMock(); tool.name = "search_compliance_docs"
    tool.ainvoke.return_value = [{"text": "mask PII", "source": "GDPR Art. 32", "framework": "gdpr"}]
    with patch.object(comp, "audit_with_diff_cache",
                      side_effect=_triage_result(comp.ComplianceQuery(needs_lookup=True, queries=["PII logging"]))), \
         patch.object(comp, "load_mcp_tools", AsyncMock(return_value=[tool])):
        out = asyncio.run(comp.compliance_node(_state("diff --git a/log.py b/log.py\n+log.info(user.pan)")))
    assert out["audit"]["compliance_context"] == [{"text": "mask PII", "source": "GDPR Art. 32", "framework": "gdpr"}]
    tool.ainvoke.assert_awaited_once_with({"query": "PII logging", "k": 3})


def test_tool_unavailable_is_fail_soft():
    with patch.object(comp, "audit_with_diff_cache",
                      side_effect=_triage_result(comp.ComplianceQuery(needs_lookup=True, queries=["auth change"]))), \
         patch.object(comp, "load_mcp_tools", AsyncMock(return_value=[])):   # no tools loaded
        out = asyncio.run(comp.compliance_node(_state("diff --git a/auth.py b/auth.py\n+x")))
    assert out["audit"]["compliance_context"] == []
    assert "tool unavailable" in out["audit"]["messages"][0]   # visible, not a crash