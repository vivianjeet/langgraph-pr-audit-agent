"""Tests in this file: the SECURITY audit node's procedural-rule injection (16.6).

WHAT IS PINNED
- test_security_audit_injects_rules_into_prompt:
    Proves a procedural rule reaches the security node's SYSTEM prompt (audit nodes
    inject rules into the system prompt; the 16.5 plan test pins the USER prompt).

HOW IT WORKS (no DB, no LLM)
- `call_gemini_async` is patched with an async spy that captures the system message the node
  built and returns a minimal valid SecurityAuditOutput so the node completes.
- `AMS.rules_block` is patched to return a sentinel block string directly, so the test
  is independent of the channel's keys/contents - it only asserts the WIRING: whatever
  rules_block returns must land in the system prompt via the {{rules}} .replace().

WHY rules_block is patched (not the channel seeded)
- The node calls `AMS.rules_block(state["procedural"], (RuleCategory.SECURITY,))`. Patching
  rules_block lets us assert the node injects its formatted output, without depending on
  category-key plumbing (covered separately by the plan/learn tests).
"""
import asyncio
from unittest.mock import patch
import src.nodes.security_audit as sec_mod


def test_security_audit_injects_rules_into_prompt():
    captured = {}

    async def _fake_call_gemini(model, messages, response_model, max_output_tokens):
        captured["system"] = next(m["content"] for m in messages if m["role"] == "system")
        return sec_mod.SecurityAuditOutput(reasoning="ok", findings=[])

    with patch.object(sec_mod, "call_gemini_async", side_effect=_fake_call_gemini), \
         patch("src.nodes.security_audit.AMS.rules_block",
               return_value="RULE-SENTINEL: parameterised queries only\n\n"):
        asyncio.run(sec_mod.security_audit_node({
            "audit": {"parsed_diff": "diff --git a/src/login.py b/src/login.py\n+x",
                      "audit_plan": {"focus_areas": ["injection"]}},
        }))

    assert "RULE-SENTINEL" in captured["system"]      # the org rule reached the security prompt
