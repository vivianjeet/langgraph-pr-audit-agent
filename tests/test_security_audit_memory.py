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


class _LLMRes:
    """Minimal stand-in for LLMResult: the node reads .output (+ token/cost fields on the cache/thinking
    notes). Lets a test return a parsed model through the router path without importing LLMResult."""
    def __init__(self, output):
        self.output = output
        self.input_tokens = self.output_tokens = self.cache_read_tokens = 0
        self.cost_usd = 0.0


def test_security_audit_injects_rules_into_prompt():
    captured = {}

    async def _fake_acall(tier, *, messages, response_model, max_output_tokens, cache=False, thinking=0):
        # unregulated path now routes through llm.acall (tier='balanced'); res.output is the parsed model.
        captured["system"] = next(m["content"] for m in messages if m["role"] == "system")
        return _LLMRes(sec_mod.SecurityAuditOutput(reasoning="ok", findings=[]))

    with patch.object(sec_mod.llm, "acall", side_effect=_fake_acall), \
         patch("src.nodes.security_audit.AMS.rules_block",
               return_value="RULE-SENTINEL: parameterised queries only\n\n"):
        asyncio.run(sec_mod.security_audit_node({
            "audit": {"parsed_diff": "diff --git a/src/login.py b/src/login.py\n+x",
                      "audit_plan": {"focus_areas": ["injection"]}},
        }))

    assert "RULE-SENTINEL" in captured["system"]      # the org rule reached the security prompt


def test_security_audit_uses_thinking_when_warranted():
    # A complex regulated diff (thinking_warranted -> True) must take the Pro extended-thinking call:
    # llm.acall(tier="powerful", thinking=THINKING_BUDGET), NOT the cache path. Proves the Day-34 gate
    # is wired. No LLM/DB: thinking_warranted is forced True and llm.acall is an async spy.
    import src.config as cfg
    captured = {}

    class _Res:
        output = sec_mod.SecurityAuditOutput(reasoning="ok", findings=[])
        input_tokens = output_tokens = cache_read_tokens = 0
        cost_usd = 0.0

    async def _fake_acall(tier, *, messages, response_model, max_output_tokens,
                          cache=False, thinking=0):
        captured.update(tier=tier, thinking=thinking, cache=cache)
        return _Res()

    with patch.object(sec_mod, "thinking_warranted", return_value=True), \
         patch.object(sec_mod.llm, "acall", side_effect=_fake_acall), \
         patch("src.nodes.security_audit.AMS.rules_block", return_value=""):
        asyncio.run(sec_mod.security_audit_node({
            "audit": {"parsed_diff": "[ADDED] x\n[ADDED] y",
                      "audit_plan": {"focus_areas": ["injection"]},
                      "compliance_context": [{"framework": "gdpr", "text": "t", "source": "s"},
                                             {"framework": "hipaa", "text": "t", "source": "s"}]},
        }))

    assert captured["thinking"] == cfg.THINKING_BUDGET   # took the thinking path
    assert captured["cache"] is False                    # thinking and cache are mutually exclusive
    assert captured["tier"] == "powerful"                # still Pro
