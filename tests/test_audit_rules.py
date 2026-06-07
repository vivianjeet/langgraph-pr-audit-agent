"""Tests in this file: QUALITY + COVERAGE audit nodes' procedural-rule injection (16.6).

Mirror of test_security_audit_memory.py for the other two audit domains. Each audit
node injects ONLY its own category's rules into its SYSTEM prompt.

WHAT IS PINNED
- test_quality_audit_injects_rules_into_prompt : a rule reaches the quality node's prompt.
- test_coverage_audit_injects_rules_into_prompt: a rule reaches the coverage node's prompt.

HOW IT WORKS (no DB, no LLM)
- `audit_with_diff_cache` patched with an async spy that captures the `instructions` it receives
  (the rendered system prompt, the per-node variable part) and returns a minimal valid
  <Domain>AuditOutput + empty cache-note so the node completes.
- `AMS.rules_block` patched to return a sentinel block, so the test asserts the WIRING
  ({{rules}} .replace into the instructions), independent of channel keys/contents.

WHY this is symmetric with security
- Which node enforces which rule is a property of the rule's CATEGORY, not a privileged
  node. quality reads `quality`, coverage reads `coverage`; the wiring is identical.
"""
import asyncio
from unittest.mock import patch
import src.nodes.quality_audit as qual_mod
import src.nodes.coverage_audit as cov_mod


def test_quality_audit_injects_rules_into_prompt():
    captured = {}

    async def _fake(diff, instructions, response_model, max_output_tokens, **kw):
        captured["instructions"] = instructions
        return qual_mod.QualityAuditOutput(reasoning="ok", findings=[]), ""

    with patch.object(qual_mod, "audit_with_diff_cache", side_effect=_fake), \
         patch("src.nodes.quality_audit.AMS.rules_block",
               return_value="RULE-SENTINEL: no god objects\n\n"):
        asyncio.run(qual_mod.quality_audit_node({
            "audit": {"parsed_diff": "diff --git a/x.py b/x.py\n+x",
                      "audit_plan": {"focus_areas": []}},
        }))
    assert "RULE-SENTINEL" in captured["instructions"]


def test_coverage_audit_injects_rules_into_prompt():
    captured = {}

    async def _fake(diff, instructions, response_model, max_output_tokens, **kw):
        captured["instructions"] = instructions
        return cov_mod.CoverageAuditOutput(reasoning="ok", findings=[]), ""

    with patch.object(cov_mod, "audit_with_diff_cache", side_effect=_fake), \
         patch("src.nodes.coverage_audit.AMS.rules_block",
               return_value="RULE-SENTINEL: new fn needs a test\n\n"):
        asyncio.run(cov_mod.coverage_audit_node({
            "audit": {"parsed_diff": "diff --git a/x.py b/x.py\n+x",
                      "audit_plan": {"focus_areas": []}},
        }))
    assert "RULE-SENTINEL" in captured["instructions"]
