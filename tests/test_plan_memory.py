from unittest.mock import patch
import src.nodes.plan as plan_mod
from src.llm_client import LLMResult


def test_plan_injects_recalled_rules_into_prompt():
    captured = {}

    def _fake_call(tier, *, messages, response_model, max_output_tokens, cache=False):
        # plan now caches the DIFF (messages[0]) and varies the instructions (messages[1]) through the
        # sync router (llm.call(cache=True)); the rules ride in the instructions. The cache path returns
        # native JSON in res.output, which the node parses with model_validate_json.
        captured["instructions"] = messages[1]["content"]
        from src.state import AuditPlan, Severity
        parsed = plan_mod.PlanAuditOutput(
            reasoning="ok",
            plan=AuditPlan(focus_areas=["auth"], risk_level=Severity.HIGH,
                           audit_depth="deep", files_to_prioritize=[]),
        )
        return LLMResult(output=parsed.model_dump_json(), model="gemini-2.5-flash")

    # No need to patch recall: the plan READS rules from the `procedural` channel
    # (retrieve already populated it), so just seed the channel in the input state.
    # The channel is keyed by category VALUE strings ("security"/"quality"/"coverage"),
    # matching recall_all_rules' `out[cat.value]`.
    with patch.object(plan_mod.llm, "call", side_effect=_fake_call):
        plan_mod.plan_audit_node({
            "audit": {
                "parsed_diff": "diff --git a/src/login.py b/src/login.py\n+x",
                "files_changed": ["src/login.py"],
            },
            "semantic": [],
            "episodic": [],
            "procedural": {"security": ["RULE-SENTINEL: parameterised queries only"]},
        })

    assert "RULE-SENTINEL" in captured["instructions"]   # the org rule reached the LLM prompt
