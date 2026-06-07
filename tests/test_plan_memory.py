from unittest.mock import patch
import src.nodes.plan as plan_mod


def test_plan_injects_recalled_rules_into_prompt():
    captured = {}

    def _fake_call_gemini(model, messages, response_model, max_output_tokens):
        captured["user"] = next(m["content"] for m in messages if m["role"] == "user")
        # return a minimal valid plan so the node completes
        from src.state import AuditPlan, Severity
        return plan_mod.PlanAuditOutput(
            reasoning="ok",
            plan=AuditPlan(focus_areas=["auth"], risk_level=Severity.HIGH,
                           audit_depth="deep", files_to_prioritize=[]),
        )

    # No need to patch recall: the plan READS rules from the `procedural` channel
    # (retrieve already populated it), so just seed the channel in the input state.
    # The channel is keyed by category VALUE strings ("security"/"quality"/"coverage"),
    # matching recall_all_rules' `out[cat.value]`.
    with patch.object(plan_mod, "call_gemini", side_effect=_fake_call_gemini):
        plan_mod.plan_audit_node({
            "audit": {
                "parsed_diff": "diff --git a/src/login.py b/src/login.py\n+x",
                "files_changed": ["src/login.py"],
            },
            "semantic": [],
            "episodic": [],
            "procedural": {"security": ["RULE-SENTINEL: parameterised queries only"]},
        })

    assert "RULE-SENTINEL" in captured["user"]      # the org rule reached the LLM prompt
