from unittest.mock import patch
import src.nodes.plan as plan_mod


def test_plan_injects_recalled_rules_into_prompt():
    captured = {}

    def _fake_diff_cache(diff, instructions, response_model, max_output_tokens):
        # plan now caches the DIFF and varies the instructions - the rules ride in `instructions`.
        captured["instructions"] = instructions
        from src.state import AuditPlan, Severity
        parsed = plan_mod.PlanAuditOutput(
            reasoning="ok",
            plan=AuditPlan(focus_areas=["auth"], risk_level=Severity.HIGH,
                           audit_depth="deep", files_to_prioritize=[]),
        )
        return parsed, ""                                # (parsed, cache_note)

    # No need to patch recall: the plan READS rules from the `procedural` channel
    # (retrieve already populated it), so just seed the channel in the input state.
    # The channel is keyed by category VALUE strings ("security"/"quality"/"coverage"),
    # matching recall_all_rules' `out[cat.value]`.
    with patch.object(plan_mod, "audit_with_diff_cache_sync", side_effect=_fake_diff_cache):
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
