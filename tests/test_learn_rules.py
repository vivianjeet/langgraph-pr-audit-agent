from unittest.mock import patch
from src.memory import AgentMemorySystem as AMS
from src.state import RuleCategory, RuleStatus


def _f(severity, desc, fp="src/login.py"):
    return {"severity": severity, "file_path": fp, "description": desc}


def test_learns_high_severity_as_pending_and_maps_category():
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = []        # nothing stored yet
        added = AMS.learn_rules_from_findings(
            security=[_f("critical", "SQL injection via f-string")],
            quality=[_f("high", "God object: 600-line class")],
            coverage=[_f("high", "No test for the auth bypass path")],
        )
    assert added == 3
    # each finding stored under its DOMAIN category, ALL as learned_pending:
    cats = {c.args[0]: c.kwargs.get("status") for c in vs.add_rule.call_args_list}
    assert cats == {
        RuleCategory.SECURITY: RuleStatus.LEARNED_PENDING,
        RuleCategory.QUALITY: RuleStatus.LEARNED_PENDING,
        RuleCategory.COVERAGE: RuleStatus.LEARNED_PENDING,
    }


def test_skips_low_and_medium_severity():
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = []
        added = AMS.learn_rules_from_findings(
            security=[_f("medium", "minor"), _f("low", "nit"), _f("info", "fyi")],
        )
    assert added == 0
    vs.add_rule.assert_not_called()


def test_dedups_against_existing_rules_any_status():
    # a finding whose text already exists (e.g. a pending copy from last run) is NOT re-added
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = ["SQL injection via f-string"]
        added = AMS.learn_rules_from_findings(
            security=[_f("critical", "SQL injection via f-string")],   # duplicate
        )
    assert added == 0
    vs.add_rule.assert_not_called()


def test_dedups_identical_findings_within_one_run():
    # two identical findings in the SAME call insert only once (intra-run dedup):
    # the first insert adds the rule to the in-memory `existing` set, so the second is skipped.
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = []        # DB empty - dedup is purely intra-run
        added = AMS.learn_rules_from_findings(
            security=[_f("high", "Same issue"), _f("high", "Same issue")],
        )
    assert added == 1
    assert vs.add_rule.call_count == 1