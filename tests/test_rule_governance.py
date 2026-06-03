"""Mocked unit tests for procedural-rule governance (no DB, no LLM - patch src.memory.vs).

Pins the rule LIFECYCLE transitions, since these gate what reaches an audit prompt:
  test_approve_sets_learned_approved   - approve -> learned_approved (active)
  test_reject_sets_rejected_not_delete - reject keeps the row (status=rejected), never deletes
  test_retire_sets_retired             - retire deactivates an active rule, keeps the row
  test_delete_hard_removes             - delete is a hard DELETE (the footgun path)
  test_pending_rules_delegates         - pending_rules() returns vs.list_pending_rules()
  test_active_rules_delegates          - active_rules() returns vs.list_active_rules()
  test_similar_rules_delegates         - similar_rules(id) passes k through to vs.similar_rules
  test_learning_stores_human_decision  - learn_rules_from_findings passes human_decision through
                                         to add_rule as source_decision (provenance, not a gate)
  test_learning_not_gated_by_decision  - a 'reject' verdict still learns (the row is just tagged)
"""
from unittest.mock import patch
from src.memory import AgentMemorySystem as AMS
from src.state import RuleStatus


def test_approve_sets_learned_approved():
    with patch("src.memory.vs") as vs:
        AMS.approve_rule(7)
        vs.set_rule_status.assert_called_once_with(7, RuleStatus.LEARNED_APPROVED)


def test_reject_sets_rejected_not_delete():
    # The load-bearing test: reject keeps the row (so get_all_rule_contents dedup still sees it
    # and the agent won't re-propose it), it must NOT hard-delete.
    with patch("src.memory.vs") as vs:
        AMS.reject_rule(7)
        vs.set_rule_status.assert_called_once_with(7, RuleStatus.REJECTED)
        vs.delete_rule.assert_not_called()


def test_retire_sets_retired():
    with patch("src.memory.vs") as vs:
        AMS.retire_rule(7)
        vs.set_rule_status.assert_called_once_with(7, RuleStatus.RETIRED)
        vs.delete_rule.assert_not_called()


def test_delete_hard_removes():
    with patch("src.memory.vs") as vs:
        AMS.delete_rule(7)
        vs.delete_rule.assert_called_once_with(7)
        vs.set_rule_status.assert_not_called()


def test_pending_rules_delegates():
    with patch("src.memory.vs") as vs:
        vs.list_pending_rules.return_value = [{"id": 1, "category": "security", "content": "x"}]
        assert AMS.pending_rules() == [{"id": 1, "category": "security", "content": "x"}]
        vs.list_pending_rules.assert_called_once_with()


def test_active_rules_delegates():
    with patch("src.memory.vs") as vs:
        vs.list_active_rules.return_value = [{"id": 1, "category": "security",
                                             "status": "seeded", "content": "x"}]
        assert AMS.active_rules() == [{"id": 1, "category": "security",
                                       "status": "seeded", "content": "x"}]
        vs.list_active_rules.assert_called_once_with()


def test_similar_rules_delegates():
    with patch("src.memory.vs") as vs:
        vs.similar_rules.return_value = [{"id": 2, "status": "seeded",
                                          "content": "y", "similarity": 0.91}]
        assert AMS.similar_rules(7) == [{"id": 2, "status": "seeded",
                                         "content": "y", "similarity": 0.91}]
        vs.similar_rules.assert_called_once_with(7, k=3)


def _crit(desc):
    """A learnable (CRITICAL) finding dict, the shape finalize passes to learning."""
    return {"severity": "critical", "file_path": "auth/login.py", "description": desc}


def test_learning_stores_human_decision():
    # The PR's human verdict must reach add_rule as source_decision (provenance for the reviewer).
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = []          # nothing stored -> no dedup skip
        added = AMS.learn_rules_from_findings(
            security=[_crit("SQL injection via f-string in login")],
            human_decision="needs-changes",
        )
    assert added == 1
    _, kwargs = vs.add_rule.call_args
    assert kwargs["status"] == RuleStatus.LEARNED_PENDING
    assert kwargs["source_decision"] == "needs-changes"


def test_learning_not_gated_by_decision():
    # A 'reject' verdict does NOT stop learning - the rule is still captured (just tagged),
    # because the human approval gate (review_rules.py), not the PR verdict, is the real filter.
    with patch("src.memory.vs") as vs:
        vs.get_all_rule_contents.return_value = []
        added = AMS.learn_rules_from_findings(
            security=[_crit("SQL injection via f-string in login")],
            human_decision="reject",
        )
    assert added == 1
    assert vs.add_rule.call_args.kwargs["source_decision"] == "reject"
