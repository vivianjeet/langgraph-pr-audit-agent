# Pins merge_audit: the nested `audit` channel must restore the per-field reducer
# semantics AuditState declared (messages/node_errors accumulate; scalars overwrite).
# This is what keeps the parallel audit fan-in (security/quality/coverage) from
# clobbering each other's messages once AuditState lives one level down.
from src.memory import merge_audit


def test_messages_accumulate_like_operator_add():
    old = {"messages": ["a"], "node_errors": ["e1"]}
    new = {"messages": ["b"], "node_errors": ["e2"]}
    merged = merge_audit(old, new)
    assert merged["messages"] == ["a", "b"]
    assert merged["node_errors"] == ["e1", "e2"]


def test_scalars_are_last_writer_wins():
    merged = merge_audit({"security_score": 1.0}, {"security_score": 0.4})
    assert merged["security_score"] == 0.4


def test_simulated_parallel_fan_in_keeps_all_branch_messages():
    # Three audit branches each return their slice; folding them through merge_audit
    # (as LangGraph does for the `audit` channel) must keep ALL three messages.
    base = {"messages": ["plan done"]}
    for branch in ([{"messages": ["sec"]}, {"messages": ["qual"]}, {"messages": ["cov"]}]):
        base = merge_audit(base, branch)
    assert base["messages"] == ["plan done", "sec", "qual", "cov"]


def test_handles_none_and_missing_keys():
    assert merge_audit(None, {"messages": ["x"]})["messages"] == ["x"]
    assert merge_audit({"messages": ["x"]}, None)["messages"] == ["x"]
    # A scalar-only update doesn't crash on absent message lists.
    assert merge_audit({}, {"security_score": 0.5}) == {"security_score": 0.5, "messages": [], "node_errors": []}
