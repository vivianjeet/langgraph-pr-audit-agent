# Locks the AgentMemorySystem contract under the nested AMSState design:
# reads come FROM the audit substate, write helpers RETURN AMSState-shaped partial
# updates and never mutate the held state.
from unittest.mock import patch
from src.memory import AgentMemorySystem as AMS


def test_read_returns_value_from_audit_substate():
    ams = AMS({"audit": {"parsed_diff": "the diff"}})
    assert ams.read("parsed_diff") == "the diff"


def test_read_returns_default_for_missing_key_or_missing_audit():
    assert AMS({"audit": {}}).read("missing", "fallback") == "fallback"
    # No audit channel at all -> still safe, returns default.
    assert AMS({}).read("missing", "fallback") == "fallback"


def test_append_message_returns_nested_update_and_does_not_mutate():
    state = {"audit": {"messages": ["existing"]}}
    ams = AMS(state)
    update = ams.append_message("new")
    # AMSState-shaped partial update (merge_audit will accumulate it)...
    assert update == {"audit": {"messages": ["new"]}}
    # ...and the held state is untouched (the reducer, not the facade, merges).
    assert state == {"audit": {"messages": ["existing"]}}


def test_write_audit_wraps_fields_under_audit():
    assert AMS.write_audit(security_score=0.4) == {"audit": {"security_score": 0.4}}


def test_persistent_helpers_delegate_to_vectorstore():
    with patch("src.memory.vs") as vs:
        vs.retrieve_similar_prs.return_value = [{"pr_summary": "x"}]
        assert AMS.recall_similar_prs("diff", k=2) == [{"pr_summary": "x"}]
        vs.retrieve_similar_prs.assert_called_once_with("diff", k=2)

        AMS.recall_rules("auth")
        vs.get_rules.assert_called_once_with("auth")

        AMS.recall_episodes("diff", k=3)
        vs.retrieve_episodes.assert_called_once_with("diff", k=3)
