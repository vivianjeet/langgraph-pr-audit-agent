"""Tests in this file: the baseline-rule SEED script's idempotency (DB mocked).

- test_seed_inserts_all_when_db_empty   : an empty DB gets every SEED_RULES entry, all status=seeded.
- test_seed_is_idempotent_skips_existing : a re-run with everything already present inserts nothing.
- test_seed_inserts_only_new_on_rerun    : adding a new rule + re-running inserts ONLY the new one.

All DB access (vs.get_all_rule_contents / vs.add_rule) is patched - no Postgres needed.
"""
from unittest.mock import patch
import scripts.seed_rules as seed_mod
from src.state import RuleStatus


def _total_seed_rules():
    return sum(len(v) for v in seed_mod.SEED_RULES.values())


def test_seed_inserts_all_when_db_empty():
    with patch.object(seed_mod.vs, "get_all_rule_contents", return_value=[]), \
         patch.object(seed_mod.vs, "add_rule") as add_rule:
        added = seed_mod.seed()

    assert added == _total_seed_rules()
    assert add_rule.call_count == _total_seed_rules()
    # every insert is tagged seeded (active immediately, unlike learned_pending)
    for call in add_rule.call_args_list:
        assert call.kwargs.get("status") == RuleStatus.SEEDED


def test_seed_is_idempotent_skips_existing():
    # DB already holds every seed rule (normalised) -> nothing new to insert
    all_existing = [r.lower() for rules in seed_mod.SEED_RULES.values() for r in rules]
    with patch.object(seed_mod.vs, "get_all_rule_contents", return_value=all_existing), \
         patch.object(seed_mod.vs, "add_rule") as add_rule:
        added = seed_mod.seed()

    assert added == 0
    add_rule.assert_not_called()


def test_seed_inserts_only_new_on_rerun():
    # Simulate a re-run after a new rule was added to ONE category: that category's DB
    # already has its OLD rules; the new one is missing -> exactly one insert.
    cat = next(iter(seed_mod.SEED_RULES))           # first category
    new_rule = "BRAND-NEW: a rule not yet stored anywhere."
    seed_mod.SEED_RULES[cat].append(new_rule)
    try:
        def _existing(category):
            # return everything EXCEPT the brand-new rule
            return [r for r in seed_mod.SEED_RULES[category] if r != new_rule]

        with patch.object(seed_mod.vs, "get_all_rule_contents", side_effect=_existing), \
             patch.object(seed_mod.vs, "add_rule") as add_rule:
            added = seed_mod.seed()

        assert added == 1
        assert add_rule.call_count == 1
        assert add_rule.call_args.args[1] == new_rule
    finally:
        seed_mod.SEED_RULES[cat].remove(new_rule)   # don't leak into other tests
