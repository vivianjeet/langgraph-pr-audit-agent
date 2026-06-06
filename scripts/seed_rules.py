"""Seed baseline (human-authored) procedural rules so the procedural-memory store has org
policy active from a fresh DB. These are SEEDED rules (org policy), so they are active
immediately - distinct from the learned_pending rules the agent proposes during audits.

IDEMPOTENT: safe to re-run. It dedups against rules ALREADY stored for each category
(any status, via get_all_rule_contents), so adding new entries to SEED_RULES below and
re-running inserts only the new ones - existing rows are left untouched.

Run:  python -m scripts.seed_rules
"""
from src.state import RuleCategory, RuleStatus
from src.db import vectorstore as vs

# Baseline org policy, grouped by domain. To seed more later: add lines here and re-run.
SEED_RULES: dict[RuleCategory, list[str]] = {
    RuleCategory.SECURITY: [
        "Never build SQL with f-strings or string concatenation; use parameterised queries.",
        "No secrets, API keys or credentials committed in code or config.",
        "Validate and sanitise all external input before use.",
    ],
    RuleCategory.QUALITY: [
        "No function longer than ~50 lines; extract helpers instead.",
        "Public functions must have type hints on parameters and return value.",
    ],
    RuleCategory.COVERAGE: [
        "Every new public function or endpoint must ship with at least one test.",
    ],
}


def seed() -> int:
    """Insert any SEED_RULES not already present. Returns the count actually added."""
    added = 0
    for category, rules in SEED_RULES.items():
        try:
            existing = {r.strip().lower() for r in vs.get_all_rule_contents(category)}
        except Exception:
            existing = set()
        for rule in rules:
            text = rule.strip()
            if not text or text.lower() in existing:
                continue
            vs.add_rule(category, text, status=RuleStatus.SEEDED)
            existing.add(text.lower())
            added += 1
    return added


if __name__ == "__main__":
    n = seed()
    print(f"Seeded {n} new rule(s)." if n else "No new rules to seed (all already present).")
