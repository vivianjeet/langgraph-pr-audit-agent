# decide whether a PR's audit warrants extended-thinking (the expensive tier).
# Cheap heuristic FIRST (no LLM): a PR is "complex" when it spans multiple regulated concerns or is
# large. Only the complex slice pays the thinking tax. Deterministic so it's testable + auditable.
def thinking_warranted(parsed_diff: str, compliance_context: list) -> bool:
    """True when the audit should use extended thinking. Signals:
      - the compliance lookup hit MORE THAN ONE framework (multi-regulation interplay), OR
      - the diff is large (many changed lines) AND touched at least one regulated framework.
    A single-framework or unregulated diff never needs thinking - that's the 85% cheap path."""
    frameworks = {c.get("framework") for c in compliance_context if c.get("framework")}
    if len(frameworks) >= 2:
        return True                                  # cross-regulation analysis = genuinely hard
    # SIZE SIGNAL - count what `parsed_diff` ACTUALLY contains. ingest.parse_github_diff strips raw
    # git away and emits only "[ADDED]"/"[REMOVED]"/"[FILE MODIFIED]" lines - there are NO "@@" hunk
    # markers left, so `parsed_diff.count("@@")` would be 0 forever and this branch would be DEAD.
    # Count the real markers instead: changed lines (ADDED+REMOVED) is the size proxy here.
    changed_lines = parsed_diff.count("[ADDED]") + parsed_diff.count("[REMOVED]")
    return bool(frameworks) and changed_lines >= 40  # big regulated change