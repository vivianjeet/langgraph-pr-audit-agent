"""Offline rule-governance CLI: the human control plane over procedural rules.

Two phases:
  1. Review PENDING rules (proposed by audits, status=learned_pending) -> approve / reject / skip.
     For each pending rule it shows a near-duplicate HINT (cosine over the stored embedding) so a
     human can collapse reworded re-learns that exact-text dedup can't catch.
  2. Manage ACTIVE rules (seeded + learned_approved) -> retire (soft, un-learn-safe) or delete (hard).

This is OUT-OF-BAND store maintenance, deliberately NOT part of any audit run: learning happens in
finalize (after the in-graph human_review pause), and rules from clean runs never reach that pause,
so governance needs its own tool. (See the design note in the Day-21 plan / README.)

Run:  python -m scripts.review_rules
"""
import sys
import textwrap
from src.memory import AgentMemorySystem as AMS
from src.state import RuleStatus

# ANSI palette (same raw-escape convention as src/db/vectorstore.py - no extra dependency).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[96m"      # the rule being decided / headers
_BLUE = "\033[94m"      # seeded (human-authored baseline)
_YELLOW = "\033[93m"    # pending / duplicate warnings
_GREEN = "\033[92m"     # approved
_RED = "\033[91m"       # rejected / delete warning
_GREY = "\033[90m"      # metadata (ids), the "Learned" origin word


def _wrap(text: str, indent: str) -> str:
    """Wrap long rule text, every line sharing the given indent. No content lost."""
    return textwrap.fill(text, width=184, initial_indent=indent, subsequent_indent=indent)


# Map each RuleStatus to a colored, human-readable label that separates ORIGIN (Learned) from
# STATE (Pending/Approved/Rejected/Retired) so the lifecycle is scannable at a glance.
_STATUS_LABELS = {
    RuleStatus.SEEDED.value:           f"{_BLUE}Seeded{_RESET}",
    RuleStatus.LEARNED_PENDING.value:  f"{_GREY}Learned{_RESET} - {_YELLOW}Pending{_RESET}",
    RuleStatus.LEARNED_APPROVED.value: f"{_GREY}Learned{_RESET} - {_GREEN}Approved{_RESET}",
    RuleStatus.REJECTED.value:         f"{_GREY}Learned{_RESET} - {_RED}Rejected{_RESET}",
    RuleStatus.RETIRED.value:          f"{_DIM}Retired{_RESET}",
}


def _fmt_status(status: str) -> str:
    """Colored lifecycle label for a rule status; falls back to the raw value if unknown."""
    return _STATUS_LABELS.get(status, status)


# Color the PR verdict this rule was learned from, by sentiment, so a reviewer weighs the
# provenance: a rule learned from a REJECTED PR is a yellow flag (the human distrusted that audit).
_DECISION_COLORS = {
    "approve": _GREEN,
    "needs-changes": _YELLOW,
    "reject": _RED,
}


def _fmt_decision(decision) -> str | None:
    """Colored 'learned from a PR the human <verdict>' line. None when there was no human
    review (n/a / empty) - nothing to show, so the caller skips the line entirely."""
    if not decision or str(decision).lower() in ("n/a", "none"):
        return None
    d = str(decision).lower()
    color = _DECISION_COLORS.get(d, _GREY)
    return f"{_DIM}learned from a PR the human marked:{_RESET} {color}{_BOLD}{d}{_RESET}"


def _review_pending() -> None:
    pending = AMS.pending_rules()
    if not pending:
        print("No pending rules to review.\n")
        return
    print(f"{_BOLD}{len(pending)} pending rule(s).{_RESET} "
          f"For each: [{_GREEN}a{_RESET}]pprove / [{_RED}r{_RESET}]eject / [s]kip.\n")
    for rule in pending:
        print(f"{_CYAN}{_BOLD}  +-- #{rule['id']} {rule['category'].upper()}{_RESET}")
        decision = _fmt_decision(rule.get("source_decision"))
        if decision:
            print(f"  |   {decision}")
        print(_wrap(rule["content"], "  |   "))
        # Similarity hint, ADVISORY ONLY: cosine similarity is a rough proxy for "duplicate",
        # not proof - it can miss reworded dupes (false negative) or flag distinct-but-related
        # rules (false positive). The threshold is untuned (validating it needs a labeled
        # eval set = Repo 2 / RAGAS). So we show the score and let the human judge; the tool
        # never acts on it. See AMS.similar_rules.
        sims = AMS.similar_rules(rule["id"])
        if sims:
            print(f"  |   {_YELLOW}similar existing rule(s) - check if duplicate "
                  f"(you decide):{_RESET}")
            for sim in sims:
                pct = f"{sim['similarity']:.0%}"
                print(f"  |     {_YELLOW}{pct:>4} similar{_RESET} {_GREY}#{sim['id']}{_RESET} "
                      f"{_fmt_status(sim['status'])}")
                print(_wrap(sim["content"], "  |       "))
        print(f"{_CYAN}  +--{_RESET}")
        choice = input(f"     {_BOLD}a/r/s?{_RESET} ").strip().lower()
        if choice == "a":
            AMS.approve_rule(rule["id"]); print(f"     {_GREEN}-> approved (now active).{_RESET}\n")
        elif choice == "r":
            AMS.reject_rule(rule["id"]); print(f"     {_RED}-> rejected (kept, will not re-learn).{_RESET}\n")
        else:
            print(f"     {_DIM}-> skipped.{_RESET}\n")


def _manage_active() -> None:
    active = AMS.active_rules()
    if not active:
        print("No active rules.")
        return
    print(f"{_BOLD}{len(active)} active rule(s).{_RESET} "
          f"Actions: {_BOLD}retire <id>{_RESET} / {_BOLD}delete <id>{_RESET} / {_BOLD}done{_RESET}.\n")
    for r in active:
        print(f"{_CYAN}{_BOLD}  #{r['id']}{_RESET} {_fmt_status(r['status'])} "
              f"{_CYAN}{r['category']}{_RESET}")
        print(_wrap(r["content"], "      "))
    while True:
        cmd = input("\n  action (retire <id> / delete <id> / done): ").strip().lower().split()
        if not cmd or cmd[0] == "done":
            return
        if len(cmd) != 2 or not cmd[1].isdigit():
            print("    usage: retire <id> | delete <id> | done")
            continue
        rid = int(cmd[1])
        if cmd[0] == "retire":
            AMS.retire_rule(rid); print(f"    {_GREEN}-> rule {rid} retired (deactivated, kept).{_RESET}")
        elif cmd[0] == "delete":
            row = next((x for x in active if x["id"] == rid), None)
            if row and str(row["status"]).startswith("learned"):
                if input(f"    {_RED}{_BOLD}WARNING:{_RESET}{_RED} deleting a LEARNED rule lets the same "
                         f"finding re-learn as pending next run. Retire is safer. Delete {rid} "
                         f"anyway? [y/N] {_RESET}").strip().lower() != "y":
                    print(f"    {_DIM}-> cancelled.{_RESET}")
                    continue
            AMS.delete_rule(rid); print(f"    {_RED}-> rule {rid} hard-deleted.{_RESET}")
        else:
            print("    unknown action.")


def main() -> None:
    if not sys.stdin.isatty():       # non-interactive guard, like the repo's other scripts
        print("review_rules.py needs an interactive terminal.")
        return
    _review_pending()
    _manage_active()


if __name__ == "__main__":
    main()