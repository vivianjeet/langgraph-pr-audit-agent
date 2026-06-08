"""Tests in this file: the extended-thinking gate (src/complexity.py thinking_warranted).

WHAT IS PINNED
- The deterministic heuristic that decides whether an audit pays the thinking tax:
  * >= 2 compliance frameworks -> True (cross-regulation interplay is genuinely hard)
  * a LARGE diff (>= 40 changed lines) AND at least one framework -> True
  * a single-framework small diff -> False
  * an UNREGULATED diff (no framework) -> always False, regardless of size

WHY no LLM / no DB
- thinking_warranted is pure string + set arithmetic on already-fetched inputs, so it's
  tested directly. The size signal counts the ingest markers [ADDED]/[REMOVED] (NOT git @@
  hunks - the parsed diff has none), so the fixtures use those markers.
"""
from src.complexity import thinking_warranted


def _diff(changed_lines: int) -> str:
    # one [ADDED] marker per changed line - matches what ingest.parse_github_diff emits.
    return "\n".join("[ADDED] x = 1" for _ in range(changed_lines))


def _ctx(*frameworks):
    return [{"framework": f, "text": "...", "source": "doc"} for f in frameworks]


def test_two_frameworks_warrants_thinking():
    # cross-regulation interplay is hard even on a tiny diff.
    assert thinking_warranted(_diff(1), _ctx("gdpr", "hipaa")) is True


def test_large_single_framework_warrants_thinking():
    assert thinking_warranted(_diff(40), _ctx("gdpr")) is True


def test_small_single_framework_does_not_warrant():
    assert thinking_warranted(_diff(10), _ctx("gdpr")) is False


def test_unregulated_diff_never_warrants_even_when_large():
    # no framework -> the size signal can't fire (bool(frameworks) is False); cheap path.
    assert thinking_warranted(_diff(500), []) is False


def test_size_threshold_is_inclusive_at_40():
    # exactly 40 changed lines + a framework is the boundary - must trip.
    assert thinking_warranted(_diff(40), _ctx("pci_dss")) is True
    assert thinking_warranted(_diff(39), _ctx("pci_dss")) is False
