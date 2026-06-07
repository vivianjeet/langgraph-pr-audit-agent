"""Tests in this file: finalize_report_node's episodic-record choice.

WHAT IS PINNED
- When the `compressed` channel is non-empty (compress_node fired this run), finalize stores
  the COMPACTED HISTORY as the episode (proto-episodic loop).
- When `compressed` is empty/absent (compression was a pass-through), finalize falls back to
  its structured findings/scores summary.

HOW IT WORKS (no DB, no LLM)
- The three persistence calls finalize makes (store_pr_audit, store_episode,
  learn_rules_from_findings) are patched on the AMS class so nothing hits pgvector. We capture
  the `episode` string passed to store_episode and assert which branch produced it.
- `compressed` is a TOP-LEVEL AMSState channel, so it's set as a sibling of `audit` in the
  state dict (not inside the audit substate) - mirroring how compress_node writes it.
"""
from unittest.mock import patch
import src.nodes.finalize as fin_mod

def _state_with_decision(decision):
    s = _state()                       # the existing helper (human_decision="approve" by default)
    s["audit"]["human_decision"] = decision
    return s

def _state(compressed=None):
    """Minimal AMSState for finalize: empty findings/scores in the audit substate, plus the
    top-level `compressed` channel under test."""
    s = {
        "audit": {
            "security_findings": [], "quality_findings": [], "test_findings": [],
            "security_score": 1.0, "quality_score": 1.0, "test_score": 1.0,
            "human_decision": "approve", "iteration_count": 0,
            "files_changed": ["src/app.py"], "parsed_diff": "diff",
        },
    }
    if compressed is not None:
        s["compressed"] = compressed
    return s


def _run_capturing_episode(state):
    captured = {}

    def _capture_episode(summary, metadata=None):
        captured["episode"] = summary

    with patch.object(fin_mod.AMS, "store_episode", side_effect=_capture_episode), \
         patch.object(fin_mod.AMS, "store_pr_audit", return_value=None), \
         patch.object(fin_mod.AMS, "learn_rules_from_findings", return_value=0):
        fin_mod.finalize_report_node(state)
    return captured["episode"]


def test_finalize_uses_compressed_history_as_episode_when_present():
    compressed = ["System: [compressed 4 earlier messages] decision=reject; SQLi in login.py",
                  "System: Final report ready."]
    episode = _run_capturing_episode(_state(compressed=compressed))
    assert episode.startswith("Compressed session history:")
    assert "SQLi in login.py" in episode                 # the compacted content is the episode


def test_finalize_falls_back_to_summary_when_no_compressed():
    episode = _run_capturing_episode(_state(compressed=[]))   # empty -> pass-through happened
    assert not episode.startswith("Compressed session history:")
    assert "Files changed:" in episode                   # the structured-summary fallback
    assert "src/app.py" in episode

def _run_capturing_summary(state):
    """Capture the precedent summary store_pr_audit receives, and return (summary, markdown)."""
    captured = {}

    def _capture_audit(summary, report, embed_text=None):
        captured["summary"] = summary
        captured["status"] = report.get("status")

    with patch.object(fin_mod.AMS, "store_episode", return_value=None), \
         patch.object(fin_mod.AMS, "store_pr_audit", side_effect=_capture_audit), \
         patch.object(fin_mod.AMS, "learn_rules_from_findings", return_value=0):
        out = fin_mod.finalize_report_node(state)
    return captured, out["audit"]["final_report"]


def test_finalize_status_per_verdict():
    # Each verdict -> the right report status; "n/a"/never-escalated defaults to passed.
    cases = {
        "approve": "passed",
        "needs-changes": "changes-required",
        "reject": "rejected",
        "n/a": "passed",                # clean PR, never escalated -> NOT a block
    }
    for decision, expected in cases.items():
        captured, markdown = _run_capturing_summary(_state_with_decision(decision))
        assert captured["status"] == expected
        assert f"**Status:** {expected}" in markdown


def test_finalize_summary_records_verdict():
    # The precedent summary (what a future PR retrieves) must carry the verdict, so a deferral
    # is legible later ("this area was sent back for changes before").
    captured, _ = _run_capturing_summary(_state_with_decision("needs-changes"))
    assert "verdict: needs-changes" in captured["summary"]
    assert "status: changes-required" in captured["summary"]