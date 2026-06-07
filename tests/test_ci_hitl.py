"""WHAT THIS WHOLE FILE TESTS
=============================
The CI branch of `main.run_audit` - the path with NO unit coverage until now and the riskiest
piece of the HITL/CLI=GitHub-parity work. In CI there is no stdin, so the human's verdict comes
from the GitHub PR via `pr_human_decision()`. This file pins that:

  - no verdict yet (None)      -> stay BLOCKED: return True, DO NOT resume the graph.
  - verdict "approve"          -> resume through finalize, return False (not blocked).
  - verdict "needs-changes"    -> resume through finalize, return True (blocked).
  - verdict "reject"           -> resume through finalize, return True (blocked).

The contract under test is the return value (which run_gate turns into the CI exit code) AND
whether the graph was RESUMED (a second astream call) only when a verdict exists. A fake app
stands in for the compiled graph so there is no DB / LLM / real checkpointer: it reports a pause
before `human_review`, records aupdate_state + the resume astream, and yields a final report.
`pr_human_decision` is patched per-case.
"""
import asyncio
from unittest.mock import patch, MagicMock
import main


class _FakeApp:
    """Minimal stand-in for the compiled graph driving run_audit's CI path.
    - astream(): async-iterates nothing (we don't assert on streamed events here), but COUNTS
      calls so we can tell whether the graph was resumed (2nd astream) after the verdict.
    - aget_state(): first call reports a pause before human_review; after a resume it reports
      done (next=()) and carries a final_report so run_audit can print it."""
    def __init__(self):
        self.astream_calls = 0
        self.updated_decision = None
        self._resumed = False

    def astream(self, _input, config=None):
        self.astream_calls += 1
        if self.astream_calls >= 2:
            self._resumed = True

        async def _gen():
            if False:
                yield {}            # an empty async iterator
        return _gen()

    async def aget_state(self, config=None):
        if not self._resumed:
            # paused before human_review, with minimal scores/findings for the print block.
            return MagicMock(next=("human_review",),
                             values={"audit": {"security_score": 0.4, "quality_score": 1.0,
                                               "test_score": 1.0, "security_findings": [],
                                               "quality_findings": [], "test_findings": []}})
        return MagicMock(next=(), values={"audit": {"final_report": "REPORT"}})

    async def aupdate_state(self, config, update):
        self.updated_decision = update["audit"]["human_decision"]


def _run(verdict):
    """Drive run_audit in CI mode with pr_human_decision patched to `verdict`. Returns
    (escalated, fake_app) so a test can assert the return value AND whether a resume happened."""
    app = _FakeApp()
    with patch("scripts.git_gate.pr_human_decision", return_value=verdict):
        escalated = asyncio.run(main.run_audit("diff", ci=True, app=app))
    return escalated, app


def test_ci_no_verdict_blocks_and_does_not_resume():
    # No human review on the PR yet -> blocked (True) and the graph is NOT resumed.
    escalated, app = _run(None)
    assert escalated is True
    assert app.astream_calls == 1          # only the first pass; no resume
    assert app.updated_decision is None


def test_ci_approve_resumes_and_passes():
    # GitHub APPROVED -> resume through finalize, NOT blocked (False).
    escalated, app = _run("approve")
    assert escalated is False
    assert app.astream_calls == 2          # resumed
    assert app.updated_decision == "approve"


def test_ci_needs_changes_resumes_and_blocks():
    # CHANGES_REQUESTED -> resume (report still written) but blocked (True).
    escalated, app = _run("needs-changes")
    assert escalated is True
    assert app.astream_calls == 2
    assert app.updated_decision == "needs-changes"


def test_ci_reject_resumes_and_blocks():
    # reject -> resume, blocked (True). Same control flow as needs-changes, different label.
    escalated, app = _run("reject")
    assert escalated is True
    assert app.astream_calls == 2
    assert app.updated_decision == "reject"
