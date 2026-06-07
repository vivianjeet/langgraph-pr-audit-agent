"""Tests in this file: the --large pre-merge gate plumbing (scripts/git_gate.py).

WHAT IS PINNED
- in_ci: detects CI via env.
- github_token: prefers GITHUB_TOKEN env, falls back to `gh auth token`, else None.
- resolve_base: GITHUB_BASE_REF in CI; the interactive input() prompt locally.
- merge_is_clean: read-only `git merge-tree`; False on non-zero / conflict markers.
- pr_is_human_approved: parses the gh jq count; False on any failure.
- resolve_diff: uses origin/<base> diff; falls back to the fixture when empty.

HOW IT WORKS (no real git / gh / network)
- subprocess.run is patched with a fake that returns a CompletedProcess-like object keyed by
  the command, so each helper is exercised against controlled returncode/stdout. os.environ and
  builtins.input are patched per-test. Nothing touches a real repo.
"""
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import scripts.git_gate as gg


def _proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

def _gh(stdout, rc=0):
    return MagicMock(stdout=stdout, returncode=rc)

# ---- in_ci ----

def test_in_ci_true_when_github_actions_set():
    with patch.dict("os.environ", {"GITHUB_ACTIONS": "true"}, clear=True):
        assert gg.in_ci() is True

def test_in_ci_false_when_no_env():
    with patch.dict("os.environ", {}, clear=True):
        assert gg.in_ci() is False


# ---- github_token ----

def test_github_token_prefers_env():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "envtok"}, clear=True):
        assert gg.github_token() == "envtok"

def test_github_token_falls_back_to_gh_cli():
    with patch.dict("os.environ", {}, clear=True), \
         patch.object(gg.subprocess, "run", return_value=_proc(0, "ghtok\n")):
        assert gg.github_token() == "ghtok"

def test_github_token_none_when_neither():
    with patch.dict("os.environ", {}, clear=True), \
         patch.object(gg.subprocess, "run", return_value=_proc(1, "")):
        assert gg.github_token() is None


# ---- resolve_base ----

def test_resolve_base_from_env_in_ci():
    with patch.dict("os.environ", {"GITHUB_BASE_REF": "release"}, clear=True):
        assert gg.resolve_base(in_ci_=True) == "release"

def test_resolve_base_prompts_locally():
    with patch("builtins.input", return_value="develop"):
        assert gg.resolve_base(in_ci_=False) == "develop"

def test_resolve_base_defaults_to_main_on_empty_prompt():
    with patch("builtins.input", return_value=""):
        assert gg.resolve_base(in_ci_=False) == "main"


# ---- merge_is_clean ----

def test_merge_is_clean_true_on_zero_exit():
    with patch.object(gg.subprocess, "run", return_value=_proc(0, "treeoid")):
        ok, why = gg.merge_is_clean("main", in_ci_=True)
    assert ok and "clean" in why

def test_merge_is_clean_false_on_nonzero():
    with patch.object(gg.subprocess, "run", return_value=_proc(1, "", "boom")):
        ok, why = gg.merge_is_clean("main", in_ci_=True)
    assert not ok and ("boom" in why or "conflict" in why.lower())

def test_merge_is_clean_false_on_conflict_marker():
    with patch.object(gg.subprocess, "run", return_value=_proc(0, "CONFLICT (content)")):
        ok, why = gg.merge_is_clean("main", in_ci_=True)
    assert not ok


# ---- pr_is_human_approved ----

def test_pr_approved_true_when_latest_is_approved():
    # Now verdict-based: pr_is_human_approved delegates to pr_human_decision; the latest review
    # state APPROVED -> True. (Was count-based; pr_human_decision is the single source of truth.)
    with patch.object(gg.subprocess, "run", return_value=_proc(0, '"APPROVED"\n')):
        assert gg.pr_is_human_approved() is True

def test_pr_approved_false_when_changes_requested():
    with patch.object(gg.subprocess, "run", return_value=_proc(0, '"CHANGES_REQUESTED"\n')):
        assert gg.pr_is_human_approved() is False

def test_pr_approved_false_on_gh_failure():
    with patch.object(gg.subprocess, "run", return_value=_proc(1, "")):
        assert gg.pr_is_human_approved() is False


# ---- resolve_diff ----

def test_resolve_diff_returns_git_diff():
    with patch.dict("os.environ", {}, clear=True), \
         patch.object(gg.subprocess, "run", return_value=_proc(0, "diff --git a b\n+code")):
        out = gg.resolve_diff(demo=False, base="main")
    assert out.startswith("diff --git")

def test_resolve_diff_falls_back_to_fixture_when_empty():
    from tests.test_integration import REAL_DIFF
    with patch.dict("os.environ", {}, clear=True), \
         patch.object(gg.subprocess, "run", return_value=_proc(0, "")):
        out = gg.resolve_diff(demo=False, base="main")
    assert out == REAL_DIFF

def test_pr_human_decision_maps_states():
    cases = {'"APPROVED"': "approve", '"CHANGES_REQUESTED"': "needs-changes",
             '""': None, '"COMMENTED"': None}
    for gh_out, expected in cases.items():
        with patch.object(gg.subprocess, "run", return_value=_gh(gh_out)):
            assert gg.pr_human_decision() == expected

def test_pr_human_decision_gh_failure_is_none():
    with patch.object(gg.subprocess, "run", return_value=_gh("", rc=1)):
        assert gg.pr_human_decision() is None
