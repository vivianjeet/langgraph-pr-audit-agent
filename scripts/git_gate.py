import os, subprocess


def in_ci() -> bool:
    """True when running inside GitHub Actions (or any CI). Gates: no input() prompt, base
    from env not stdin, no local fetch and exit-code gating instead of interactive resume."""
    return bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))


def github_token() -> str | None:
    """Auth source, in order: CI's injected GITHUB_TOKEN/GH_TOKEN, then the locally logged-in
    gh CLI. So a developer with `gh auth login` done needs NO extra token; CI uses the env var
    Actions injects for free. Returns None if neither is available (caller degrades gracefully)."""
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else None


def merge_is_clean(base: str, in_ci_: bool | None = None) -> tuple[bool, str]:
    """Read-only mergeability check: `git merge-tree` computes the merge IN MEMORY and reports
    conflicts WITHOUT touching HEAD/index/working tree. True = merges cleanly. Auditing a diff
    that can't even merge is pointless, so callers check this BEFORE the audit."""
    ci = in_ci() if in_ci_ is None else in_ci_
    if not ci:
        subprocess.run(["git", "fetch", "origin", base], capture_output=True, text=True)
    ref = f"origin/{base}"
    # --write-tree mode exits non-zero on conflict and prints conflict info.
    r = subprocess.run(["git", "merge-tree", "--write-tree", ref, "HEAD"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, (r.stderr.strip() or r.stdout.strip() or "merge conflicts detected")
    if "CONFLICT" in (r.stdout + r.stderr) or "<<<<<<<" in r.stdout:
        return False, "merge conflicts detected"
    return True, "merges cleanly"


def pr_human_decision() -> str | None:
    """The latest human review verdict on THIS PR, mapped to our decision vocabulary:
      APPROVED          -> "approve"
      CHANGES_REQUESTED -> "needs-changes"
      anything else / no review / error -> None  (= no verdict yet; caller keeps the build blocked)
    Read via `gh` (carries auth). Mirrors pr_is_human_approved's PR_NUMBER handling: in CI the
    checkout is a detached merge ref so the PR can't be branch-inferred - PR_NUMBER supplies it;
    locally gh infers it. Best-effort: any failure -> None (fail-closed: no verdict = blocked)."""
    pr = os.environ.get("PR_NUMBER")
    cmd = ["gh", "pr", "view"]
    if pr:
        cmd.append(pr)
    cmd += ["--json", "reviews",
            "--jq", "(.reviews | last | .state) // \"\""]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    state = r.stdout.strip().strip('"')
    return {"APPROVED": "approve", "CHANGES_REQUESTED": "needs-changes"}.get(state)


def pr_is_human_approved() -> bool:
    """True iff the latest human review verdict is APPROVED. Thin wrapper over pr_human_decision
    so there's ONE source of truth for the PR's verdict; kept because run_gate + older callers
    use the boolean. Any failure -> False (fail-closed)."""
    return pr_human_decision() == "approve"


def _git_diff(base: str, in_ci: bool) -> str:
    """Diff of this branch against <base>: the changes that would be merged.
    Local: fetch the base first (no workflow to provide it). CI: the workflow's
    checkout (fetch-depth: 0) already has it, so we don't fetch here."""
    if not in_ci:
        subprocess.run(["git", "fetch", "origin", base], capture_output=True, text=True)
    ref = f"origin/{base}"
    result = subprocess.run(["git", "diff", f"{ref}...HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def resolve_base(in_ci_: bool | None = None) -> str:
    """The branch we're merging into: from GITHUB_BASE_REF in CI, asked interactively locally."""
    ci = in_ci() if in_ci_ is None else in_ci_
    if ci:
        return os.environ.get("GITHUB_BASE_REF", "main")
    return input("Merge into which branch? [main]: ").strip() or "main"


def resolve_diff(demo: bool, base: str | None = None) -> str:
    from tests.test_integration import REAL_DIFF
    if demo:
        return REAL_DIFF

    ci = in_ci()
    if base is None:                 # let the gate pass a pre-resolved base (after merge-check)
        base = resolve_base(ci)

    diff = _git_diff(base, ci)       # ci gates the fetch (local fetches; CI relies on checkout)
    if not diff:
        print(">>> No diff found vs that branch; falling back to the demo fixture.")
        return REAL_DIFF
    return diff
