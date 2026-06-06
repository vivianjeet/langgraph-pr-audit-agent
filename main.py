
import argparse
import sys
import uuid
from src.llm_retry import QuotaExhaustedError
from src.state import Severity
import asyncio

async def run_audit(diff_text: str, large: bool = False, ci: bool = False, app=None) -> bool:
    """Runs the end to end graph execution on the provided diff.

    Local (ci=False): if it pauses for human review, PROMPT for a decision and resume.
    CI (ci=True): there is no stdin, so DON'T prompt - if it paused before human_review, report
    and return escalated=True so the caller can gate the build (exit code); never resume here.

    `app`: the compiled graph to drive. Defaults to the in-RAM (MemorySaver) app; the --durable
    path passes a SQLite-backed one (see _run_with_app). Threading it in keeps run_audit agnostic
    to which checkpointer backs the threads.

    Returns True if the audit escalated to human review (CI path), else False.
    """
    if app is None:
        from src.graph import app          # default: the module-level in-RAM app

    print("Starting LangGraph PR Audit Agent... \n")
    initial_state = {
        "audit": {"messages": [diff_text]},
        "force_compress": large,            # read by compress_node inside the graph
    }
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Running graph...\n")

    async def _drain(stream):
       async  for event in stream:
            for node_name, node_state in event.items():
                print(f"--- Node Executed: {node_name} ---")
                # finalize's message is the full report; we print final_report once at the
                # end, so skip it here to avoid printing the whole report (with its
                # "Learned N rules" note) twice.
                if node_name == "finalize":
                    print()
                    continue
                # Node outputs are AMSState-shaped: the audit substate carries messages.
                audit = (node_state or {}).get("audit") or {}
                if audit.get("messages"):
                    print(audit["messages"][-1])
                print()
    try:
        # First pass - runs untill END or untill it pauses before human_review
        await _drain(app.astream(initial_state, config=config))

        # If the graph paused, .next will contain the node we're interrupted before.
        snapshot = await app.aget_state(config)
        if snapshot.next and "human_review" in snapshot.next:
            v = snapshot.values.get("audit", {})
            print(">>> Graph PAUSED for human review (critical findings / low score).")
            print(f">>> All scores = {v.get('security_score')}    "
                    f"quality_score = {v.get('quality_score')}    "
                    f"test_score = {v.get('test_score')}")
            # Findings may come back as Pydantic objects OR (if a checkpoint round-trip degraded
            # them) plain dicts - read either shape so the resume path can't crash on .severity.
            def _fld(f, name, default=None):
                return f.get(name, default) if isinstance(f, dict) else getattr(f, name, default)
            criticals = [f for bucket in ("security_findings", "quality_findings", "test_findings")
                         for f in v.get(bucket, [])
                         if str(_fld(f, "severity")) in (Severity.CRITICAL, Severity.CRITICAL.value)]
            for f in criticals:
                print(f">>>    CRITICAL {_fld(f,'file_path')}:{_fld(f,'line_number','?')} - {_fld(f,'description')}")

            if ci:
                # No stdin in CI - don't prompt, don't resume. Report and hand the gate decision
                # back to the caller (it checks the PR's approval state and sets the exit code).
                print(">>> CI: escalated to human review. Caller will gate the build.")
                return True

            decision = input(">>> Enter decision [approve/reject/needs-changes]: ").strip() or "approve"

            # Inject the decision into state, then resume by streaming with None input.
            # Goes through the audit channel so merge_audit applies it to the substate.
            await app.aupdate_state(config, {"audit": {"human_decision": decision}})
            print("\n>>> Resuming after human decision...")
            await _drain(app.astream(None, config=config))

        final_audit = (await app.aget_state(config)).values.get("audit", {})
        print(final_audit.get("final_report", "(no report)"))
        return False   # ran to completion without (unresolved) escalation

    except QuotaExhaustedError as e:
        print(f"\n[ABORTED] {e}")
        print("  All keys exhausted. Re-run after the daily reset (midnight PT) or enable API billing.")
        return False   # NO report emitted, by design


async def run_gate(large: bool, durable: bool = False):
    """The --large pre-merge gate. Audits the REAL diff (changes vs the branch being merged into)
    and forces compression. Two surfaces:
      - LOCAL: ask the base branch, verify it merges cleanly (abort on conflict), audit, report.
      - CI: base from env, no prompt; if the audit escalates to human review, pass only when a
        human has already Approved the PR (GitHub holds that state) - else exit 1 to block merge.
    """
    from scripts.git_gate import (
        in_ci, resolve_base, merge_is_clean, resolve_diff, pr_is_human_approved,
    )
    ci = in_ci()
    base = resolve_base(ci)

    if not ci:
        ok, why = merge_is_clean(base, ci)
        print(f">>> Merge-compatibility vs '{base}': {why}")
        if not ok:
            print(">>> Resolve conflicts before auditing. Aborting.")
            sys.exit(1)

    diff = resolve_diff(demo=False, base=base)
    if durable:
        from src.graph import durable_app
        async with durable_app() as app:
            escalated = await run_audit(diff, large=large, ci=ci, app=app)
    else:
        escalated = await run_audit(diff, large=large, ci=ci)

    if ci:
        if escalated and not pr_is_human_approved():
            print("::error::Audit escalated to human review and no APPROVED review found. Blocking merge.")
            sys.exit(1)
        sys.exit(0)


async def _run_with_app(diff_text: str, large: bool, durable: bool) -> bool:
    """Pick the checkpointer backend, then run one audit. --durable -> a SQLite-backed app whose
    threads survive a process restart; otherwise the default in-RAM app."""
    if durable:
        from src.graph import durable_app
        async with durable_app() as app:
            return await run_audit(diff_text, large=large, app=app)
    return await run_audit(diff_text, large=large)


def main():
    parser = argparse.ArgumentParser(description="LangGraph PR Audit Agent")
    parser.add_argument("--test", action="store_true", help="Run the end-to-end smoke test")
    parser.add_argument("--demo", action="store_true", help=("Run the demo - "
            "Interactive human-in-the-loop audit on the SQL-injection diff"))
    parser.add_argument("--large", action="store_true",
                    help=("Force the context-compression pass over the session "
                          "messages even when below the 80%% budget threshold. Without it, "
                          "compression still AUTO-fires if a session genuinely hits 80%%."))
    parser.add_argument("--durable", action="store_true",
                    help=("Persist graph threads to a SQLite file (checkpoints.sqlite) so a run "
                          "paused for human review can resume across process restarts. Default is "
                          "in-RAM threads that don't survive the process."))
    args = parser.parse_args()

    if args.test:
        # import smoke tests function and run it
        from tests.smoke_test import run_smoke_test
        run_smoke_test()
    elif args.demo:
        from tests.test_integration import REAL_DIFF
        asyncio.run(_run_with_app(REAL_DIFF, args.large, args.durable))
    elif args.large:
        # --large = the real pre-merge gate: audit the real diff, force compression, and in CI
        # gate the build on findings vs the PR's human-approval state.
        asyncio.run(run_gate(large=True, durable=args.durable))
    else:
        # No flags = normal audit on the real diff via the same gate, compression NOT forced
        # (auto-fires only if the session hits 80% of budget).
        asyncio.run(run_gate(large=False, durable=args.durable))

if __name__ == "__main__":
    main()