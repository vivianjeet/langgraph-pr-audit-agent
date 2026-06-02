
import argparse
import uuid
from src.graph import app
from src.llm_retry import QuotaExhaustedError
from src.state import Severity

def run_audit(diff_text: str):
    """Runs the end to end graph execution on the provided diff
    If it pauses for human review, prompt for a decision and resume.
    """
    
    print("Starting LangGraph PR Audit Agent... \n")
    initial_state = {
        "audit": {"messages": [diff_text]}
    }
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print("Running graph...\n")

    def _drain(stream):
        for event in stream:
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
        _drain(app.stream(initial_state, config=config))

        # If the graph paused, .next will contain the node we're interrupted before.
        snapshot = app.get_state(config)
        if snapshot.next and "human_review" in snapshot.next:
            v = snapshot.values.get("audit", {})
            print(">>> Graph PAUSED for human review (critical findings / low score).")
            print(f">>> All scores = {v.get('security_score')}    "
                    f"quality_score = {v.get('quality_score')}    "
                    f"test_score = {v.get('test_score')}")
            criticals = [f for bucket in ("security_findings", "quality_findings", "test_findings")
                         for f in v.get(bucket, []) if f.severity == Severity.CRITICAL]
            for f in criticals:
                print(f">>>    CRITICAL {f.file_path}:{getattr(f, 'line_number', '?')} - {f.description}")
            decision = input(">>> Enter decision [approve/reject/needs-changes]: ").strip() or "approve"

            # Inject the decision into state, then resume by streaming with None input.
            # Goes through the audit channel so merge_audit applies it to the substate.
            app.update_state(config, {"audit": {"human_decision": decision}})
            print("\n>>> Resuming after human decision...")
            _drain(app.stream(None, config=config))

        final_audit = app.get_state(config).values.get("audit", {})
        print(final_audit.get("final_report", "(no report)"))
    
    except QuotaExhaustedError as e:
        print(f"\n[ABORTED] {e}")
        print("  All keys exhausted. Re-run after the daily reset (midnight PT) or enable API billing.")
        return   # NO report emitted, by design
        

def main():
    parser = argparse.ArgumentParser(description="LangGraph PR Audit Agent")
    parser.add_argument("--test", action="store_true", help="Run the end-to-end smoke test")
    parser.add_argument("--demo", action="store_true", help=("Run the demo - "
            "Interactive human-in-the-loop audit on the SQL-injection diff"))
    args = parser.parse_args()

    if args.test:
        # import smoke tests function and run it
        from tests.smoke_test import run_smoke_test
        run_smoke_test()
    elif args.demo:
        from tests.test_integration import REAL_DIFF
        run_audit(REAL_DIFF)
    else:
        print("LangGraph PR Audit Agent is ready.")
        print("Run 'python main.py --test' to execute the smoke test.")

if __name__ == "__main__":
    main()