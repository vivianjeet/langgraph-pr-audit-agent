# Custom LangSmith evaluators: score audit OUTPUT quality, beyond "did it run".
# Run via langsmith.evaluate() against a dataset of diffs (see __main__).
from langsmith.evaluation import evaluate

from src.state import Severity
from src.config import LOW_SCORE_THRESHOLD

def every_finding_has_cwe(run, example) -> dict:
    """Security traceability: every security finding MUST carry a CWE id."""
    findings = (run.outputs or {}).get("security_findings", [])
    ok = all(getattr(f, "cwe_id", None) for f in findings)
    return {"key": "cwe_traceability", "score": 1.0 if ok else 0.0}

def score_consistent_with_findings(run, example) -> dict:
    """A high security_score with a CRITICAL finding present is a contradiction."""
    out = run.outputs or {}
    score = out.get("security_score", 1.0)
    has_critical = any(
        getattr(f, "severity", None) == Severity.CRITICAL
        for f in out.get("security_findings", [])
    )
    consistent = not (has_critical and score >= LOW_SCORE_THRESHOLD)
    return {"key": "score_consistency", "score" : 1.0 if consistent else 0.0}

if __name__ == "__main__":
    from src.graph import app

    def target(inputs: dict):
        import uuid
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        for _ in app.stream({"messages" : [inputs["diff"]]},config=cfg):
            pass
        return app.get_state(cfg).values
    
    evaluate(
        target,
        data="pr-audit-eval-set",
        evaluators=[every_finding_has_cwe, score_consistent_with_findings],
        experiment_prefix="pr-audit"
    )
    print("Evaluation submitted to LangSmith.")