from src.memory import AgentMemorySystem as AMS, AMSState
from src.llm_client import score_audit
import src.config as cfg


def _weighted_score(findings) -> float:
    """1.0 = clean. Each finding MULTIPLICATIVELY erodes the remaining score, so several moderate
    findings approach 0 with diminishing returns instead of summing past it - SEVERITY, not raw
    COUNT, drives the risk. (A linear sum let 4 HIGH findings hit exactly 0.0, scoring a messy
    class the same as a catastrophe; multiplicative gives 4 HIGH -> 0.24 - low, still escalates,
    but distinguishable.) A single CRITICAL still bites hard (0.4). Deterministic, no LLM, one
    function for all 3 dimensions.
    """
    if not findings:
        return 1.0
    score = 1.0
    for f in findings:
        score *= (1.0 - cfg.SEVERITY_PENALTY.get(f.severity, 0.0))
    return round(score, 2)

def synthesize_report_node(state: AMSState):
    """ Aggregate the three audits into a single risk score the router can act on"""
    ams = AMS(state)
    sec = ams.read("security_findings",[])
    qual = ams.read("quality_findings",[])
    test = ams.read("test_findings",[])

    sec_score = _weighted_score(sec)
    qual_score = _weighted_score(qual)
    test_score = _weighted_score(test)

    # FAIL CLOSED: if an AUDIT node errored out (not just plan), it returned empty
    # findings -> a 1.0 here would be a FALSE clean. Force max risk so routing escalates.
    errors = ams.read("node_errors", [])
    audit_errors = [e for e in errors if e.split(":")[0] in
                    ("security_audit", "quality_audit", "coverage_audit")]
    if audit_errors:
        sec_score = qual_score = test_score = 0.0
    
    summary = (
        f"System: Synthesized report. \n"
        f"security_score = {sec_score} \n"
        f"quality_score = {qual_score} \n"
        f"test_score = {test_score},"
        f"(security={len(sec)}, quality={len(qual)}, test={len(test)} findings)"
    )

    if audit_errors:
        summary += f"\n⚠ FAIL-CLOSED: audit node(s) errored, scores forced to 0.0: {audit_errors}"

    # Attach the three dimension scores to the run's Langfuse trace (no-op if unconfigured).
    # Sent separately, not combined - the system escalates on the worst axis, it never aggregates.
    score_audit({"security_score": sec_score,
                 "quality_score": qual_score,
                 "test_score": test_score})

    return {"audit": {
        "messages" : [summary],
        "security_score": sec_score,
        "quality_score": qual_score,
        "test_score": test_score,
    }}