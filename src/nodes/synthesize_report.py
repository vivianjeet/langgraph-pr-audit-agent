from src.state import AuditState, Severity


_SEVERITY_PENALTY = {
    Severity.CRITICAL: 0.6,
    Severity.HIGH: 0.3,
    Severity.MEDIUM: 0.15,
    Severity.LOW: 0.07,
    Severity.INFO: 0.02,
    Severity.NONE: 0.0
}

def _weighted_score(findings) -> float:
    """1.0 = clean. Subtract severity-weighed penalties, clamp to [0, 1].
    Deterministic = testable, $0 (no LLM). One function for all 3 dimensions (DRY:
    collapses the three identical _compute_*_score helpers into one).
    """
    if not findings:
        return 1.0
    penalty = sum(_SEVERITY_PENALTY.get(f.severity,0.0) for f in findings)
    return max(0.0, round(1.0 - penalty, 2))

def synthesize_report_node(state: AuditState):
    """ Aggregate the three audits into a single risk score the router can act on"""
    sec = state.get("security_findings",[])
    qual = state.get("quality_findings",[])
    test = state.get("test_findings",[])

    sec_score = _weighted_score(sec)
    qual_score = _weighted_score(qual)
    test_score = _weighted_score(test)

    # FAIL CLOSED: if an AUDIT node errored out (not just plan), it returned empty
    # findings -> a 1.0 here would be a FALSE clean. Force max risk so routing escalates.
    errors = state.get("node_errors", [])
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
    
    return {
        "messages" : [summary],
        "security_score": sec_score,
        "quality_score": qual_score,
        "test_score": test_score,
    }