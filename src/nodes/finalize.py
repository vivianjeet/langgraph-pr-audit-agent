# Finalize: assemble the structured report (JSON + Markdown) and persist for future precendent.
from src.memory import AgentMemorySystem as AMS, AMSState
from src.text_utils import clip

# `report` findings are dicts, so `severity` is a plain string ("critical"); rank
# explicitly so `[:3]` is the 3 WORST findings, not just the first 3 the LLM emitted.
_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4, "none": 5}

def _findings_to_dicts(findings):
    return [f.model_dump(mode="json") for f in findings]

def finalize_report_node(state: AMSState):
    """
    Produce the final report dict + markdown
    store it in pgvector for future retrieval.
    """
    ams = AMS(state)
    report = {
        "security_score": ams.read("security_score",1.0),
        "quality_score": ams.read("quality_score",1.0),
        "test_score": ams.read("test_score", 1.0),
        "human_decision": ams.read("human_decision", "n/a"),
        "iterations" : ams.read("iteration_count", 0),
        "security_findings": _findings_to_dicts(ams.read("security_findings",[])),
        "quality_findings": _findings_to_dicts(ams.read("quality_findings",[])),
        "test_findings": _findings_to_dicts(ams.read("test_findings",[])),
        "files_changed": ams.read("files_changed",[]),
    }

    md = [
        "# PR Audit Report",
        f"**Security score:** {report['security_score']}  |  "
        f"**Quality score:** {report['quality_score']}  |  "
        f"**Test score:** {report['test_score']}  |  "
        f"**Human decision:** {report['human_decision']}  |  "
        f"**Reflexion iterations:** {report['iterations']}",
        f"**Files changed:** {', '.join(report['files_changed']) or 'n/a'}",
        "",
        f"## Security findings ({len(report['security_findings'])})",
    ]

    for f in report["security_findings"]:
        md.append(f"- `{f['severity']}` {f['file_path']}:{f['line_number']} "
                  f"[{f['cwe_id']}] - {f['description']}")
    md.append(f"\n## Quality findings ({len(report['quality_findings'])})")
    for f in report["quality_findings"]:
        md.append(f"- `{f['severity']}` {f['file_path']}:{f['line_number']} - {f['description']}")
    md.append(f"\n## Test-coverage gaps ({len(report['test_findings'])})")
    for f in report["test_findings"]:
        md.append(f"- `{f['severity']}` {f['file_path']} - {f['description']}")

    markdown = "\n".join(md)
    # Compliance audit trail: each claim with the exact regulatory span it's grounded in.
    citations = ams.read("compliance_citations", [])
    if citations:
        lines = ["", "### Compliance audit trail", ""]
        for c in citations:
            lines.append(f"- {c['claim']}")
            for cite in c["citations"]:
                lines.append(f"  > \"{cite['cited_text']}\" -- {cite['source']}")
        markdown += "\n".join(lines) + "\n"

    # Persist so the NEXT similar PR can retrieve this as precedent (best-effort).
    summary = (f"Files: {report['files_changed']}; quality_score {report['quality_score']}; "
        f"security_score {report['security_score']}; test_score {report['test_score']}; "
        f"{len(report['security_findings'])} security findings, "
        f"{len(report['quality_findings'])} quality findings, "
        f"{len(report['test_findings'])} test findings"
    )

    parsed_diff = ams.read("parsed_diff","")
    try:
        ams.store_pr_audit(summary, report, embed_text=parsed_diff or summary)
    except Exception as e:
        markdown += f"\n\n_(Note: audit not persisted: {e})_"
    
    # Episodic: a human-readable recap of THIS session,
    # for future "have we seen this before? recall." Covers ALL THREE finding types
    # (not security-only) so cross-session recall reflects the whole audit.
    def _top(findings, n=3):
        return sorted(findings, key=lambda f: _SEV_RANK.get(str(f["severity"]).lower(), 99))[:n]

    def _fmt(findings):
        return "; ".join(
            f"{f['severity']} {f['file_path']} {clip(f['description']).rstrip('. ')}" for f in findings
        ) or "none"

    top_security = _top(report["security_findings"])
    top_quality = _top(report["quality_findings"])
    top_test = _top(report["test_findings"])
    # Prefer the compacted session history (from compress_node, `compressed` channel) as the
    # episodic record when compression fired this run; else fall back to the structured summary.
    # compressed is a TOP-LEVEL AMSState channel -> read off state, not via ams.read (audit substate).
    compressed = state.get("compressed", [])
    if compressed:
        episode = "Compressed session history:\n" + "\n".join(str(m) for m in compressed)
    else:
        episode = (
            f"Files changed: {report['files_changed']}. "
            f"security={report['security_score']} quality={report['quality_score']} "
            f"test={report['test_score']}, decision={report['human_decision']}. "
            f"Top security: {_fmt(top_security)}. "
            f"Top quality: {_fmt(top_quality)}. "
            f"Top test gaps: {_fmt(top_test)}."
        )
    try:
        AMS.store_episode(
            episode,
            metadata={
                "files": report["files_changed"],
                "security_score": report["security_score"],
                "quality_score": report["quality_score"],
                "test_score": report["test_score"],
            },
        )
    except Exception as e:
        markdown += f"\n\n_(Note: episode not persisted: {e})_"

    # Procedural learning: promote THIS run's strongest findings into standing org rules
    # so future audits enforce them. Guard-railed (see AMS.learn_rules_from_findings):
    # only high-severity findings, deduped against existing rules. Best-effort.
    try:
        learned = AMS.learn_rules_from_findings(
            security=top_security, quality=top_quality, coverage=top_test,
            human_decision=report["human_decision"],
        )
        if learned:
            markdown += f"\n\n_(Learned {learned} new org rule(s) from this audit.)_"
    except Exception as e:
        markdown += f"\n\n_(Note: rule learning skipped: {e})_"

    return {"audit": {"messages" : [f"System: Final report ready.\n{markdown}"],
            "final_report": markdown}}