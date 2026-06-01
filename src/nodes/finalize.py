# Finalize: assemble the structured report (JSON + Markdown) and persist for future precendent.
from src.state import AuditState
from src.db.vectorstore import store_pr_audit

def _findings_to_dicts(findings):
    return [f.model_dump(mode="json") for f in findings]

def finalize_report_node(state: AuditState):
    """
    Produce the final report dict + markdown;
    store it in pgvector for future retrieval.
    """
    report = {
        "security_score": state.get("security_score",1.0),
        "quality_score": state.get("quality_score",1.0),
        "test_score": state.get("test_score", 1.0),
        "human_decision": state.get("human_decision", "n/a"),
        "iterations" : state.get("iteration_count", 0),
        "security_findings": _findings_to_dicts(state.get("security_findings",[])),
        "quality_findings": _findings_to_dicts(state.get("quality_findings",[])),
        "test_findings": _findings_to_dicts(state.get("test_findings",[])),
        "files_changed": state.get("files_changed",[])
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

    # Persist so the NEXT similar PR can retrieve this as precedent (best-effort).
    summary = (f"Files: {report['files_changed']}; quality_score {report['quality_score']}; "
        f"security_score {report['security_score']}; test_score {report['test_score']}; "
        f"{len(report['security_findings'])} security findings, "
        f"{len(report['quality_findings'])} quality findings, "
        f"{len(report['test_findings'])} test findings"
    )

    parsed_diff = state.get("parsed_diff","")
    try:
        store_pr_audit(summary, report, embed_text=parsed_diff or summary)
    except Exception as e:
        markdown += f"\n\n_(Note: audit not persisted: {e})_"
    
    return {"messages" : [f"System: Final report ready.\n{markdown}"],
            "final_report": markdown}