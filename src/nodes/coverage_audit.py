# Test-coverage auditor: flags missing/weak tests for changed code (ReAct + Instructor)
from pydantic import BaseModel, Field
from src.state import CoverageFinding, RuleCategory
from src.llm_retry import QuotaExhaustedError
from src.memory import AgentMemorySystem as AMS, AMSState
from src.llm_client import audit_with_diff_cache
import src.config as cfg

class CoverageAuditOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Follow this ReAct flow: Reason (what test coverage issues exist? "
            "what behaviour MUST be tested? are there any edge cases which "
            "probably dont require testing) -> "
            "Act (Analyse lines and existing tests. compare changed code vs added tests) -> "
            "Observe (findings, gaps, edge cases) -> Verify "
            "(false positive check? real gap? No edge cases missed?)"
        )
    )
    findings: list[CoverageFinding] = Field(
        default_factory=list,
        description= ("List of identified test coverage issues. Missing or inadequate tests. "
            "Empty if none found"
        )
    )

async def coverage_audit_node(state: AMSState):
    """Analyse the parsed PR for missing test coverage (critical for sage deployement). Plan aware"""

    ams = AMS(state)
    parsed_diff = ams.read("parsed_diff", "")
    plan = ams.read("audit_plan", {})
    focus = ", ".join(plan.get("focus_areas",[])) or "general review (no plan available)"

    if not parsed_diff.strip():
        return {"audit": {
            "messages": ["System: test_coverage_audit skipped - No parsed diff found in state."],
            "test_findings": [],
        }}
    # Procedural memory: enforce this node's DOMAIN rules (coverage) literally. Rules were
    # recalled ONCE in retrieve and live in the `procedural` channel - read from there.
    # "" when no coverage rules exist, so the {{rules}} placeholder collapses.
    rules_block = AMS.rules_block(state.get("procedural", {}), (RuleCategory.COVERAGE,))

    system_prompt = (
        "You are a test engineer reviewing a PR for test coverage. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise test coverage for them: {{focus}}\n"
        "{{rules}}"
        "Identify code paths that changed but have NO corresponding test, focusing on: "
        "- Payment / transaction logic (must have edge-case + failure tests)\n"
        "- Authentication / authorization changes\n"
        "- Input validation and error handling\n\n"
        "A new function with no test is a finding. A bug-fix with no regression test is a finding.\n"
        "CONSOLIDATE: do NOT emit one finding per function or per line. Group related gaps "
        "into a single finding - e.g. 'Service has 60 new methods with no tests' is ONE finding, "
        "not 60. Emit at most a handful of findings, highest-impact first.\n\n"
        "Assign each finding a severity using THIS scale for TEST COVERAGE, and do not inflate it:\n"
        "- CRITICAL: an untested security- or payment-critical path (auth, money movement, PII handling).\n"
        "- HIGH: untested core business logic that can fail silently.\n"
        "- MEDIUM: missing tests on ordinary changed code.\n"
        "- LOW: trivial/boilerplate code where a test adds little (renames, pass-through methods).\n"
        "If every changed path is adequately tested, return an EMPTY findings list. Do not invent gaps.\n\n"

    )
    # Instructions = the rendered system prompt (per-node VARIABLE part); the diff is cached and shared
    # across the fan-out nodes - see audit_with_diff_cache. Falls back to plain Flash internally.
    instructions = system_prompt.replace("{{focus}}", focus).replace("{{rules}}", rules_block)
    try:
        response, cache_note = await audit_with_diff_cache(
            parsed_diff, instructions, CoverageAuditOutput, cfg.AUDIT_MAX_OUTPUT_TOKENS)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {"audit": {
            "messages": [f"System: coverage_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "test_findings": [],
            "node_errors": [f"coverage_audit: {type(e).__name__} - {str(e)}"]
        }}

    new_message = (
        "System: Test audit completed. \n"
        f"{cache_note}"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} gaps\n"
    )
    return {"audit": {
        "messages": [new_message],
        "test_findings": response.findings,
    }}