# Checks the PR for code-quality issues: code smells, maintainability, best practices.
# Same ReAct pattern as the security node, so the LLM reasons before it reports.
from pydantic import BaseModel, Field
from src.state import QualityFinding, RuleCategory
from src.llm_retry import QuotaExhaustedError
from src.memory import AgentMemorySystem as AMS, AMSState
from src.llm_client import audit_with_diff_cache
import src.config as cfg

class QualityAuditOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Follow this ReAct flow: "
            "Reason (What quality concerns exist? What code smells are present?) -> "
            "Act(Act by checking code against common quality issues like code smells, "
            "maintainability concerns, adherence to best and clean practices) -> "
            "Observe (findings) -> "
            "Verify (are these findings valid after false positive check?)"
        )
    )
    findings: list[QualityFinding] = Field(
        default_factory=list,
        description= "List of identified quality issues. Empty if none found."
    )

async def quality_audit_node(state: AMSState):
    """
    Analyses the parsed PR for code quality issues using the ReAct pattern.
    Validates output via instructor to enforce compliance with the QualityFinding schema.
    Plan Aware
    """

    ams = AMS(state)
    # Get the parsed diff from the ingest node (should be the last message)
    parsed_diff = ams.read("parsed_diff","")
    plan = ams.read("audit_plan",{})
    focus = ", ".join(plan.get("focus_areas",[] )) or "general review (no plan available)"

    if not parsed_diff.strip():
        return {"audit": {
            "messages": ["System: quality_audit skipped - No parsed diff found in state."],
            "quality_findings": [],
        }}

    # Procedural memory: enforce this node's DOMAIN rules (quality) literally. Rules were
    # recalled ONCE in retrieve and live in the `procedural` channel - read from there.
    # "" when no quality rules exist, so the {{rules}} placeholder collapses.
    rules_block = AMS.rules_block(state.get("procedural", {}), (RuleCategory.QUALITY,))

    system_prompt = (
        "You are a senior software engineer conducting a PR code quality audit. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise them: {{focus}}\n"
        "{{rules}}"
        "Analyze the following code changes specifically focusing on:\n"
        "- Code smells and anti-patterns\n"
        "- Hardcoded values and magic numbers\n"
        "- High cyclomatic complexity\n"
        "- DRY/SOLID principle violations\n\n"
        "Assign each finding a severity using THIS scale, and do not inflate it:\n"
        "- CRITICAL: reserved for issues that break the build or cause data loss. "
        "Code-quality smells are almost NEVER critical.\n"
        "- HIGH: a real maintainability risk a reviewer must address before merge.\n"
        "- MEDIUM: worth fixing but not blocking.\n"
        "- LOW: minor / stylistic.\n"
        "A rename, a small refactor, or a stylistic nit is LOW - never HIGH or CRITICAL.\n"
        "If the diff has no quality issues, return an EMPTY findings list. Do not invent findings.\n\n"
    )
    # Instructions = the rendered system prompt (the per-node VARIABLE part); the diff is cached and
    # shared across the fan-out nodes - see audit_with_diff_cache. Falls back to plain Flash internally.
    instructions = system_prompt.replace("{{focus}}", focus).replace("{{rules}}", rules_block)
    try:
        response, cache_note = await audit_with_diff_cache(
            parsed_diff, instructions, QualityAuditOutput, cfg.AUDIT_MAX_OUTPUT_TOKENS)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {"audit": {
            "messages": [f"System: quality_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "quality_findings": [],
            "node_errors": [f"quality_audit: {type(e).__name__} - {str(e)}"]
        }}

    new_message = (
        "System: Quality checks complete. \n"
        f"{cache_note}"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} issues\n"
    )
    return {"audit": {
        "messages": [new_message],
        "quality_findings": response.findings
    }}


           



