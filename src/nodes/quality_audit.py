# Checks the PR for code-quality issues: code smells, maintainability, best practices.
# Same ReAct pattern as the security node, so the LLM reasons before it reports.
from pydantic import BaseModel, Field
from src.state import QualityFinding, AuditState
from src.llm_retry import call_gemini, QuotaExhaustedError

FAST_MODEL = "gemini-2.5-flash"
SMALL_TOKEN_COUNT = 4000

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

def quality_audit_node(state: AuditState):
    """
    Analyses the parsed PR for code quality issues using the ReAct pattern.
    Validates output via instructor to enforce compliance with the QualityFinding schema.
    Plan Aware
    """

    # Get the parsed diff from the ingest node (should be the last message)
    parsed_diff = state.get("parsed_diff","")
    plan = state.get("audit_plan",{})
    focus = ", ".join(plan.get("focus_areas",[] )) or "general review (no plan available)"

    if not parsed_diff.strip():
        return {
            "messages": ["System: quality_audit skipped - No parsed diff found in state."],
            "quality_findings": [],
        }

    system_prompt = (
        "You are a senior software engineer conducting a PR code quality audit. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise them: {{focus}}\n"
        "Analyze the following code changes specifically focusing on:\n"
        "- Code smells and anti-patterns\n"
        "- Hardcoded values and magic numbers\n"
        "- High cyclomatic complexity\n"
        "- DRY/SOLID principle violations\n\n"
    )
    user_prompt = (
        "Code diff to analyze:\n"
        "{{diff}}"
    )
    messages=[
            {"role":"system","content":system_prompt.replace("{{focus}}",focus)},
            {"role":"user","content": user_prompt.replace("{{diff}}", parsed_diff)}
        ]
    try:
        response = call_gemini(model=FAST_MODEL, messages=messages,
                               response_model=QualityAuditOutput,
                               max_output_tokens=SMALL_TOKEN_COUNT)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {
            "messages": [f"System: quality_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "quality_findings": [],
            "node_errors": [f"quality_audit: {type(e).__name__} - {str(e)}"]
        }

    new_message = (
        "System: Quality checks complete. \n"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} issues\n"
    )
    return {
        "messages": [new_message], 
        "quality_findings": response.findings
    }


           



