# Test-coverage auditor: flags missing/weak tests for changed code (ReAct + Instructor)
import os
import instructor
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.state import CoverageFinding, AuditState
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

gennai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client = instructor.from_genai(gennai_client)

FAST_MODEL = "gemini-2.5-flash"
SMALL_TOKEN_COUNT = 4000

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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini(messages):
    return client.chat.completions.create(
        model=FAST_MODEL,
        messages=messages,
        response_model=CoverageAuditOutput,
        max_retries=2,
        generation_config={"max_output_tokens": SMALL_TOKEN_COUNT},
    )

def coverage_audit_node(state: AuditState):
    """Analyse the parsed PR for missing test coverage (critical for sage deployement). Plan aware"""

    parsed_diff = state.get("parsed_diff", "")
    plan = state.get("audit_plan", {})
    focus = ", ".join(plan.get("focus_areas",[])) or "general review (no plan available)"

    if not parsed_diff.strip():
        return {
            "messages": ["System: test_coverage_audit skipped - No parsed diff found in state."],
            "test_findings": [],
        }
    system_prompt = (
        "You are a test engineer reviewing a PR for test coverage. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise test coverage for them: {{focus}}\n"
        "Identify code paths that changed but have NO corresponding test, focusing on: "
        "- Payment / transaction logic (must have edge-case + failure tests)\n"
        "- Authentication / authorization changes\n"
        "- Input validation and error handling\n"
        "A new function with no test is a finding. A bug-fix with no regression test is a finding.\n\n"
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
        response = _call_gemini(messages)
    except Exception as e:
        return {
            "messages": [f"System: coverage_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "test_findings": [],
        }

    new_message = (
        "System: Test audit completed. \n"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} gaps\n"
    )
    return {
        "messages": [new_message],
        "test_findings": response.findings,
    }