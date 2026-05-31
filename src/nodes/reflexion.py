# Reflection : a SMARTER model critiques the audit and decides if a second pass is warranted

import os
import instructor
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.state import AuditState
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client = instructor.from_genai(genai_client)

SMART_MODEL = "gemini-2.5-pro"
MEDIUM_TOKEN_COUNT = 6000

RELEVANT_PREFIXES = (
    "System: Audit plan",
    "System: Security checks complete",
    "System: Quality checks complete",
    "System: Test audit completed",
    "System: Synthesized report"
)


class ReflectionOutput(BaseModel):
    gaps_identified: list[str] = Field(
        description="Concrete things the audit likely missed (categories, file types, edge cases)"
    )
    additional_checks_needed: list[str] = Field(
        description="Sepcific extra checks the next pass should perform"
    )
    confidence_score: float = Field(
        ge=0.0, le=1.0,
        description="How confident the critic is that the audit is now complete (0-1)"
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_gemini(messages):
    return client.chat.completions.create(
        model=SMART_MODEL,
        messages=messages,
        response_model=ReflectionOutput,
        max_retries=2,
        generation_config={"max_output_tokens": MEDIUM_TOKEN_COUNT},
    )
def reflexion_node(state: AuditState):
    """Critique the synthesized report; append critique to history; bump the loop guard"""
    current = state.get("iteration_count",0)
    
    # Walk newest->oldest; keep the first time we see each prefix, skip older dupes
    latest = {}
    for m in reversed(state.get("messages",[])):
        text = str(m)
        for prefix in RELEVANT_PREFIXES:
            if text.startswith(prefix) and prefix not in latest:
                latest[prefix] = text
    # Re Order to match RELEVANT_PREFIXES (dict preserves insertions so we
    #  rebuild in canonical order)
    # Give the critic the recent transcript so that it can reason about what 
    # was/wasn't checked.
    transcript = "\n".join(latest[p] for p in RELEVANT_PREFIXES if p in latest)
    sec_findings = state.get("security_findings",[])
    quality_findings = state.get("quality_findings",[])
    test_findings = state.get("test_findings",[])
    sec_score = state.get("security_score",1.0)
    quality_score = state.get("quality_score",1.0)
    test_score = state.get("test_score",1.0)

    system_prompt = (
        "You are a principal security reviewer auditing ANOTHER reviewer's work on a PR. "
        "Be skeptical. Critique the audit below: \n"
        "- Did it check ALL relevant OWASP categories?\n"
        "- Are there changed file types or code paths it ignored ?\n"
        " Is the security_score consistent with the security_findings "
        "(e.g. high score but auth code touched, or any such security discrepancy)\n"
        " Is the quality_score consistent with the quality_findings "
        "(e.g. high score but code quality bad, or any such quality discrepancy)\n"
        " Is the test_score consistent with the test_coverage_findings "
        "(e.g. high score but a lot of test missing, or any such test coverage discrepancy)\n\n"
    )
    user_prompt = (
        f"security_score: {sec_score}\n"
        f"quality_score: {quality_score}\n"
        f"test_score: {test_score}\n"
        f"security_findings: {sec_findings}\n"
        f"quality_findings: {quality_findings}\n"
        f"test_coverage_findings: {test_findings}\n"
        f"Recent transcript: {transcript}"
    )
    messages=[
            {"role" : "system", "content" : system_prompt},
            {"role" : "user", "content" : user_prompt}
        ]
    try:
        critique = _call_gemini(messages)
    except Exception as e:
        return {
            "messages": [f"System: reflection failed after retries ({type(e).__name__}); no findings recorded."],
            "security_findings": [],
        }

    msg = (
        f"System: Reflexion (iteration {current + 1}).\n"
        f"Gaps: {critique.gaps_identified}\n"
        f"Extra checks: {critique.additional_checks_needed}\n"
        f"Critic confidence: {critique.confidence_score}\n"
    )

    return {
        "messages": [msg],
        "iteration_count": current + 1,
        "confidence_score": critique.confidence_score,
        "gaps_identified": critique.gaps_identified
    }
