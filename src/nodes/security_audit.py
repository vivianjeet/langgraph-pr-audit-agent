# this node checks for security of the code changes, 
# using a ReAct pattern to elicit detailed reasoning from the LLM 
# and to ensure a comprehensive analysis.
import os
import instructor
from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.state import SecurityFinding, AuditState
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

# initialise instructor client with Gemini (Gemini 2.5 Turbo)
# client = instructor.from_genai(genai(api_key=os.environ.get("GEMINI_API_KEY")))
genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client = instructor.from_genai(genai_client)

FAST_MODEL = "gemini-2.5-flash"
SMALL_TOKEN_COUNT = 4000

class SecurityAuditOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Follow this ReAct flow: Reason (what security concerns exist?) -> "
            "Act (Analyse lines) -> Observe (findings) -> Verify (false positive "
            "check?)"
        )
    )
    findings: list[SecurityFinding] = Field(
        default_factory=list,
        description= "List of identified security vulnerabilities. Emppty if none found"
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
        response_model=SecurityAuditOutput,
        max_retries=2,
        generation_config={"max_output_tokens": SMALL_TOKEN_COUNT},
    )

def security_audit_node(state: AuditState):
    """
    Analyses the parsed PR for security vulnerabilities using the ReAct pattern.
    Validates output via instructor to enforce compliance with the SecurityFinding schema.
    Plan Aware
    """

    # Get the parsed diff from the ingest node (should be the last message)
    parsed_diff = state.get("parsed_diff","")
    plan = state.get("audit_plan",{})
    focus = ", ".join(plan.get("focus_areas",[])) or "general review (no plan available)"
    
    if not parsed_diff.strip():
        return {
            "messages": ["System: security_audit skipped - No parsed diff found in state."],
            "security_findings": [],
        }

    system_prompt = (
        "You are a senior security engineer conducting a PR audit. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise them: {{focus}}\n"
        "Analyse the following code changes for security vulnerabilities, specifically "
        "focussing on: \n"
        "- OWASP Top 10 \n"
        "- SQL Injection \n"
        "- PII data leaks (e.g. accidentally committing secrets or keys, "
        "or exposing personal data like Customer Records, PAN, Aadhaar etc) \n"
        "- Authentication bypass \n"
        "- Insecure dependencies \n\n"
    )
    user_prompt = (
        "Code diff to analyze:\n"
        "{{diff}}"
    )
    messages = [
        {"role": "system", "content": system_prompt.replace("{{focus}}",focus)},
        {"role": "user", "content": user_prompt.replace("{{diff}}",parsed_diff)},
    ]
    try:
        response = _call_gemini(messages)
    except Exception as e:
        return {
            "messages": [f"System: security_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "security_findings": [],
        }

    #Format a system message summary
    new_message = (
        "System: Security checks complete. \n"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} issues\n"
    )

    return {
        "messages" : [new_message],
        "security_findings": response.findings
    }