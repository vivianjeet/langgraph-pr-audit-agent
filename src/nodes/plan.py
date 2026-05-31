import os
import instructor
from google import genai
from dotenv import load_dotenv
from src.state import AuditState, AuditPlan, Severity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field

load_dotenv()

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
client = instructor.from_genai(genai_client)

FAST_MODEL = "gemini-2.5-flash"
SMALL_TOKEN_COUNT = 4000

class PlanAuditOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Triage rationale: what about this change drives risk? "
            "Consider blast radius (auth/payment/PII), surface area, and "
            "whether the change touches critical paths. Conclude with why "
            "this risk_level and audit_depth are warranted."
        )
    )
    plan: AuditPlan

@retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry = retry_if_exception_type(Exception),
        reraise=True
)
def _call_gemini(messages):
    return client.chat.completions.create(
        model=FAST_MODEL,
        messages=messages,
        response_model=PlanAuditOutput,
        max_retries=2,
        generation_config={"max_output_tokens": SMALL_TOKEN_COUNT}
)

def plan_audit_node(state: AuditState):
    """
    Look at the diff once, decide where to spend the audit effort
    """
    default_plan = AuditPlan(
            focus_areas=[],
            risk_level=Severity.NONE,
            audit_depth="shallow",
            files_to_prioritize=[]
        )
    
    parsed_diff = state.get("parsed_diff","")
    if not parsed_diff.strip():
        return {
            "messages" : ["System: plan skipped - no parsed diff found in state."],
            "audit_plan": default_plan.model_dump()
        }
    files = state.get("files_changed",[])
    system_prompt = (
        "You are the lead reviewer triaging a code change before deep audit. "
        "Given the diff and the list of changed files, produce an audit plan:\n"
        "- focus_areas: the 2-5 themes worth investigating\n"
        "- risk_level: overall a-priori risk\n"
        "- audit_depth: 'deep' if payment/auth/PII touched, 'standard' for normal logic changes, "
        "else 'shallow' \n"
        "- files_to_prioritize: subset of the changed files most likely to carry risk\n\n"
    )
    user_prompt = (
        "Changed files: {{files}}\n"
        "Diff: {{diff}}"
    )
    messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content": user_prompt.replace("{{files}}", str(files)).replace("{{diff}}", parsed_diff)}
        ]
    try:
        response: PlanAuditOutput = _call_gemini(messages)
    except Exception as e:
        return {
            "messages": [f"System: plan failed after retries ({type(e).__name__}); using default plan."],
            "audit_plan": default_plan.model_dump()
        }
    
    valid = set(files)
    response.plan.files_to_prioritize = [
        f for f in response.plan.files_to_prioritize if f in valid
    ]

    return {
        "messages" : [f"System: Audit plan -> depth={response.plan.audit_depth}, "
                      f"reasoning: {response.reasoning}, "
                      f"risk={response.plan.risk_level.value}, focus={response.plan.focus_areas}"],
        "audit_plan": response.plan.model_dump()
    }