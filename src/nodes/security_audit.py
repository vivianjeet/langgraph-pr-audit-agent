# Checks the security of the code changes. Uses a ReAct pattern so the
# LLM reasons step by step before it reports findings.
from pydantic import BaseModel, Field
from src.state import SecurityFinding, RuleCategory
from src.llm_retry import call_gemini_async, QuotaExhaustedError
from src.memory import AgentMemorySystem as AMS, AMSState

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

async def security_audit_node(state: AMSState):
    """
    Analyses the parsed PR for security vulnerabilities using the ReAct pattern.
    Validates output via instructor to enforce compliance with the SecurityFinding schema.
    Plan Aware
    """

    ams = AMS(state)
    # Get the parsed diff from the ingest node (should be the last message)
    parsed_diff = ams.read("parsed_diff","")
    plan = ams.read("audit_plan",{})
    focus = ", ".join(plan.get("focus_areas",[])) or "general review (no plan available)"

    if not parsed_diff.strip():
        return {"audit": {
            "messages": ["System: security_audit skipped - No parsed diff found in state."],
            "security_findings": [],
        }}

    # Procedural memory: enforce this node's DOMAIN rules LITERALLY. Rules were recalled
    # ONCE in the retrieve node and live in the `procedural` channel - read them from there
    # (no re-query). security/quality/coverage pull their own.
    rules_block = AMS.rules_block(state.get("procedural", {}), (RuleCategory.SECURITY,))

    # Compliance passages the compliance node pulled (MCP). Inject verbatim so a finding can
    # cite the regulation it breaks. Empty -> the placeholder collapses (no prompt pollution).

    compliance = ams.read("compliance_context", [])
    compliance_block = ""
    if compliance:
        compliance_block = (
            "Relevant regulatory passages (cite the source + framework when a finding maps to one):\n"
            + "\n".join(f"- [{c.get('framework','?')}] {c.get('text','')} (src: {c.get('source','?')})"
                        for c in compliance)
            + "\n\n"
        )

    system_prompt = (
        "You are a senior security engineer conducting a PR audit. "
        "The lead reviewer's audit plan flagged these focus areas - prioritise them: {{focus}}\n"
        "{{rules}} \n"
        "{{compliance}} \n"
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
        {"role": "system", "content": system_prompt
            .replace("{{focus}}", focus)
            .replace("{{rules}}", rules_block)
            .replace("{{compliance}}", compliance_block)},
        {"role": "user", "content": user_prompt.replace("{{diff}}",parsed_diff)},
    ]
    try:
        response = await call_gemini_async(model=FAST_MODEL,messages=messages,
                               response_model=SecurityAuditOutput,
                               max_output_tokens=SMALL_TOKEN_COUNT)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {"audit": {
            "messages": [f"System: security_audit failed after retries ({type(e).__name__}); no findings recorded."],
            "security_findings": [],
            "node_errors": [f"security_audit: {type(e).__name__} - {str(e)}"]
        }}

    #Format a system message summary
    new_message = (
        "System: Security checks complete. \n"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} issues\n"
    )

    return {"audit": {
        "messages" : [new_message],
        "security_findings": response.findings
    }}