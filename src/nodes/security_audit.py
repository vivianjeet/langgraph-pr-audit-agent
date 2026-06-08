# Checks the security of the code changes. Uses a ReAct pattern so the
# LLM reasons step by step before it reports findings.
from pydantic import BaseModel, Field
from src.state import SecurityFinding, RuleCategory
from src.llm_retry import call_gemini_async, QuotaExhaustedError
from src.memory import AgentMemorySystem as AMS, AMSState
import src.config as cfg
from src.llm_client import llm, cached_system

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
        "Assign each finding a severity using THIS scale for SECURITY, and do not inflate it:\n"
        "- CRITICAL: a directly exploitable vulnerability - SQL/command injection, auth bypass, "
        "RCE, or a committed secret/credential.\n"
        "- HIGH: a serious weakness that needs a specific precondition to exploit "
        "(e.g. missing authz check, sensitive data exposure).\n"
        "- MEDIUM: a hardening gap or defence-in-depth issue, not directly exploitable.\n"
        "- LOW: minor / informational.\n"
        "Only report ACTUAL security issues. A rename, refactor or non-security change has NO "
        "security findings - return an EMPTY list. Do not invent vulnerabilities.\n\n"

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

    cache_note = ""
    if compliance:
        # Security caches the PREFIX (instructions+rules+compliance), NOT the diff - the opposite axis
        # from the other Flash nodes (which cache the diff via audit_with_diff_cache). The prefix is
        # byte-identical across DIFFERENT PRs of the same corpus, so this is the CROSS-PR optimization:
        # it pays when several PRs are audited within one cache window, reusing the cached prefix across
        # them. Security is on Pro (tier="powerful"), so it CANNOT share the Flash
        # diff-handle anyway - a CachedContent is model-bound. NOTE: today the prefix is usually under
        # Gemini's ~2048-token cache floor, so this falls back to plain Flash (below) until the
        # rules/compliance corpus grows past it; it's a deliberate forward-looking path, not dead code.
        try:
            stable = messages[0]["content"]
            diff_msg = messages[1]["content"]
            res = await llm.acall(tier="powerful", cache=True,
                                   response_model=SecurityAuditOutput,
                                   messages=[cached_system(stable),
                                             {"role": "user", "content": diff_msg}],
                                    max_output_tokens=cfg.AUDIT_MAX_OUTPUT_TOKENS)
            response = SecurityAuditOutput.model_validate_json(res.output)
            # Cache observability: cache_read_tokens>0 on a repeat run is the proof the stable
            # prefix was served from the CachedContent (claim). Surface it on the message.
            cache_note = (f"Cache: read={res.cache_read_tokens} input={res.input_tokens} "
                          f"output={res.output_tokens} cost=${res.cost_usd:.6f}\n")
        except QuotaExhaustedError:
            raise
        except Exception:
            # The cache couldn't be built (e.g. the prefix is under Gemini's ~2048-token
            # cache floor). That only defeats the CACHE, not Pro - a regulated diff's
            # security audit must STAY on Pro. Re-run uncached on the powerful tier (not
            # Flash). Going back through the router also means this call is traced and
            # honestly reports model=Pro. If Pro is genuinely exhausted the router's
            # rotation / QuotaExhaustedError handling applies, since fallback is no longer
            # disabled once the cache flag is dropped.
            # The non-cache router path runs through Instructor, so res.output is an
            # already-parsed SecurityAuditOutput (unlike the cache path, which returns raw
            # JSON text and needs model_validate_json). Use it directly.
            res = await llm.acall(tier="powerful", messages=messages,
                                  response_model=SecurityAuditOutput,
                                  max_output_tokens=cfg.AUDIT_MAX_OUTPUT_TOKENS)
            response = res.output
    else:
        try:
            response = await call_gemini_async(model=cfg.GEMINI_FLASH_MODEL,messages=messages,
                               response_model=SecurityAuditOutput,
                               max_output_tokens=cfg.AUDIT_MAX_OUTPUT_TOKENS)
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
        f"{cache_note}"
        f"Reasoning: {response.reasoning}\n"
        f"Found {len(response.findings)} issues\n"
    )

    return {"audit": {
        "messages" : [new_message],
        "security_findings": response.findings
    }}