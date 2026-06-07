from src.state import AuditPlan, Severity, RuleCategory
from pydantic import BaseModel, Field
from src.llm_retry import QuotaExhaustedError
from src.llm_client import audit_with_diff_cache_sync
from src.memory import AgentMemorySystem as AMS, AMSState
from src.text_utils import clip as _clip
import src.config as cfg

class PlanAuditOutput(BaseModel):
    reasoning: str = Field(
        description=(
            "Triage rationale: what about this change drives risk? "
            "Consider blast radius (auth/payment/PII), surface area and "
            "whether the change touches critical paths. Conclude with why "
            "this risk_level and audit_depth are warranted."
        )
    )
    plan: AuditPlan

def plan_audit_node(state: AMSState):
    """
    Look at the diff once, decide where to spend the audit effort
    """
    default_plan = AuditPlan(
            focus_areas=[],
            risk_level=Severity.NONE,
            audit_depth="shallow",
            files_to_prioritize=[]
        )
    
    ams = AMS(state)
    parsed_diff = ams.read("parsed_diff","")

    if not parsed_diff.strip():
        return {"audit": {
            "messages" : ["System: plan skipped - no parsed diff found in state."],
            "audit_plan": default_plan.model_dump()
        }}

    # Pull memory: semantic (similar past audits), episodic (past sessions) and procedural
    # (org rules) were ALL recalled ONCE in the retrieve node and live in the TOP-LEVEL
    # channels. Read them off `state` directly (ams.read is audit-scoped, won't see them);
    # no DB re-query here. The plan uses rules to STEER triage (focus_areas/audit_depth);
    # the audit nodes separately enforce them verbatim from the same `procedural` channel.
    similar = state.get("semantic", []) or []
    episodes = state.get("episodic", []) or []
    procedural = state.get("procedural", {}) or {}
    rules = [r for cat in RuleCategory for r in procedural.get(cat.value, [])]

    precedent_block = ""
    if similar:
        precedent_block += "Similar past audits:\n" + "\n".join(
            f"- {_clip(s['pr_summary'], cfg.CLIP_WIDTH_LONG)}" for s in similar) + "\n"
    if episodes:
        precedent_block += "Relevant past sessions:\n" + "\n".join(
            f"- {_clip(e['summary'], cfg.CLIP_WIDTH_LONG)}" for e in episodes) + "\n"
    if rules:
        precedent_block += "Standing org rules to apply:\n" + "\n".join(
            f"- {r}" for r in rules) + "\n"

    files = ams.read("files_changed",[])
    system_prompt = (
        "You are the lead reviewer triaging a code change before deep audit. "
        "Given the diff and the list of changed files, produce an audit plan:\n"
        "- focus_areas: the 2-5 themes worth investigating\n"
        "- risk_level: overall a-priori risk\n"
        "- audit_depth: 'deep' if payment/auth/PII touched, 'standard' for normal logic changes, "
        "else 'shallow' \n"
        "- files_to_prioritize: subset of the changed files most likely to carry risk\n\n"
    )
    # The diff is the cached part (shared with compliance/quality/coverage); the variable part is this
    # node's instructions + the changed-files list + any precedent. Reuses the handle compliance primed.
    instructions = (
        system_prompt
        + "Changed files: " + str(files) + "\n"
        + precedent_block
        + "\nThe code diff is provided in context."
    )
    try:
        response, _ = audit_with_diff_cache_sync(
            parsed_diff, instructions, PlanAuditOutput, cfg.AUDIT_MAX_OUTPUT_TOKENS)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {"audit": {
            "messages": [f"System: plan failed after retries ({type(e).__name__}); using default plan."],
            "audit_plan": default_plan.model_dump(),
            "node_errors": [f"plan: {type(e).__name__} - {str(e)}"]
        }}
    
    valid = set(files)
    response.plan.files_to_prioritize = [
        f for f in response.plan.files_to_prioritize if f in valid
    ]

    return {"audit": {
        "messages" : [f"System: Audit plan -> depth={response.plan.audit_depth}, "
                      f"reasoning: {response.reasoning}, "
                      f"risk={response.plan.risk_level.value}, focus={response.plan.focus_areas}"],
        "audit_plan": response.plan.model_dump()
    }}