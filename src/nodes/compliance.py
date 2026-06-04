# src/nodes/compliance.py - pull regulatory context for the PR via MCP tools, 
# ReAct-style.
# This is the agent acting as an MCP CLIENT. Sits between retrieve and plan: 
# precedent (retrieve)
# and regulatory context (here) both land before the plan triages.
# Fail-closed: no tools / tool error / not-regulated -> empty context + a 
# visible message,
# never a crash and never a silent "looks clean".
from pydantic import BaseModel, Field
from src.memory import AgentMemorySystem as AMS, AMSState
from src.mcp_client import load_mcp_tools
from src.llm_retry import call_gemini_async, QuotaExhaustedError
from src.text_utils import clip

FAST_MODEL = "gemini-2.5-flash"
COMPLIANCE_TOKENS = 2000

class ComplianceQuery(BaseModel):
    """LLM triage of weather the diff is regulated, and what to search for."""
    needs_lookup: bool = Field(
        description=(
            "True if this diff touches ANY regulated concern across frameworks: "
            "personal data / PII (privacy), payment-card data (PCI), patient health "
            "data (HIPAA), auth, money movement, or audit loggin. "
            "False for docs/typos/tests."
        )
    )
    queries: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 short compliance search queries derived from the diff. Empty if "
            "needs_lookup is False"
        )
    )

async def compliance_node(state: AMSState):
    """
    Reason: is this diff regulated, and for what? 
    Act: run search_compliance_docs per query. 
    Observe: Collect passages into the audit substate so the security 
    prompt can cite them.
    """
    ams = AMS(state)
    parsed_diff = ams.read("parsed_diff","")
    if not parsed_diff.strip():
        return {
            "audit": {
                "messages": ["System: compliance skipped - no parsed diff."],
                "compliance_context" : []
            }
        }
    
    # --- Reason: ask the model whether a lookup is warranted, and for what ---
    system_prompt = (
        "You triage whether a code diff touches any regulated concerns across compliance"
        " frameworks: personal data / PII (privacy: GDPR/DPDP), payment-card data "
        "(PCI-DSS), patient health data (HIPAA), authentication, money movement, or "
        "audit logging. If so, propose 1-3 short search queries for a multi-framework "
        "compliance document store. If not, set needs_lookup=False with no queries."
    )
    messages = [
        {"role": "system", "content" : system_prompt},
        {"role": "user",  "content": f"Diff:\n{parsed_diff}"},
    ]
    try:
        triage = await call_gemini_async(model=FAST_MODEL, messages=messages,
                                response_model=ComplianceQuery,
                                max_output_tokens=COMPLIANCE_TOKENS)
    except QuotaExhaustedError:
        raise
    except Exception as e:
        return {
            "audit": {
                "messages": [f"System: compliance triage failed ({type(e).__name__}); no context."],
                "compliance_context": [],
                "node_errors": [f"compliance: triage {type(e).__name__} - {e}"],
                }
        }
    
    if not triage.needs_lookup or not triage.queries:
        return {
            "audit": {
                "messages" : ["System: compliance - diff not regulated no lookup."],
                "compliance_context": []
            }
        }
    
    # --- Act: run the search_compliance_docs MCP tool for each query. ---
    tools = await load_mcp_tools()
    search = next((t for t in tools if t.name == "search_compliance_docs"), None)
    if search is None:
        return {
            "audit":{
                "messages": ["System: compliance - search tool unavailable; no context"],
                "compliance_context": []
            }
            
        }
    hits = []
    for q in triage.queries[:3]:
        try:
            result = await search.ainvoke({"query":q, "k":3})
        except Exception:
            continue
        hits.extend(result if isinstance(result, list) else [])
    
    # --- Observe: surface a trace line + carry the passages for the security prompt. ---
    lines = [f"- [{h.get('framework', '?')}] {clip(h.get('text', ''), 160)} (src: {h.get('source', '?')})"
             for h in hits]
    msg = ("System: Compliance context:\n" + "\n".join(lines)) if lines else \
          "System: Compliance - no matching passages above threshold."
    return {"audit": {"messages": [msg], "compliance_context": hits}}