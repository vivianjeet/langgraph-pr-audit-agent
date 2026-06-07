# src/nodes/compliance.py - pull regulatory context for the PR via MCP tools.
# The agent acts as an MCP client here. The node sits between retrieve and plan, so
# precedent (retrieve) and regulatory context (here) both land before the plan triages.
# Fail-closed: no tools, a tool error or an unregulated diff all give empty context plus
# a visible message. Never a crash, never a silent "looks clean".
import json
from pydantic import BaseModel, Field
from src.memory import AgentMemorySystem as AMS, AMSState
from src.mcp_client import load_mcp_tools
from src.llm_retry import call_gemini_async, QuotaExhaustedError
from src.text_utils import clip
from src.citations import cited_compliance_claims
import src.config as cfg


def _coerce(item):
    """One MCP item -> the original {text, source, framework, similarity} dict or None.
    The wire wraps our server's dict THREE possible ways, so unwrap each:
      1. a JSON string of the dict;
      2. an MCP content block {'type':'text','text': '<json of the dict>'} - the dict is
         buried in .text, so the top level only has `text`/`type` (that's why framework/source
         came back '?': they were never at the top level, they're inside the nested JSON);
      3. already the raw dict (direct/in-process call).
    Drop anything that won't parse rather than poison the prompt with '?'-filled noise."""
    if isinstance(item, str):
        try:
            item = json.loads(item)
        except (ValueError, TypeError):
            return None
    if not isinstance(item, dict):
        return None
    # Content-block wrapper: the real payload is the JSON in .text, not this outer dict.
    if "framework" not in item and isinstance(item.get("text"), str):
        try:
            inner = json.loads(item["text"])
        except (ValueError, TypeError):
            return item            # genuine plain-text passage, not a wrapped dict; keep as-is
        if isinstance(inner, dict):
            return inner
    return item


def _normalize_hits(result) -> list[dict]:
    """The server returns list[dict], but langchain-mcp-adapters re-serializes tool output over
    the wire - as a JSON string, a list of content blocks or a JSON string of the whole list.
    Normalise every shape back to list[dict] so h.get('framework')/'source' resolve (else '?')."""
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (ValueError, TypeError):
            return []
    if isinstance(result, dict):
        result = [result]
    return [d for d in (_coerce(i) for i in (result or [])) if d is not None]


class ComplianceQuery(BaseModel):
    """LLM triage of whether the diff is regulated, and what to search for."""
    needs_lookup: bool = Field(
        description=(
            "True if this diff touches ANY regulated concern across frameworks: "
            "personal data / PII (privacy), payment-card data (PCI), patient health "
            "data (HIPAA), auth, money movement, insecure data handling (e.g. SQL injection "
            "/ untrusted input in queries) or audit logging. "
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
    """Decide if the diff is regulated and for what, then run search_compliance_docs for
    each query the model proposes and collect the passages into the audit substate so the
    security prompt can cite them."""
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
        "(PCI-DSS), patient health data (HIPAA), authentication, money movement or "
        "audit logging. If so, propose 1-3 short search queries for a multi-framework "
        "compliance document store. If not, set needs_lookup=False with no queries."
    )
    messages = [
        {"role": "system", "content" : system_prompt},
        {"role": "user",  "content": "Diff:\n{{parsed_diff}}"
                                .replace("{{parsed_diff}}", parsed_diff)},
    ]
    try:
        triage = await call_gemini_async(model=cfg.GEMINI_FLASH_MODEL, messages=messages,
                                response_model=ComplianceQuery,
                                max_output_tokens=cfg.COMPLIANCE_MAX_OUTPUT_TOKENS)
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
    for q in triage.queries[:cfg.MAX_COMPLIANCE_QUERIES]:
        try:
            result = await search.ainvoke({"query":q, "k":cfg.SEARCH_DEFAULT_K})
        except Exception:
            continue
        hits.extend(_normalize_hits(result))
    
    try:
        cited = cited_compliance_claims(parsed_diff, hits)
    except Exception:
        cited = []
    
    # --- Observe: surface a trace line + carry the passages for the security prompt. ---
    lines = [f"- [{h.get('framework', '?')}] {clip(h.get('text', ''), cfg.CLIP_WIDTH_LONG)} (src: {h.get('source', '?')})"
             for h in hits]
    msg = ("System: Compliance context:\n" + "\n".join(lines)) if lines else \
          "System: Compliance - no matching passages above threshold."
    return {"audit": {"messages": [msg], "compliance_context": hits, "compliance_citations": cited}}
