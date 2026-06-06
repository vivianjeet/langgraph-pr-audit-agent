# grounded compliance citations over the compliance passages. Given the diff + the passages 
# search_compliance_docs returned, ask Gemini which passages the diff may violate and to 
# QUOTE the exact span it relied on per claim; then VERIFY each quoted span is a real substring 
# of its passage and drop hallucinated spans before returning. Trust-but-verify: the substring 
# check is what turns "the model quoted something" into "a verbatim citation". Runs on the 
# existing Gemini spine (call_gemini in llm_retry.py); Best-effort, fail-closed.

from pydantic import BaseModel, Field
from src.llm_retry import call_gemini, QuotaExhaustedError

CITATION_MODEL = "gemini-2.5-flash" # grounded extraction over short passages, not deep reasoning
CITATION_TOKENS = 1024

class _Citation(BaseModel):
    source: str = Field(description=("Which passage sources/document this span is quoted from"
                        " (the passages's `source`)."))
    cited_text: str = Field(description= ("The EXACT verbatim span copied from that passage - no " \
                        "paraphrase, no edits."))

class _Claim(BaseModel):
    claim: str = Field(description="One compliance concern the diff may violate.")
    citations: list[_Citation] = Field(default_factory=list, description = ("The exact passage"
                            " span(s) this concern is grounded in.") )

class _CitedClaims(BaseModel):
    claims: list[_Claim] = Field(default_factory=list)

def cited_compliance_claims(diff: str, passages: list[dict]) -> list[dict]:
    """Which compliance passages the diff may violate, each with a VERIFIED citation span.
    Returns [{claim, citations:[{source, cited_text}]}]. Best-effort: [] on any failure or when
    there are no passages (a missing citation is NOT a clean bill of health - fail-closed).
    Every returned cited_text is guaranteed to be a real substring of the passage it names."""
    if not passages:
        return []
    # Index passage text by source so we can verify a quoted span really came from that passage.
    by_source: dict[str, str] = {}
    for p in passages:
        by_source.setdefault(p.get("source","?"),"")
        by_source[p.get("source","?")] += "\n" + (p.get("text","") or "")
    catalogue = "\n\n".join(f"[source: {p.get('source','?')}] {p.get('text','')}" for p in passages)
    messages = [
        {"role": "system", "content" : ("You ground compliance findings in regulatory text. "
                                        "Given a code diff and a list of regulatory passages, list"
                                        " which passages the diff may violate. For each concern, "
                                        "COPY the exact span you relied on VERBATIM from the passage into"
                                        "cited_text (no paraphrase) and set source to that passage's source"
                                        ". Do not invent text that is not in a passage.")},
        {"role": "user", "content": "Passages:\n{{catalogue}}\n\nDiff:\n{{diff}}"
            .replace("{{catalogue}}", catalogue).replace("{{diff}}", diff)},
    ]
    try: out = call_gemini(model=CITATION_MODEL, messages=messages, response_model=_CitedClaims,
                           max_output_tokens=CITATION_TOKENS)
    except QuotaExhaustedError:
        raise
    except Exception:
        return []
    
    # VERIFY: keep only spans that are a real substring of the named passage. This is the trust
    # boundary - a hallucinated span (model invented a quote) fails the check and is dropped.
    verified = []
    for claim in out.claims:
        good = [{"source": c.source, "cited_text": c.cited_text} for c in claim.citations
                if c.cited_text and c.cited_text.strip() in by_source.get(c.source,"")]
        if good:
            verified.append({"claim" : claim.claim, "citations": good})
    return verified