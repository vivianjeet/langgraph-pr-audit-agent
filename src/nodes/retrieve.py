# Pull similar past PR audits so the palnner/audits can recognise known risky patterns.
from src.state import AuditState
from src.db.vectorstore import retrieve_similar_prs

def retrieve_context_node(state: AuditState):
    """
    Embed the incoming diff summary, fetch precedent audits (semantic,
    threshold 0.7).
    """
    parsed_diff = state.get("parsed_diff","")
    try:
        similar = retrieve_similar_prs(parsed_diff, k = 3)
    except Exception as e:
        # The DB down should never crash the audit - degrade gracefully
        return {"messages":[f"System: Context retrieval skipped ({e})"],
                "similar_prs" : []}
    
    if not similar:
        return {"messages" : ["System: No similar past PRs found above threshold."],
                "similar_prs" : []}
    
    lines = [f"- {s['pr_summary'][:120]} (sim={s['similarity']:.2f})" for s in similar]
    return {
        "messages" : ["System: Retrieved similar PR history:\n" + "\n".join(lines)],
        "similar_prs" : similar,
    }