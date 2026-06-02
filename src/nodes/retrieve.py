# Pull similar past PR audits so the palnner/audits can recognise known risky patterns.
from src.memory import AgentMemorySystem as AMS, AMSState
from src.text_utils import clip
from src.state import RuleCategory

# All procedural-rule categories, recalled ONCE here and written to the `procedural`
# channel; plan + each audit node read their own subset from the channel (no re-query).
RULE_CATEGORIES = tuple(RuleCategory)   # (SECURITY, QUALITY, COVERAGE) as enum members

def retrieve_context_node(state: AMSState):
    """
    Embed the incoming diff summary, fetch precedent audits (semantic,
    threshold 0.7). Retrieved precedent lands in the top-level `semantic` channel;
    the human-readable trace line goes to the audit substate's messages.

    Fetch precedent for this PR: similar past audits (semantic) + similar past
    sessions (episodic). Both degrade gracefully if their store is unavailable.
    """
    ams = AMS(state)
    parsed_diff = ams.read("parsed_diff","")

    try:
        similar = ams.recall_similar_prs(parsed_diff, k = 3)
    except Exception as e:
        # The DB down should never crash the audit - degrade gracefully.
        # don't return early - we still want to try episodic recall below.
        similar = []
        sem_err = f"System: Semantic retrieval skipped ({e})"
    else:
        sem_err = None

    try:
        episodes = AMS.recall_episodes(parsed_diff, k=2)
    except Exception:
        episodes = []

    # Procedural rules: recalled ONCE here (grouped by category) into the `procedural`
    # channel; plan + audit nodes read their own categories from it, never re-querying.
    rules = AMS.recall_all_rules(RULE_CATEGORIES)

    msgs = []
    if sem_err:
        msgs.append(sem_err)
    if similar:
        lines = [f"- {clip(s['pr_summary'], 120)} (sim={s['similarity']:.2f})" for s in similar]
        msgs.append("System: Retrieved similar PR history:\n" + "\n".join(lines))
    if episodes:
        elines = [f"- {clip(e['summary'], 140)} (sim={e['similarity']:.2f})" for e in episodes]
        msgs.append("System: Recalled past sessions:\n" + "\n".join(elines))
    if not msgs:
        msgs = ["System: No similar past PRs or sessions found above threshold."]

    # Precedent lands in its own top-level channel (semantic / episodic), NOT in the
    # audit substate - keeps the per-run AuditState clean and matches AMSState's schema.
    # Only the human-readable trace goes to audit.messages.
    return {
        "audit": {"messages": msgs},
        "semantic": similar,
        "episodic": episodes,
        "procedural": rules,
    }