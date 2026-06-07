# AgentMemorySystem (AMS): the graph state IS the four-type memory system.
#   in-context  -> AMSState["audit"]   (the per-run AuditState; ephemeral, LangGraph-owned)
#   semantic    -> AMSState["semantic"] + pgvector precedent (similar past PR audits)
#   episodic    -> AMSState["episodic"] + pgvector (compressed past-session summaries)
#   procedural  -> AMSState["procedural"] + pgvector (keyed org rules / templates)
#   compressed  -> AMSState["compressed"] compacted session history
#
# AMS is STATEFUL: construct one per node as `ams = AMS(state)` wrapping the AMSState
# LangGraph just handed in. It lives/dies within a single node call (no stale snapshot).
#
# READ side  -> ams.read("key", default) reads the AUDIT substate (state["audit"]).
# WRITE side -> helpers RETURN partial-update dicts in AMSState shape. AMS NEVER mutates
#               the held state: writes flow back through the node's `return {...}` so the
#               reducers (merge_audit on `audit`; last-writer-wins on the memory channels)
#               + checkpointing + LangSmith tracing all fire on the returned dict.
# The three persistent types delegate to src/db/vectorstore.py.
from typing import Annotated, TypedDict
from src.state import AuditState, RuleCategory, RuleStatus
from src.db import vectorstore as vs
import src.config as cfg

_NO_LEARN = {"needs-changes", "reject"}

def merge_audit(old: dict | None, new: dict | None) -> dict:
    """Reducer for the nested `audit` channel of AMSState.

    LangGraph's reducers (operator.add) are registered on TOP-LEVEL channels. Once
    AuditState lives ONE level down (under AMSState["audit"]), the top-level channel
    is `audit` - a single dict with NO reducer, so LangGraph would last-writer-wins
    overwrite it. Under the parallel audit fan-out (security/quality/coverage all
    return `audit` at once) that silently loses writes.

    This reducer restores the per-field semantics AuditState declared: the two
    operator.add fields (`messages`, `node_errors`) accumulate; every other field is
    last-writer-wins (correct - only one branch writes each scalar/findings list).
    """
    old = old or {}
    new = new or {}
    merged = {**old, **new}
    for key in ("messages", "node_errors"):
        merged[key] = (old.get(key) or []) + (new.get(key) or [])
    return merged


class AMSState(TypedDict):
    """The AgentMemorySystem IS the graph state: four memory types as sibling channels.

    - audit      : in-context working memory (the per-run AuditState), custom-merged
                   by `merge_audit` so its inner reducers survive the nesting.
    - semantic   : similar past PR audits retrieved THIS run (was AuditState.similar_prs)
    - episodic   : compressed past-session summaries recalled this run
    - procedural : org rules active this run, grouped {category: [rules]} - recalled ONCE
                   (in retrieve) and read from the channel by plan + every audit node, so
                   rules are never re-queried per node (same recall-once pattern as the
                   semantic/episodic channels).
    The three memory channels are last-writer-wins: one node populates each per run.
    - compressed : compacted session history written by compress_node (proto-episodic).
    - force_compress : entry flag (from --large) that forces compression regardless of size.
    """
    audit: Annotated[AuditState, merge_audit]
    semantic: list
    episodic: list
    procedural: dict
    compressed: list
    force_compress: bool        # set once at entry from the --large flag; read by compress_node


class AgentMemorySystem:
    def __init__(self, state: "AMSState | dict"):
        # Cheap: just hold a reference to the live AMSState. No I/O here.
        self.state = state

    # ---- 1. In-context memory (the live run's AuditState, under state["audit"]) ----
    def read(self, key: str, default=None):
        """Read a value from the audit substate (state['audit'][key])."""
        return (self.state.get("audit") or {}).get(key, default)

    @staticmethod
    def write_audit(**fields) -> dict:
        """Wrap audit-field updates into the nested AMSState shape merge_audit expects."""
        return {"audit": fields}

    @staticmethod
    def append_message(text: str) -> dict:
        """Return a state update that appends one message (merge_audit accumulates it)."""
        return {"audit": {"messages": [text]}}

    # ---- 2. Semantic memory ( similar past PR audits ) ----
    @staticmethod
    def recall_similar_prs(diff_text: str, k: int = cfg.SEARCH_DEFAULT_K) -> list[dict]:
        return vs.retrieve_similar_prs(diff_text, k=k)

    @staticmethod
    def store_pr_audit(pr_summary: str, report: dict, embed_text: str | None = None) -> None:
        vs.store_pr_audit(pr_summary, report, embed_text=embed_text)

    # ---- 3. Episodic memory ( compressed session summaries ) ----
    @staticmethod
    def recall_episodes(query_text: str, k: int = cfg.SEARCH_DEFAULT_K) -> list[dict]:
        return vs.retrieve_episodes(query_text, k=k)

    @staticmethod
    def store_episode(summary: str, metadata: dict | None = None) -> None:
        vs.store_episode(summary, metadata=metadata)

    # ---- 4. Procedural memory ( keyed org rules / templates ) ----
    @staticmethod
    def recall_rules(category: RuleCategory) -> list[str]:
        return vs.get_rules(category)

    @staticmethod
    def add_rule(category: RuleCategory, rule: str, status: RuleStatus) -> None:
        vs.add_rule(category, rule, status)

        # ---- 4b. Procedural rule governance (offline review CLI) ----
    @staticmethod
    def pending_rules() -> list[dict]:
        return vs.list_pending_rules()

    @staticmethod
    def active_rules() -> list[dict]:
        return vs.list_active_rules()

    @staticmethod
    def similar_rules(rule_id: int, k: int = cfg.SEARCH_DEFAULT_K) -> list[dict]:
        return vs.similar_rules(rule_id, k=k)

    @staticmethod
    def approve_rule(rule_id: int) -> None:
        vs.set_rule_status(rule_id, RuleStatus.LEARNED_APPROVED)

    @staticmethod
    def reject_rule(rule_id: int) -> None:
        vs.set_rule_status(rule_id, RuleStatus.REJECTED)    # keep the row -> not re-learned

    @staticmethod
    def retire_rule(rule_id: int) -> None:
        vs.set_rule_status(rule_id, RuleStatus.RETIRED)     # deactivate active rule, keep row

    @staticmethod
    def delete_rule(rule_id: int) -> None:
        vs.delete_rule(rule_id)                             # hard remove

    # Severities important enough to become a standing rule (skip low/info/medium noise).
    _LEARNABLE_SEVERITIES = {"critical", "high"}

    @staticmethod
    def learn_rules_from_findings(security=None, quality=None, coverage=None,
                                  human_decision=None) -> int:
        """Promote THIS run's strongest findings into standing procedural rules so future
        audits enforce them. GUARD-RAILED to avoid rule bloat / a self-reinforcing loop:
          - only CRITICAL/HIGH findings qualify (skip low-signal nits),
          - each finding maps to its domain category (security->security, quality->quality,
            test->coverage),
          - deduped against rules ALREADY stored for that category (case-insensitive), so
            re-auditing the same PR does not keep re-inserting the same rule.
        `human_decision` (this PR's verdict) is stored on each learned rule as provenance AND
        gates learning: a "needs-changes"/"reject" verdict suppresses it entirely (returns 0),
        since rules learned from soon-to-change or abandoned code would pollute future audits.
        "approve" (or a never-escalated run, where human_decision is "n/a"/None) learns as normal.
        Best-effort: any per-rule DB error is swallowed. Returns the count actually added.
        """
        # A "needs-changes"/"reject" verdict means this code is being revised or abandoned -
        # do NOT promote its findings into standing rules. (Precedent + episode are still
        # stored: the deferral is honest history; only LEARNING is suppressed.)
        if str(human_decision or "").strip().lower() in _NO_LEARN:
            return 0
        buckets = ((RuleCategory.SECURITY, security), 
                   (RuleCategory.QUALITY, quality), (RuleCategory.COVERAGE, coverage))
        added = 0
        for category, findings in buckets:
            if not findings:
                continue
            try:
                existing = {r.strip().lower() for r in vs.get_all_rule_contents(category)}
            except Exception:
                existing = set()
            for f in findings:
                if str(f.get("severity", "")).lower() not in AgentMemorySystem._LEARNABLE_SEVERITIES:
                    continue
                rule = str(f.get("description", "")).strip()
                if not rule or rule.lower() in existing:
                    continue
                try:
                    vs.add_rule(category, rule, status=RuleStatus.LEARNED_PENDING,
                                source_decision=human_decision)
                except Exception:
                    continue
                existing.add(rule.lower())   # dedup within this run too
                added += 1
        return added

    @staticmethod
    def recall_all_rules(categories: tuple[RuleCategory, ...]) -> dict[str, list[str]]:
        """Recall org rules for ALL given categories in ONE place (the retrieve node),
        grouped {category: [rules]}, to populate the `procedural` channel. Every other
        node then reads that channel and never re-queries the DB (recall-once pattern,
        mirroring semantic/episodic). Best-effort per category: a category that errors
        is omitted, never blocks the run.
        """
        out: dict[str, list[str]] = {}
        for cat in categories:
            try:
                rules = vs.get_rules(cat)
            except Exception:
                continue
            if rules:
                out[cat.value] = rules
        return out

    @staticmethod
    def rules_block(procedural: dict, categories: tuple[RuleCategory, ...]) -> str:
        """Format the already-recalled rules (from the `procedural` channel) for LITERAL
        injection into a node's system prompt. Reads from the passed-in dict - does NOT
        hit the DB (recall happened once in retrieve). Each node passes its OWN domain
        categories so it enforces only the rules it can act on (security -> security; quality -> quality; coverage -> coverage). Returns "" when no rules match, so
        the {{rules}} placeholder collapses to nothing - no prompt pollution.
        """
        procedural = procedural or {}
        rules: list[str] = []
        for cat in categories:
            rules.extend(procedural.get(cat.value, []))
        if not rules:
            return ""
        return (
            "Standing org rules - you MUST explicitly check each and flag any violation:\n"
            + "\n".join(f"- {r}" for r in rules) + "\n\n"
        )
