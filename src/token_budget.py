# TokenBudgetManager: explicit context-budget allocation. Keeps highest-priority content
# first, trims lowest-priority when over budget and logs every trim (never silent).
# Generic by design - operates on labelled text segments, knows nothing about AuditState.
from dataclasses import dataclass

@dataclass
class Segment:
    priority: int          # lower number = higher priority (0 = never trim)
    label: str             # e.g. "system_prompt", "query", "chunk:auth.py", "history:msg3"
    text: str

# Cheap, dependency-free token estimate. VERIFIED: the google-genai client exposes no LOCAL
# token counter (its count_tokens is a network API call, not a local tokenizer like tiktoken),
# so a local heuristic avoids an extra API round-trip just to make a budgeting decision.
# ~4 chars/token is the standard rough heuristic and is fine here (we're deciding what to drop,
# not billing). The `counter` param lets you swap in client.models.count_tokens later if needed.
def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)

class TokenBudgetManager:
    def __init__(self, budget_tokens: int, counter=estimate_tokens):
        self.budget = budget_tokens
        self.counter = counter
    
    def fit(self, segments: list[Segment]) -> tuple[list[Segment], list[str]]:
        """
        Return (kept_segments, trim_log). Keeps segments by ascending priority until the
        budget is exhausted; drops the rest. Priority ties keep input order. Anything with
        priority 0 is mandatory and kept even if it alone exceeds the budget (logged as a warning).
        kept_segments are returned in ORIGINAL input order.
        """
        # Decide keep/drop in priority order, but track by INDEX (not by value) so that two
        # segments with identical content/labels are handled independently - a dataclass
        # generates __eq__, so `seg in kept` would conflate duplicate-content segments.
        ordered = sorted(range(len(segments)), key=lambda i: (segments[i].priority, i))
        keep_idx, log, used = set(), [], 0
        for i in ordered:
            seg = segments[i]
            cost = self.counter(seg.text)
            if seg.priority == 0:
                keep_idx.add(i)
                used += cost
                if used > self.budget:
                    log.append(f"WARNING: mandatory '{seg.label}' ({cost} tok) pushes "
                               f"usage to {used}/{self.budget} - over budget but kept.")
                continue
            if used + cost <= self.budget:
                keep_idx.add(i)
                used += cost
            else:
                log.append(f"trimmed '{seg.label}' ({cost} tok) - would exceed "
                           f"{self.budget} (at {used}).")
        kept_in_order = [segments[i] for i in range(len(segments)) if i in keep_idx]
        return kept_in_order, log