"""Per-tier LLM spend + fallback events, read back from Langfuse.

Pairs with the router's single trace callback (src/llm_client._trace): every call is logged as a
generation tagged with its tier and any fallback. This pulls those generations and rolls them up by
tier - how many calls, how much spend, and how often a tier fell back to a cheaper one. Falls back
cleanly to "no data" when Langfuse isn't configured - the report is an optional view, not a
dependency (observability is off the critical path).

Usage:
    python -m scripts.cost_report            # last 500 generations
    python -m scripts.cost_report --limit 2000
"""
import argparse
from collections import defaultdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _client():
    """The Langfuse client if keys are set, else None. Mirrors llm_client._langfuse so the report
    is a no-op without keys rather than a crash."""
    import os
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return None
    try:
        from langfuse import Langfuse
        return Langfuse()
    except Exception:
        return None


def _gen_cost(o) -> float:
    """Best-effort spend for one generation. Prefer Langfuse's own cost_details.total; if the list
    projection didn't return it, recompute from the model + token counts with the router's price
    table (the same numbers _trace logged), so the rollup is accurate either way."""
    cd = getattr(o, "cost_details", None)
    if isinstance(cd, dict) and cd.get("total") is not None:
        return float(cd["total"])
    ud = getattr(o, "usage_details", None) or {}
    model = getattr(o, "model", None)
    if model and ud:
        from src.llm_client import _price
        return _price(model, int(ud.get("input", 0)), int(ud.get("output", 0)),
                      cache_read=int(ud.get("cache_read", 0)))
    return 0.0


def report(limit: int):
    lf = _client()
    if lf is None:
        print("Langfuse not configured (set LANGFUSE_*); no cost data.")
        return

    from langfuse.api.core.request_options import RequestOptions
    opts = RequestOptions(timeout_in_seconds=60)
    fields = "name,model,costDetails,usageDetails,metadata"

    by_tier = defaultdict(lambda: {"calls": 0, "cost": 0.0, "fallbacks": 0})
    fetched, cursor = 0, None
    while fetched < limit:
        page = lf.api.observations.get_many(type="GENERATION", limit=min(100, limit - fetched),
                                            cursor=cursor, fields=fields, request_options=opts)
        if not page.data:
            break
        for o in page.data:
            meta = getattr(o, "metadata", None) or {}
            tier = meta.get("tier", "unknown")
            row = by_tier[tier]
            row["calls"] += 1
            row["cost"] += _gen_cost(o)
            if meta.get("fell_back_from"):
                row["fallbacks"] += 1
        fetched += len(page.data)
        cursor = getattr(getattr(page, "meta", None), "cursor", None)
        if not cursor:
            break

    if not by_tier:
        print("No generations logged yet - run an audit with LANGFUSE_* set.")
        return

    print(f"{'tier':22s}{'calls':>7s}{'cost$':>10s}{'fallbk':>8s}")
    print("-" * 47)
    for tier in sorted(by_tier):
        r = by_tier[tier]
        print(f"{tier:22s}{r['calls']:>7d}{r['cost']:>10.4f}{r['fallbacks']:>8d}")
    tot_calls = sum(r["calls"] for r in by_tier.values())
    tot_cost = sum(r["cost"] for r in by_tier.values())
    tot_fb = sum(r["fallbacks"] for r in by_tier.values())
    print("-" * 47)
    print(f"{'TOTAL':22s}{tot_calls:>7d}{tot_cost:>10.4f}{tot_fb:>8d}")


def main():
    ap = argparse.ArgumentParser(description="Per-tier LLM spend + fallback events from Langfuse.")
    ap.add_argument("--limit", type=int, default=500, help="how many recent generations to scan")
    args = ap.parse_args()
    report(args.limit)


if __name__ == "__main__":
    main()
