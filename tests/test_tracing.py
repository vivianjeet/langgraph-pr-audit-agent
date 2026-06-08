# Tracing is best-effort and OFF the fail-closed path: no keys -> no-op, and a tracing error must
# NEVER propagate into an audit (unlike a provider outage, which must fail closed). These tests pin
# that boundary, plus the cost-by-type breakdown the dashboard reads.
from unittest.mock import patch
import src.llm_client as lc


def _result():
    return lc.LLMResult(output="x", model="m", backend="gemini",
                        input_tokens=100, output_tokens=20, cost_usd=0.001)


def test_no_keys_means_no_trace(monkeypatch):
    # Unconfigured Langfuse -> _trace is a silent no-op (conftest already strips the keys, but be
    # explicit). No client is built, nothing is sent, nothing raises.
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    lc._LANGFUSE = None
    lc._trace(_result(), "fast", 0.1)


def test_trace_failure_is_swallowed():
    # A live-but-broken client must not break the audit. _Boom raises from the v4 call path
    # (start_as_current_observation - NOT the dead v2 .generation()), and _trace swallows it.
    class _Boom:
        def start_as_current_observation(self, **k):
            raise RuntimeError("langfuse down")
    with patch.object(lc, "_langfuse", return_value=_Boom()):
        lc._trace(_result(), "fast", 0.1)  # must not raise


def test_score_audit_is_noop_without_client():
    # Scores are best-effort too: no client -> no-op, no raise.
    with patch.object(lc, "_langfuse", return_value=None):
        lc.score_audit({"security_score": 0.4, "quality_score": 1.0, "test_score": 0.5})


def test_score_audit_failure_is_swallowed():
    class _Boom:
        def score_current_trace(self, **k):
            raise RuntimeError("scores down")
    with patch.object(lc, "_langfuse", return_value=_Boom()):
        lc.score_audit({"security_score": 0.4})  # must not raise


def test_flush_traces_is_noop_without_client():
    with patch.object(lc, "_langfuse", return_value=None):
        lc.flush_traces()  # no client -> no-op, no raise


def test_audit_trace_is_noop_without_client():
    # The context manager must still yield (so callers can always `with` it) when unconfigured.
    with patch.object(lc, "_langfuse", return_value=None):
        with lc.audit_trace("thread-1", label="cli:test"):
            pass  # body runs; no trace opened


def test_cost_breakdown_is_all_positive_and_sums_to_total():
    # The cache-read line must be a real (positive) cost, not a negative "saving" - a negative
    # cost_details line is invalid for the dashboard. And the parts must sum to the total, which
    # must equal _price (the authoritative cost the rest of the system uses).
    model = lc.cfg.GEMINI_PRO_MODEL
    b = lc._price_breakdown(model, in_tok=1000, out_tok=200, cache_read=400)
    assert all(v >= 0 for v in b.values())
    assert abs((b["input"] + b["cache_read"] + b["output"]) - b["total"]) < 1e-9
    assert abs(b["total"] - lc._price(model, 1000, 200, cache_read=400)) < 1e-9


def test_cost_breakdown_no_cache():
    # With no cache reads the cache_read line is zero and total still matches _price.
    model = lc.cfg.GEMINI_FLASH_MODEL
    b = lc._price_breakdown(model, in_tok=500, out_tok=100, cache_read=0)
    assert b["cache_read"] == 0.0
    assert abs(b["total"] - lc._price(model, 500, 100)) < 1e-9
