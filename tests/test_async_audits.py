"""Tests: the 3 audit nodes are async and overlap their LLM calls"""
import asyncio, inspect, time
from unittest.mock import patch
import src.nodes.security_audit as sec
import src.nodes.quality_audit as qual
import src.nodes.coverage_audit as cov

def test_audit_nodes_are_coroutines():
    for mod, fn in ((sec, "security_audit_node"), (qual, "quality_audit_node"),
                    (cov, "coverage_audit_node")):
        assert inspect.iscoroutinefunction(getattr(mod, fn)), f"{fn} must be async"

def test_calls_actually_overlap():
    # Each node's LLM call sleeps 0.3s on a worker thread; run all three concurrently and assert
    # total < 3x (proves overlap, not sequential). Uses the real call_gemini_async (to_thread).
    class _Out:
        reasoning = "r"; findings = []
    state = {"audit": {"parsed_diff": "diff --git a/x b/x\n+code", "audit_plan": {"focus_areas": []}},
             "procedural": {}}
    async def _run():
        # call_gemini_async is patched below (in the caller's `with`) so no real LLM is hit;
        # call the three coroutines via gather and time them.
        t0 = time.perf_counter()
        await asyncio.gather(sec.security_audit_node(state),
                             qual.quality_audit_node(state),
                             cov.coverage_audit_node(state))
        return time.perf_counter() - t0
    # NOTE: patch call_gemini_async in each module so no real LLM is hit:
    async def _fake_async(*a, **k):
        await asyncio.sleep(0.3); return _Out()
    with patch.object(sec, "call_gemini_async", _fake_async), \
         patch.object(qual, "call_gemini_async", _fake_async), \
         patch.object(cov, "call_gemini_async", _fake_async):
        elapsed = asyncio.run(_run())
    assert elapsed < 0.7, f"expected overlap (<0.7s), got {elapsed:.2f}s"   # 3x0.3=0.9 if serial
    
def test_concurrent_failure_rotates_exactly_once():
    """Three concurrent calls all hit a daily-exhausted KEY1. The lock + double-check must rotate
    ONCE to KEY2 (not burn KEY3/KEY4), serve all three and never run past the pool."""
    import asyncio
    from unittest.mock import patch
    import src.llm_retry as lr

    # 4 keys so an UN-locked impl would visibly over-rotate (KEY1→2→3→4); locked impl stops at KEY2.
    with patch.object(lr, "_KEYS", ["KEY1", "KEY2", "KEY3", "KEY4"]):
        lr._key_idx = 0
        lr._refresh_clients()

        class _Out:
            reasoning = "r"; findings = []

        def _raw(model, messages, response_model, max_output_tokens):
            if lr.current_key() == "KEY1":     # only the first key is daily-exhausted
                raise Exception("RESOURCE_EXHAUSTED ... quota ... PerDay ... retryDelay: 999s")
            return _Out()

        async def _run():
            with patch.object(lr, "_raw_chat", _raw):
                return await asyncio.gather(
                    lr.call_gemini_async("m", [], _Out, 10),
                    lr.call_gemini_async("m", [], _Out, 10),
                    lr.call_gemini_async("m", [], _Out, 10),
                )

        results = asyncio.run(_run())
        assert len(results) == 3               # all three served
        assert lr._key_idx == 1                # rotated EXACTLY once (KEY1→KEY2); KEY3/KEY4 untouched