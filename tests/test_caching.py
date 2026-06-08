# the cached-system payload + cost accounting on a context-cache read.
from unittest.mock import patch
import asyncio
import src.llm_client as lc
import src.llm_retry as lr


def test_cached_system_packages_the_prefix():
    block = lc.cached_system("RULES + COMPLIANCE + INSTRUCTIONS")
    assert block["role"] == "system"
    assert block["content"].startswith("RULES")
    assert block["_cache_label"] == "audit-system"


def _resp(cache_read):
    um = type("U", (), {"prompt_token_count": 1000, "candidates_token_count": 50,
                        "cached_content_token_count": cache_read})
    return type("R", (), {"usage_metadata": um, "text": "x"})


def test_cache_read_lowers_cost():
    # same 1000 input tokens, but 900 served from the context cache -> cheaper than 0 cached.
    # The single rotation unit (call_cached_generate) is real; we patch the RAW SDK calls it drives.
    with patch.object(lr, "_raw_create_cache", lambda *a, **k: "cachedContents/abc"):
        with patch.object(lr, "_raw_cached_generate", lambda *a, **k: _resp(900)):
            lc._CACHE_HANDLES.clear()
            hot = asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-A", "diff", 500))
        with patch.object(lr, "_raw_cached_generate", lambda *a, **k: _resp(0)):
            lc._CACHE_HANDLES.clear()
            cold = asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-B", "diff", 500))
    assert hot.cache_read_tokens == 900
    assert hot.cost_usd < cold.cost_usd              # the cache read is cheaper, provably


def test_stale_handle_is_evicted_and_recreated():
    # A stale/cross-project handle returns the verified Gemini 403 "CachedContent not found".
    # The rotation unit must evict the dead handle and re-create ONCE, then succeed - not propagate.
    created = {"n": 0}
    def _create(*a, **k):
        created["n"] += 1
        return f"cachedContents/{created['n']}"
    calls = {"n": 0}
    def _generate(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("403 PERMISSION_DENIED CachedContent not found (or permission denied)")
        return _resp(900)                            # 2nd attempt (fresh handle) succeeds
    with patch.object(lr, "_raw_create_cache", _create), \
         patch.object(lr, "_raw_cached_generate", _generate):
        lc._CACHE_HANDLES.clear()
        res = asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-A", "diff", 500))
    assert calls["n"] == 2                            # it retried once after the stale 403
    assert created["n"] == 2                          # and re-created the handle on the retry
    assert res.cache_read_tokens == 900               # the recovered call returned the cached read


def test_non_stale_error_propagates_not_retried():
    # Any OTHER error on generate must NOT be swallowed as a stale handle - it propagates.
    def _generate(*a, **k):
        raise RuntimeError("500 INTERNAL server error")
    with patch.object(lr, "_raw_create_cache", lambda *a, **k: "cachedContents/x"), \
         patch.object(lr, "_raw_cached_generate", _generate):
        lc._CACHE_HANDLES.clear()
        try:
            asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-A", "diff", 500))
            assert False, "expected the non-stale error to propagate"
        except RuntimeError as e:
            assert "INTERNAL" in str(e)


def test_handle_reused_across_runs_on_same_key():
    # The handle is keyed by the LIVE key index, so two runs on the same key reuse ONE handle
    # (regression: keying on the pre-call key created a 2nd handle after the first rotation).
    created = {"n": 0}
    def _create(*a, **k):
        created["n"] += 1
        return f"cachedContents/{created['n']}"
    with patch.object(lr, "_key_idx", 0), \
         patch.object(lr, "_raw_create_cache", _create), \
         patch.object(lr, "_raw_cached_generate", lambda *a, **k: _resp(900)):
        lc._CACHE_HANDLES.clear()
        asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-A", "diff one", 500))
        asyncio.run(lc._acall_cached("gemini-2.5-flash", "sys-A", "diff two", 500))
    assert created["n"] == 1                          # created ONCE, reused on the 2nd run
    assert len(lc._CACHE_HANDLES) == 1


def test_diff_cache_helper_caches_diff_and_returns_note():
    # audit_with_diff_cache passes the DIFF as the cached part (arg 1) and instructions as the
    # variable part (arg 2) into _acall_cached; on success returns (parsed, note).
    from pydantic import BaseModel
    class _Out(BaseModel):
        ok: bool
    async def _acall_cached(model, stable, user_content, max_tok, response_schema=None):
        assert stable == "THE-DIFF"                      # diff is the cached part
        assert user_content == "INSTRUCTIONS"            # instructions vary
        return lc.LLMResult(output='{"ok": true}', model="gemini-2.5-flash",
                            cache_read_tokens=900, input_tokens=1000, output_tokens=10)
    with patch.object(lc, "_acall_cached", side_effect=_acall_cached):
        parsed, note = asyncio.run(lc.audit_with_diff_cache("THE-DIFF", "INSTRUCTIONS", _Out, 500))
    assert parsed.ok is True
    assert "Cache(diff): read=900" in note


def test_diff_cache_helper_falls_back_when_cache_fails():
    # diff too small / any cache error -> plain Flash, empty note, identical result shape.
    from pydantic import BaseModel
    class _Out(BaseModel):
        ok: bool
    async def _acall_cached(*a, **k):
        raise RuntimeError("400 Cached content is too small")
    async def _flash(*a, **k):
        return _Out(ok=True)
    with patch.object(lc, "_acall_cached", side_effect=_acall_cached), \
         patch.object(lc, "call_gemini_async", side_effect=_flash):
        parsed, note = asyncio.run(lc.audit_with_diff_cache("small", "INSTRUCTIONS", _Out, 500))
    assert parsed.ok is True
    assert note == ""                                    # no cache note on the fallback path


def test_diff_cache_sync_helper_reuses_same_core():
    # The SYNC twin (plan) hits the same _cached_call core + same _CACHE_HANDLES as the async one,
    # so it reuses whatever handle compliance primed. Here we just prove it returns (parsed, note).
    from pydantic import BaseModel
    class _Out(BaseModel):
        ok: bool
    def _cached_call(model, stable, user_content, max_tok, response_schema=None):
        assert stable == "THE-DIFF" and user_content == "INSTR"
        return lc.LLMResult(output='{"ok": true}', model="gemini-2.5-flash",
                            cache_read_tokens=700, input_tokens=800, output_tokens=10)
    with patch.object(lc, "_cached_call", side_effect=_cached_call):
        parsed, note = lc.audit_with_diff_cache_sync("THE-DIFF", "INSTR", _Out, 500)
    assert parsed.ok is True
    assert "Cache(diff): read=700" in note


def test_router_cache_flag_routes_to_cached_path():
    # acall(cache=True) must reach _acall_cached through the public method.
    async def _cached(*a, **k):
        return lc.LLMResult(output="ok", model="gemini-2.5-pro", cache_read_tokens=500)
    with patch.object(lc, "_acall_cached", side_effect=_cached) as c:
        res = asyncio.run(lc.UnifiedLLMClient().acall(
            tier="powerful", cache=True,
            messages=[lc.cached_system("RULES"), {"role": "user", "content": "diff"}]))
    c.assert_awaited_once()                          # the flag is wired through the public method
    assert res.cache_read_tokens == 500