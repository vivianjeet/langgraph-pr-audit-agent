# the router's tier selection, fallback and fail-closed contract.
from unittest.mock import patch, AsyncMock
import asyncio
import pytest
import src.llm_client as lc

def test_fast_tier_routes_through_the_spine():
    with patch.object(lc, "call_gemini_async", AsyncMock(return_value="ok")) as g:
        res = asyncio.run(lc.UnifiedLLMClient().acall(tier="fast", messages=[{"role": "user", "content": "x"}]))
    assert res.output == "ok"
    g.assert_awaited_once()                          # the spine served the call
    # the fast tier maps to the cheapest model
    assert g.await_args.kwargs["model"] == lc.TIER_TABLE["fast"].model


def test_powerful_tier_selects_the_pro_model():
    with patch.object(lc, "call_gemini_async", AsyncMock(return_value="hi")) as g:
        res = asyncio.run(lc.UnifiedLLMClient().acall(tier="powerful", messages=[{"role": "user", "content": "x"}]))
    assert g.await_args.kwargs["model"] == "gemini-2.5-pro"   # the tier table chose the model
    assert res.model == "gemini-2.5-pro"


def test_fallback_records_the_origin_tier():
    # balanced raises quota -> chain drops to fast, which succeeds. Both go through the spine.
    calls = {"n": 0}
    async def _spine(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise lc.QuotaExhaustedError("balanced out of quota")
        return "served-by-fast"
    with patch.object(lc, "call_gemini_async", side_effect=_spine):
        res = asyncio.run(lc.UnifiedLLMClient().acall(tier="balanced", messages=[{"role": "user", "content": "x"}]))
    assert res.output == "served-by-fast"
    assert res.fell_back_from == "balanced"          # the fallback is VISIBLE on the result


def test_all_tiers_failing_on_quota_reraises_quota_exhausted():
    # Fail-closed: when every tier is quota-exhausted, the router must re-raise QuotaExhaustedError
    # AS ITSELF (not mask it as a generic RuntimeError), so a node's `except QuotaExhaustedError`
    # still fires and aborts instead of degrading to a false-clean score.
    with patch.object(lc, "call_gemini_async", AsyncMock(side_effect=lc.QuotaExhaustedError("x"))):
        with pytest.raises(lc.QuotaExhaustedError):
            asyncio.run(lc.UnifiedLLMClient().acall(tier="balanced", messages=[{"role": "user", "content": "x"}]))


def test_all_tiers_failing_on_other_error_raises_runtimeerror():
    # A NON-quota failure across all tiers still raises (never returns a fabricated result), but as the
    # generic RuntimeError - only quota gets the special fail-closed type.
    with patch.object(lc, "call_gemini_async", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(RuntimeError, match="all LLM tiers failed"):
            asyncio.run(lc.UnifiedLLMClient().acall(tier="balanced", messages=[{"role": "user", "content": "x"}]))