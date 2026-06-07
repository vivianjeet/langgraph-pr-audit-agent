import logging
import hashlib
from dataclasses import dataclass
from src.llm_retry import (
    call_cached_generate, call_gemini, call_gemini_async, QuotaExhaustedError,
)
import src.config as cfg

_CACHE_HANDLES: dict[str, str] = {}  
log = logging.getLogger(__name__)

_PRICES = {
    cfg.GEMINI_FLASH_LITE_MODEL:       (0.10,  0.40),
    cfg.GEMINI_FLASH_MODEL:            (0.30,  2.50),
    cfg.GEMINI_PRO_MODEL:              (1.25,  10.00),
}

@dataclass
class LLMResult:
    """A result: the parsed/text output + accounting for the cost dashboard"""
    output: object      # pydantic model (response_model) or str
    model: str
    backend: str = "gemini"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    fell_back_from: str | None = None # the tier we tried first, if we fell back

@dataclass(frozen=True)
class _Tier:
    model: str

TIER_TABLE = {
    "fast": _Tier(cfg.GEMINI_FLASH_LITE_MODEL),
    "balanced": _Tier(cfg.GEMINI_FLASH_MODEL),
    "powerful": _Tier(cfg.GEMINI_PRO_MODEL),
    "cite" : _Tier(cfg.CITE_MODEL)
}

_FALLBACK = ["balanced", "fast"]

def _price(model: str, in_tok: int, out_tok: int, cache_read: int = 0) -> float:
    pin, pout = _PRICES.get(model, (0.0,0.0))
    # cached input is ~25% of base input price (Gemini context cache); approximate for the dashboard.
    billable_in = (in_tok - 0.75*cache_read)
    return (billable_in * pin + out_tok * pout) / 1_000_000

async def _acall_gemini(model, messages, response_model, max_output_tokens) -> LLMResult:
    out = await call_gemini_async(model=model, messages=messages,
                                  response_model=response_model,
                                  max_output_tokens=max_output_tokens)
    return LLMResult(output=out, model=model)


def _call_gemini(model, messages, response_model, max_output_tokens) -> LLMResult:
    """Sync twin of _acall_gemini: the plain (non-cache) chat call through the sync spine."""
    out = call_gemini(model=model, messages=messages,
                      response_model=response_model, max_output_tokens=max_output_tokens)
    return LLMResult(output=out, model=model)


def _resolve_chain(tier: str, special: bool):
    """The tier order to try: just `tier` for the special (cache/thinking) paths - those are
    deliberate, tier-specific calls that must NOT silently fall back to a cheaper model - else
    `tier` followed by the fallback chain. Shared by acall and call."""
    return [tier] if special else [tier] + [t for t in _FALLBACK if t != tier]


def _exhausted(last_err, tier):
    """Build the terminal error after every tier failed. Fail-closed: a quota exhaustion reaches the
    node AS QuotaExhaustedError (nodes branch on it to abort, not degrade to a clean score); masking
    it as RuntimeError would let a generic except swallow it into a false-clean path. Shared by
    acall and call so both call shapes honour the same fail-closed contract."""
    if isinstance(last_err, QuotaExhaustedError):
        return last_err
    return RuntimeError(f"all LLM tiers failed (last: {type(last_err).__name__} : {last_err})")

def cached_system(stable: str, *, label: str = "audit-system") -> dict:
    """Package a stable system prefix for the cache path. The router pulls `content` out as the
    text registered into the CachedContent; the per-run diff is sent separately as the user message."""
    return {"role": "system", "content": stable, "_cache_label": label}


def cached_diff(diff: str, *, label: str = "audit-diff") -> dict:
    """Package the PR diff as the cached part. Mirror of cached_system but for the OTHER cache axis:
    within ONE audit the diff is identical across the FLASH nodes (compliance/plan/quality/coverage)
    while each node's instructions differ - so caching the diff once and reusing it across those nodes
    is the per-PR win. compliance primes it (runs first), the rest reuse. NOTE: security is NOT in this
    set - it's on Pro (model-bound cache) and uses the prefix axis (cached_system) for the cross-PR
    batch case instead. Same packaging shape as cached_system: messages[0] is cached, messages[1]
    varies."""
    return {"role": "system", "content": diff, "_cache_label": label}

def _cached_call(model, stable_system, user_content, max_output_tokens,
                 response_schema=None) -> LLMResult:
    """SYNC core of the context-cache call: register the stable part as a CachedContent once (keyed
    by model+prefix, scoped to the live key inside call_cached_generate), reuse the handle later,
    generate and shape the LLMResult. cached_content_token_count on a repeat call proves reuse. Both
    create + generate run through the spine's rotation (one unit, same key). `response_schema` asks
    Gemini for native JSON so `res.output` parses (the cache path can't use Instructor). Async callers
    wrap this in asyncio.to_thread (the only reason a sync core exists); sequential callers run it
    directly - both share THIS body and the same _CACHE_HANDLES, so they hit the same handle."""
    from google.genai import types

    cache_key_base = hashlib.sha256(f"{model}:{stable_system}".encode()).hexdigest()

    def _build_config(handle):
        gen_cfg = dict(cached_content=handle, max_output_tokens=max_output_tokens)
        if response_schema is not None:
            gen_cfg["response_mime_type"] = "application/json"
            gen_cfg["response_schema"] = response_schema
        return types.GenerateContentConfig(**gen_cfg)

    # ONE rotation unit owns create-if-needed + generate + stale-handle recovery (see llm_retry):
    # keeping the handle and the inference on the same key is why this isn't two rotation calls.
    resp = call_cached_generate(model, user_content, stable_system, "300s",
                                _build_config, _CACHE_HANDLES, cache_key_base)
    um = resp.usage_metadata
    # Gemini can return None for these on a structured/cached call (e.g. candidates_token_count when
    # the JSON schema path emits no separate candidate count) - coalesce to 0 so _price never gets a
    # None (a None silently crashed _price -> the node's except swallowed it as a Flash fallback).
    cache_read = getattr(um, "cached_content_token_count", 0) or 0
    in_tok = um.prompt_token_count or 0
    out_tok = um.candidates_token_count or 0
    return LLMResult(
        output=resp.text, model=model,
        input_tokens=in_tok, output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cost_usd=_price(model, in_tok, out_tok, cache_read=cache_read),
    )


async def _acall_cached(model, stable_system, user_content, max_output_tokens,
                        response_schema=None) -> LLMResult:
    """Async entry to the cache path: run the sync core on a worker thread so the parallel fan-out
    (security/quality/coverage) can overlap their Gemini I/O. Reuses _cached_call unchanged - we only
    move the blocking call off the event loop."""
    import asyncio
    return await asyncio.to_thread(
        _cached_call, model, stable_system, user_content, max_output_tokens, response_schema)

class UnifiedLLMClient:
    """The router. `acall(tier=...)` picks a Gemini model from TIER_TABLE, executes through the spine,
    and on failure walks the fallback chain - recording which tier actually served the call.

    Two optional flags select a Gemini-native call shape:
      - cache=True     -> `messages[0]` MUST be a system block list
                          built by cached_system() so the stable prefix is registered as a cache.
      - thinking=N     -> thinking-budget path with budget N reasoning tokens.
    Fallback is DISABLED when cache/thinking is requested: those are deliberate, tier-specific calls
    (you asked for Pro+thinking), so silently falling back to a cheaper model would defeat the point -
    it raises instead, staying fail-closed and honest about which tier ran."""

    async def acall(self, tier: str = "fast", *, messages: list,  response_model=None,
                    max_output_tokens: int = 2000, cache: bool = False, thinking: int = 0) -> LLMResult:
        special = cache or thinking
        last_err = None
        for i, t in enumerate(_resolve_chain(tier, special)):
            spec = TIER_TABLE.get(t)
            if spec is None:
                continue
            try:
                # messages for the special paths are [system_blocks, {'role':'user','content':...}]
                if thinking:
                    res = await _acall_thinking(spec.model, messages[0]["content"],
                                                messages[1]["content"], max_output_tokens,thinking)
                elif cache:
                    res = await _acall_cached(spec.model, messages[0]["content"],
                                              messages[1]["content"], max_output_tokens,
                                              response_schema=response_model)
                else:
                    res = await _acall_gemini(spec.model, messages, response_model, max_output_tokens)

                if i > 0:
                    res.fell_back_from = tier
                    log.warning("LLM tier '%s' failed; served by fallback '%s'", tier, t)
                _trace(res, tier, 0.0)
                return res
            except Exception as e:
                last_err = e
                if special:
                    break
        raise _exhausted(last_err, tier)

    def call(self, tier: str = "fast", *, messages: list, response_model=None,
             max_output_tokens: int = 2000) -> LLMResult:
        """Sync twin of acall for the SEQUENTIAL nodes (plan/reflexion/compliance) that don't need the
        event loop - same tier selection + fallback chain + fail-closed contract, on the sync spine.
        No cache/thinking flags: those are async-only (they ride asyncio.to_thread), and the
        sequential nodes don't cache anyway."""
        last_err = None
        for i, t in enumerate(_resolve_chain(tier, special=False)):
            spec = TIER_TABLE.get(t)
            if spec is None:
                continue
            try:
                res = _call_gemini(spec.model, messages, response_model, max_output_tokens)
                if i > 0:
                    res.fell_back_from = tier
                    log.warning("LLM tier '%s' failed; served by fallback '%s'", tier, t)
                _trace(res, tier, 0.0)
                return res
            except Exception as e:
                last_err = e
        raise _exhausted(last_err, tier)

# Forward-reference stubs yo be filled
async def _acall_thinking(*a, **k):
    raise NotImplementedError

def _trace(*a, **k):
    pass


llm = UnifiedLLMClient()   # THE shared router singleton - every node calls llm.acall(tier=...)


def _diff_cache_note(res) -> str:
    return (f"Cache(diff): read={res.cache_read_tokens} input={res.input_tokens} "
            f"output={res.output_tokens} cost=${res.cost_usd:.6f}\n")


# Shared docstring for the two diff-cache helpers. They cache the DIFF (the part identical across the
# Flash diff-nodes) and vary the instructions. compliance runs FIRST and primes the handle (so the
# later nodes are pure reusers, no parallel create-race); plan/quality/coverage then reuse it. All four
# share ONE handle (keyed by model+diff) because they share the Flash model - a CachedContent is
# model-bound, so security (Pro) can't join. On ANY cache failure (e.g. diff below Gemini's ~2048-token
# floor) both fall back to a plain Flash call (no cache, no note) so callers never branch.
# QuotaExhaustedError propagates (fail-closed). Sync twin exists for the SEQUENTIAL nodes (compliance is
# async/fan-out-adjacent, plan is sync) - same cache, just with/without the event loop.

async def audit_with_diff_cache(diff, instructions, response_model, max_output_tokens):
    """Async diff-cache call for the fan-out + async nodes (compliance/quality/coverage). See module
    note above _diff_cache_note for the shared design. Returns (parsed_response, cache_note)."""
    try:
        res = await _acall_cached(cfg.GEMINI_FLASH_MODEL, diff, instructions,
                                  max_output_tokens, response_schema=response_model)
        return response_model.model_validate_json(res.output), _diff_cache_note(res)
    except QuotaExhaustedError:
        raise
    except Exception:
        out = await call_gemini_async(model=cfg.GEMINI_FLASH_MODEL,
                                      messages=[{"role": "user", "content": instructions + "\n\n" + diff}],
                                      response_model=response_model,
                                      max_output_tokens=max_output_tokens)
        return out, ""


def audit_with_diff_cache_sync(diff, instructions, response_model, max_output_tokens):
    """SYNC twin of audit_with_diff_cache for the sequential plan node (no event loop needed - it runs
    alone). Hits the SAME _cached_call core and the SAME _CACHE_HANDLES, so it reuses whatever handle
    compliance primed. Returns (parsed_response, cache_note)."""
    try:
        res = _cached_call(cfg.GEMINI_FLASH_MODEL, diff, instructions,
                           max_output_tokens, response_schema=response_model)
        return response_model.model_validate_json(res.output), _diff_cache_note(res)
    except QuotaExhaustedError:
        raise
    except Exception:
        out = call_gemini(model=cfg.GEMINI_FLASH_MODEL,
                          messages=[{"role": "user", "content": instructions + "\n\n" + diff}],
                          response_model=response_model,
                          max_output_tokens=max_output_tokens)
        return out, ""
