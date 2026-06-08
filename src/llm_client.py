import logging
import hashlib
from dataclasses import dataclass
from src.llm_retry import (
    call_cached_generate, call_gemini, call_gemini_async, call_thinking, QuotaExhaustedError,
)
import src.config as cfg
import os, time
from contextvars import ContextVar
from opentelemetry import context as _otel_ctx

_LANGFUSE = None

# The OpenTelemetry context active inside the audit's parent span. audit_trace() captures it;
# _trace() re-attaches it so each LLM-call generation becomes a TRUE child of the parent span -
# even when it runs in a different LangGraph node task or the cache path's to_thread worker (OTEL's
# "current span" is contextvar-based and doesn't auto-propagate across those hops). Carrying the
# whole context (not just the trace_id) is what keeps the parent as the real root - pinning by
# trace_id alone made the trace take a child's name and spawned an empty duplicate.
_AUDIT_OTEL_CTX: ContextVar[object | None] = ContextVar("audit_otel_ctx", default=None)

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

def _langfuse():
    """Lazily build the Langfuse client IF keys are set. Tracing is best-effort: no keys -> no-op,
    a tracing failure NEVER breaks an audit (observability is not on the fail-closed path)."""
    global _LANGFUSE
    if _LANGFUSE is None and os.environ.get("LANGFUSE_PUBLIC_KEY"):
        try:
            from langfuse import Langfuse
            _LANGFUSE = Langfuse()
        except Exception as e:
            log.warning("Langfuse init failed (%s);  tracing disabled.", e)
            _LANGFUSE = False
    return _LANGFUSE or None

def _trace(res: LLMResult, tier: str, latency_s: float):
    lf = _langfuse()
    if lf is None:
        return
    # Re-attach the audit's OTEL context so this generation becomes a TRUE child of the parent
    # span (implicit context doesn't cross LangGraph's per-node tasks / the to_thread hop). Outside
    # an audit there's no captured context -> attach is skipped -> the call gets its own trace.
    saved = _AUDIT_OTEL_CTX.get()
    otel_token = _otel_ctx.attach(saved) if saved is not None else None
    try:
        with lf.start_as_current_observation(
            name=f"audit-llm:{tier}",
            as_type="generation",
            model=res.model,
            usage_details={"input": res.input_tokens, "output": res.output_tokens,
                           "cache_read": res.cache_read_tokens},
            cost_details=_price_breakdown(res.model, res.input_tokens, res.output_tokens,
                                          res.cache_read_tokens),
            metadata={"backend": res.backend,
                      "tier": tier,
                      "cache_read_tokens": res.cache_read_tokens,
                      "fell_back_from": res.fell_back_from,
                      "latency_s": round(latency_s, 3)},
        ):
            pass
    except Exception as e:
        log.warning("Langfuse trace failed (%s); continuing.", e)
    finally:
        if otel_token is not None:
            _otel_ctx.detach(otel_token)

def _price(model: str, in_tok: int, out_tok: int, cache_read: int = 0) -> float:
    pin, pout = _PRICES.get(model, (0.0,0.0))
    # cached input is ~25% of base input price (Gemini context cache); approximate for the dashboard.
    billable_in = (in_tok - 0.75*cache_read)
    return (billable_in * pin + out_tok * pout) / 1_000_000


def _price_breakdown(model: str, in_tok: int, out_tok: int, cache_read: int = 0) -> dict:
    """Same arithmetic as _price, split into components so the dashboard shows cost BY TYPE.
    All lines are POSITIVE actual costs that sum to total (Langfuse cost_details must not be
    negative). cache_read = the cached portion of input, billed at ~25% of base; it shows as its
    own (small, because discounted) line so the cache's value is visible. in_tok includes the
    cached tokens, so the plain-input line prices only the NON-cached remainder. Matches _price."""
    pin, pout = _PRICES.get(model, (0.0, 0.0))
    non_cached_in = max(in_tok - cache_read, 0)
    input_cost  = non_cached_in * pin / 1_000_000          # input NOT served from cache, full price
    cache_cost  = cache_read * 0.25 * pin / 1_000_000      # cached input at the ~25% discounted rate
    output_cost = out_tok * pout / 1_000_000
    return {
        "input":      round(input_cost, 6),
        "cache_read": round(cache_cost, 6),
        "output":     round(output_cost, 6),
        "total":      round(input_cost + cache_cost + output_cost, 6),
    }

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

async def _acall_thinking(model, messages, response_model, max_output_tokens, thinking_budget=4000) -> LLMResult:
    """Gemini thinking-budget call. `thinking_budget` is the reasoning-token budget the model may spend
    before the answer (Gemini 2.5 thinking). Used ONLY for the complex slice (gated by src/complexity.py).
    Goes through the spine's call_thinking - so it gets per-minute retry + per-day key-rotation +
    fail-closed, AND instructor's typed response_model (a validated verdict, not raw text). The async-ness
    is just the to_thread hop in call_thinking, same as _acall_gemini over call_gemini_async."""
    import asyncio
    parsed, raw = await asyncio.to_thread(
        call_thinking, model, messages, response_model, max_output_tokens, thinking_budget)
    um = raw.usage_metadata
    # thoughts_token_count is the reasoning spend - bill it as output (it's charged at the output rate).
    out_tok = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
    in_tok = um.prompt_token_count or 0
    return LLMResult(output=parsed, model=model,
                     input_tokens=in_tok, output_tokens=out_tok,
                     cost_usd=_price(model, in_tok, out_tok))


class UnifiedLLMClient:
    """The router. `acall(tier=...)` picks a Gemini model from TIER_TABLE, executes through the spine,
    and on failure walks the fallback chain - recording which tier actually served the call.

    Two optional flags select a Gemini-native call shape:
      - cache=True     -> `messages[0]` MUST be a system block list
                          built by cached_system() so the stable prefix is registered as a cache.
      - thinking=N     -> thinking-budget path with budget N reasoning tokens. Takes plain chat
                          `messages` + `response_model` (instructor), like the default path.
    Fallback is DISABLED when cache/thinking is requested: those are deliberate, tier-specific calls
    (you asked for Pro+thinking), so silently falling back to a cheaper model would defeat the point -
    it raises instead, staying fail-closed and honest about which tier ran."""
    
    async def acall(self, tier: str = "fast", *, messages: list,  response_model=None,
                    max_output_tokens: int = 2000, cache: bool = False, thinking: int = 0) -> LLMResult:
        t0 = time.perf_counter()
        special = cache or thinking
        last_err = None
        for i, t in enumerate(_resolve_chain(tier, special)):
            spec = TIER_TABLE.get(t)
            if spec is None:
                continue
            try:
                # cache takes [cached_block, {'role':'user',...}]; thinking takes plain chat messages
                # (instructor, like the default path) plus a reasoning budget.
                if thinking:
                    res = await _acall_thinking(spec.model, messages, response_model,
                                                max_output_tokens, thinking)
                elif cache:
                    res = await _acall_cached(spec.model, messages[0]["content"],
                                              messages[1]["content"], max_output_tokens,
                                              response_schema=response_model)
                else:
                    res = await _acall_gemini(spec.model, messages, response_model, max_output_tokens)

                if i > 0:
                    res.fell_back_from = tier
                    log.warning("LLM tier '%s' failed; served by fallback '%s'", tier, t)
                _trace(res, tier, time.perf_counter() - t0)
                return res
            except Exception as e:
                last_err = e
                if special:
                    break
        raise _exhausted(last_err, tier)

    def call(self, tier: str = "fast", *, messages: list, response_model=None,
             max_output_tokens: int = 2000, cache: bool = False) -> LLMResult:
        """Sync twin of acall for the SEQUENTIAL nodes (plan/reflexion) that don't need the event
        loop - same tier selection + fallback chain + fail-closed contract, on the sync spine.

        cache=True selects the context-cache path (`messages[0]` is the cached part built by
        cached_system()/cached_diff(), `messages[1]` the per-run variable part) through the sync
        _cached_call core - the SAME core + _CACHE_HANDLES the async fan-out uses, so a sync cached
        call reuses whatever handle the async nodes primed. As with acall, fallback is DISABLED for
        the cache path (a deliberate, tier-specific call must not silently drop to a cheaper model).
        No `thinking` flag here: thinking is the deep-reasoning slice and rides the async acall;
        the sequential nodes that use this method don't reason that deeply."""
        last_err = None
        for i, t in enumerate(_resolve_chain(tier, special=cache)):
            spec = TIER_TABLE.get(t)
            if spec is None:
                continue
            try:
                if cache:
                    res = _cached_call(spec.model, messages[0]["content"], messages[1]["content"],
                                       max_output_tokens, response_schema=response_model)
                else:
                    res = _call_gemini(spec.model, messages, response_model, max_output_tokens)
                if i > 0:
                    res.fell_back_from = tier
                    log.warning("LLM tier '%s' failed; served by fallback '%s'", tier, t)
                _trace(res, tier, 0.0)
                return res
            except Exception as e:
                last_err = e
                if cache:
                    break
        raise _exhausted(last_err, tier)

llm = UnifiedLLMClient()   # THE shared router singleton - every node calls llm.acall(tier=...)


def _diff_cache_note(res) -> str:
    return (f"Cache(diff): read={res.cache_read_tokens} input={res.input_tokens} "
            f"output={res.output_tokens} cost=${res.cost_usd:.6f}\n")


# Diff-cache helper for the async fan-out nodes (compliance/quality/coverage). They cache the DIFF (the
# part identical across the Flash diff-nodes) and vary the instructions. compliance runs FIRST and primes
# the handle (so the later nodes are pure reusers, no parallel create-race); quality/coverage then reuse
# it. All share ONE handle (keyed by model+diff) because they share the Flash model - a CachedContent is
# model-bound, so security (Pro) can't join. On ANY cache failure (e.g. diff below Gemini's ~2048-token
# floor) it retries plain (no cache, no note) so callers never branch. Both paths go through the router
# (llm.acall) so the fan-out is traced. QuotaExhaustedError propagates (fail-closed). The sequential plan
# node hits the same cache + handles via the sync router path (llm.call(cache=True)), not a separate helper.

async def audit_with_diff_cache(diff, instructions, response_model, max_output_tokens):
    """Async diff-cache call for the fan-out + async nodes (compliance/quality/coverage). Routes through
    the router (llm.acall) so the fan-out is tier-resolved + TRACED, not a side-door call. tier='balanced'
    resolves to the Flash model the fan-out shares (so they keep ONE _CACHE_HANDLES handle). Returns
    (parsed_response, cache_note). On a cache miss (e.g. diff below Gemini's ~2048-token floor) it retries
    plain (no cache, empty note) - also through the router. QuotaExhaustedError propagates (fail-closed).
    NOTE the two output shapes: the cache path returns native JSON (parse it), the plain acall path returns
    an already-parsed Instructor model (use res.output directly - do NOT re-parse)."""
    try:
        res = await llm.acall(tier="balanced", cache=True,
                              messages=[cached_diff(diff), {"role": "user", "content": instructions}],
                              response_model=response_model, max_output_tokens=max_output_tokens)
        return response_model.model_validate_json(res.output), _diff_cache_note(res)
    except QuotaExhaustedError:
        raise
    except Exception:
        res = await llm.acall(tier="balanced",
                              messages=[{"role": "user", "content": instructions + "\n\n" + diff}],
                              response_model=response_model, max_output_tokens=max_output_tokens)
        return res.output, ""

from contextlib import contextmanager

@contextmanager
def audit_trace(thread_id: str, label: str | None = None):
    """Open ONE parent trace for a whole audit run so every node's LLM call nests under it
    (instead of each call being its own orphan trace) and the run's scores can attach to it.
    `label` IS the trace name in the dashboard (no 'pr-audit:' prefix - the whole project is the
    PR auditor, so prefixing every trace with that is noise). Callers pass the surface+branch,
    e.g. 'cli:develop' / 'ci:develop' / 'demo' / 'batch-cli:develop'. The raw thread_id stays in
    metadata for correlation. No-op when Langfuse is unconfigured."""
    lf = _langfuse()
    if lf is None:
        yield
        return
    name = label or "audit"
    try:
        with lf.start_as_current_observation(name=name, as_type="span",
                                              metadata={"thread_id": thread_id,
                                                        "branch": label}):
            # Capture the OTEL context INSIDE the parent span and publish it, so _trace (running in
            # other node tasks / the to_thread worker) re-attaches it and nests as a true child.
            # The parent span starts first = trace root, so its name is the trace name.
            token = _AUDIT_OTEL_CTX.set(_otel_ctx.get_current())
            try:
                yield
            finally:
                _AUDIT_OTEL_CTX.reset(token)
    except Exception as e:
        log.warning("Langfuse audit_trace failed (%s); continuing untraced.", e)
        yield


def score_audit(scores: dict[str, float]):
    """Attach the audit's dimension scores to the CURRENT trace (the one audit_trace opened).
    Best-effort: no client / no active trace -> no-op. Call from inside the audit_trace block."""
    lf = _langfuse()
    if lf is None:
        return
    for name, value in scores.items():
        if value is None:
            continue
        try:
            lf.score_current_trace(name=name, value=float(value), data_type="NUMERIC")
        except Exception as e:
            log.warning("Langfuse score '%s' failed (%s); continuing.", name, e)


def flush_traces():
    """Ship any buffered Langfuse spans and BLOCK until the network export finishes.
    Call at the end of a run before the event loop / process tears down.

    Uses shutdown(), not flush(): flush() only drains the span queue to the OTEL
    exporter and returns - the actual HTTP POST runs on a background thread that the
    process can abandon on exit (the cause of 'traces don't land' under
    asyncio.run + sys.exit). shutdown() blocks until the POST completes. It's called
    once at the end of a run, so making the client unusable afterward is fine."""
    lf = _langfuse()
    if lf is not None:
        try:
            lf.shutdown()
        except Exception as e:
            log.warning("Langfuse shutdown failed (%s); continuing.", e)