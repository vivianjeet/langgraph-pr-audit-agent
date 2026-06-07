import logging
from dataclasses import dataclass
from src.llm_retry import call_gemini, call_gemini_async, QuotaExhaustedError
import src.config as cfg

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
        chain = [tier] if special else [tier] + [t for t in _FALLBACK if t != tier]
        first = tier
        last_err = None
        for i, t in enumerate(chain):
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
                                              messages[1]["content"], max_output_tokens)
                
                else:
                    res = await _acall_gemini(spec.model, messages, response_model, max_output_tokens)
                
                if i > 0:
                    res.fell_back_from = first
                    log.warning("LLM tier '%s' failed; served by fallback '%s'", first, t)
                _trace(res, tier, 0.0)
                return res
            except QuotaExhaustedError as e:
                last_err = e
                if special:
                    break
                continue
            except Exception as e:
                last_err = e
                if special:
                    break
                continue
        raise RuntimeError(f"all LLM tiers failed (last: {type(last_err).__name__} : {last_err})")

# Forward-reference stubs yo be filled
async def _acall_thinking(*a, **k):
    raise NotImplementedError

async def _acall_cached(*a, **k):
    raise NotImplementedError

def _trace(*a, **k):
    pass
