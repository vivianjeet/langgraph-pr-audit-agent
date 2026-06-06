# - shared retry/quota/rotation layer for ALL Gemini calls (chat + embeddings).
import os
import re
import logging
import instructor
from google import genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import asyncio   # add to imports
import threading 

load_dotenv()
log = logging.getLogger(__name__)

# guards _key_idx + client rebinding under concurrent fan-out
_rotate_lock = threading.Lock()

# --- API key pool (primary + optional fallbacks) ---
_KEYS = [k for k in (
    os.environ.get("GEMINI_API_KEY"),
    os.environ.get("GEMINI_API_KEY2"),
    os.environ.get("GEMINI_API_KEY3"),
    os.environ.get("GEMINI_API_KEY4"),
) if k]
if not _KEYS:
    raise RuntimeError("No GEMINI_API_KEY[/2/3] found in environment.")
_key_idx = 0

_RETRYABLE_CODES = ("429", "500", "502", "503", "504")
_NON_RETRYABLE_CODES = ("400", "401", "403", "404")
_MAX_RETRY_WAIT = 90.0

class QuotaExhaustedError(Exception):
    """
    ALL API keys are daily-exhausted - retrying/rotating won't help. 
    Abort the audit; do NOT degrade to a misleading clean score. 
    Nodes must let this propagate (fail closed).
    """

def current_key():
    return _KEYS[_key_idx]

# --- clients (rebuilt when the key rotates) ---
def _build_clients():
    gc = genai.Client(api_key=current_key())
    return gc, instructor.from_genai(gc)

_genai_client, _client = _build_clients()

def _refresh_clients():
    global _genai_client, _client
    _genai_client, _client = _build_clients()

def genai_client():
    """
    Raw genai client for embeddings (vectorstore imports this)
    """
    return _genai_client

# --- error classification ---
def _retry_delay_seconds(exc):
    text = str(exc)
    # structured 'retryDelay' : '14s'
    m = re.search(r"retryDelay[\"']?\s*:?\s*[\"']?(\d+(?:\.\d+)?)[\"']?", text)
    if m:
        return float(m.group(1))
    # humann 'Please retry in 14.9s.
    m = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
    if m:
        return float(m.group(1))
    return None

def _is_daily_quota(exc):
    text = str(exc)
    if "PerDay" in text:
        return True
    delay = _retry_delay_seconds(exc)        # no quotaId but absurd delay => daily
    return delay is not None and delay > _MAX_RETRY_WAIT

def _is_key_blocked(exc):
    """A 403 API_KEY_SERVICE_BLOCKED (or PERMISSION_DENIED on this key) is permanent
    for THIS key - rotating to another key may succeed, but retrying won't."""
    text = str(exc)
    return "API_KEY_SERVICE_BLOCKED" in text or ("403" in text and "PERMISSION_DENIED" in text)


def _is_billing_error(exc):
    """A 429 whose cause is depleted prepay credits / billing, NOT throttling. Retrying or
    waiting can't refill a wallet, so this is terminal for THIS key. Keys can belong to
    different billing projects, so we treat it like a blocked key: rotate if another key
    exists (it may be funded), else fail closed. The status is RESOURCE_EXHAUSTED but the
    message is the tell - so this MUST be checked before the generic retryable path."""
    text = str(exc).lower()
    return "credits are depleted" in text or ("billing" in text and "resource_exhausted" in text)



def _is_retryable(exc):
    text = str(exc)
    if any(c in text for c in _NON_RETRYABLE_CODES):
        return False
    if _is_billing_error(exc):               # depleted credits: terminal for this key, rotation-not-retry
        return False
    if _is_daily_quota(exc):                 # per-day handled by rotation, NOT tenacity retry
        return False
    return any(c in text for c in _RETRYABLE_CODES) or "RESOURCE_EXHAUSTED" in text


def _wait_server_then_backoff(retry_state):
    exc = retry_state.outcome.exception()
    if exc is not None:
        delay = _retry_delay_seconds(exc)
        if delay is not None:
            return min(delay + 1.0, _MAX_RETRY_WAIT)   # honour the 15s window, never oversleep
    return wait_exponential(multiplier=1, min=1, max=30)(retry_state)

def _log_before_sleep(retry_state):
    """Print why we're waiting and for how long, so a live run isn't silent."""
    wait = getattr(retry_state.next_action, "sleep", 0.0) # the wait tenacity is about to do
    attempt = retry_state.attempt_number          # which attempt just failed
    exc = retry_state.outcome.exception()
    server = _retry_delay_seconds(exc)            # what the server asked for, if any
    reason = f"server retryDelay={server}s" if server is not None else "exponential backoff"
    log.warning(
        "Gemini rate-limited (attempt %d/5, %s) -> sleeping %.1fs before retry. Last error: %s",
        attempt, reason, wait, str(exc)[:120],
    )

def _rotate_from(failed_idx: int) -> bool:
    """Rotate off the key the caller just saw fail. Double-checked under the lock: if another
    thread already advanced past `failed_idx` (concurrent fan-out hitting the same dead key),
    this thread does NOT rotate again - it returns True so the caller retries on the now-current
    key. Returns False only when failed_idx is the LAST key and it's still current (pool exhausted).
    """
    global _key_idx
    with _rotate_lock:
        if _key_idx != failed_idx:
            return True                      # someone already rotated us off the dead key; ride along
        if _key_idx + 1 < len(_KEYS):
            _key_idx += 1
            log.warning("Rotating to backup Gemini API key #%d", _key_idx + 1)
            _refresh_clients()               # rebind clients INSIDE the lock - no torn client/key pair
            return True
        return False                         # this was the last key and it's still current → exhausted

llm_retry = retry(
    stop=stop_after_attempt(5),
    wait=_wait_server_then_backoff,
    retry=retry_if_exception(_is_retryable),
    before_sleep=_log_before_sleep,
    reraise=True,
)

def _run_with_rotation(fn):
    """Run fn() with the current key. On a PER-DAY quota error, rotate 
    (thread-safe, double-checked) to the next key and retry the SAME request; 
    raise QuotaExhaustedError only when every key is exhausted.
    Per-minute 429s are handled inside fn() by the @llm_retry wait (no rotation)."""
    while True:
        idx = _key_idx                       # the key this attempt is using
        try:
            return fn()
        except Exception as e:
            if _is_daily_quota(e) or _is_key_blocked(e) or _is_billing_error(e):
                if _rotate_from(idx):        # rotate only if nobody else already did
                    continue                 # retry on the (possibly already-rotated) current key                # retry same request on the new key
                raise QuotaExhaustedError(
                    "All Gemini API keys are exhausted, key-blocked or out of billing credits "
                    "(prepay depleted). Aborting to avoid a false-clean score. If it's billing, "
                    "top up the project at https://ai.studio/projects - retrying won't help."
                ) from e
            raise                            # non-quota errors: already retried by @llm_retry, propagate

# --- chat (instructor) call shape ---
@llm_retry
def _raw_chat(model, messages, response_model, max_output_tokens):
    return _client.chat.completions.create(
        model=model, messages=messages, response_model=response_model,
        max_retries=2, generation_config={"max_output_tokens": max_output_tokens},
    )


def call_gemini(model, messages, response_model, max_output_tokens):
    """Shared structured chat call: per-minute retry + per-day key-rotation + fail-closed."""
    return _run_with_rotation(
        lambda: _raw_chat(model, messages, response_model, max_output_tokens)
    )

# --- embedding (raw genai) call shape ---
@llm_retry
def _raw_embed(model, contents, output_dim):
    return _genai_client.models.embed_content(
        model=model, contents=contents, config={"output_dimensionality": output_dim},
    )


def call_embed(model, contents, output_dim):
    """Shared embedding call: same per-minute retry + per-day key-rotation + fail-closed.
    Different API shape than chat (raw genai, not instructor), so it's a separate wrapper."""
    return _run_with_rotation(lambda: _raw_embed(model, contents, output_dim))

async def call_gemini_async(model, messages, response_model, max_output_tokens):
    """Async wrapper over the sync call_gemini: runs the blocking instructor call on a worker
    thread so several audit nodes can overlap their Gemini I/O via asyncio. Reuses the FULL
    sync stack (per-minute retry + per-day key-rotation + fail-closed) unchanged - we only move
    the blocking call off the event loop, we do NOT reimplement rotation against an async client."""
    return await asyncio.to_thread(
        call_gemini, model, messages, response_model, max_output_tokens
    )