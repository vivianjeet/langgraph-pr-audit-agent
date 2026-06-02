# History compression: collapse the oldest slice of a message list into one summary message,
# keeping the signal (decisions, findings, files, scores) and dropping exploratory noise.
# Generic over a list of stringifiable messages; LLM-summarises via the shared retry layer.

from src.llm_retry import call_gemini, QuotaExhaustedError

# Lines worth keeping verbatim if we ever fall back to a non-LLM compression path.
# Shared source of truth (defined in state.py) so this can't drift from reflexion's
# critique set; superset = reflexion's prefixes + post-synthesis decisions.
from src.state import COMPRESSION_SIGNAL_PREFIXES as COMPRESSION_SIGNAL_PREFIXES

FAST_MODEL = "gemini-2.5-flash"
SUMMARY_TOKENS = 1024

def should_compress(messages: list, count_fn, budget: int, threshold: float = 0.8) -> bool:
    """True when current message tokens reach `threshold` (default 80%) of `budget`.
    This is the TRIGGER - the curriculum's '80% context threshold'."""
    used = sum(count_fn(str(m)) for m in messages)
    return used >= threshold * budget


def compress_history(messages: list, compress_ratio: float = 0.5, min_keep: int = 2) -> list:
    """
    Compress the OLDEST `compress_ratio` of the message list (curriculum: 'oldest 50%') into
    ONE summary message, keeping the newest portion verbatim. Returns [summary, *recent].

    `compress_ratio=0.5` collapses the oldest half; `min_keep` guarantees at least that many
    recent messages survive (so a short list isn't over-compressed). Returns unchanged if there
    is nothing meaningful to compress.
    """
    n = len(messages)
    split = int(n * compress_ratio)              # how many of the OLDEST to fold up
    # Don't compress if it wouldn't actually shrink anything, or we'd keep fewer than min_keep.
    if split < 1 or (n - split) < min_keep:
        return messages
    old, recent = messages[:split], messages[split:]
    transcript = "\n".join(str(m) for m in old)

    system_prompt = (
        "You compress an AI code-audit session transcript. Preserve, verbatim where possible: "
        "decisions (approve/reject), security/quality/test scores, CRITICAL/HIGH findings, and "
        "file paths. Discard exploratory reasoning and failed tool output. Output 1 short paragraph."
    )
    messages_payload = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Transcript to compress:\n{transcript}"},
    ]
    try:
        from pydantic import BaseModel, Field

        class _Summary(BaseModel):
            summary: str = Field(description="One-paragraph compressed recap, signal preserved")

        out = call_gemini(model=FAST_MODEL, messages=messages_payload,
                          response_model=_Summary, max_output_tokens=SUMMARY_TOKENS)
        summary_text = out.summary
    except QuotaExhaustedError:
        raise
    except Exception:
        # Fallback: no LLM -> keep only signal lines from the old slice, drop the rest.
        kept = [str(m) for m in old if str(m).startswith(COMPRESSION_SIGNAL_PREFIXES)]
        summary_text = "Compressed (no-LLM fallback). Kept signal:\n" + "\n".join(kept)

    return [f"System: [compressed {len(old)} earlier messages] {summary_text}", *recent]
