# Small shared text helpers. No project imports - safe to use from any node/db module.


def clip(text: str, hi: int = 140, lo: int | None = None) -> str:
    """Truncate `text` to a readable length, preferring a natural boundary.

    Returns text unchanged if it is already <= `lo`. Otherwise looks inside the
    window text[:hi] and cuts at, in order of preference:
      1. the last sentence end (., !, ?) at or after `lo`  (punctuation kept),
      2. else the last word boundary (space) at or after `lo`,
      3. else a hard cut at `hi`.
    Trailing whitespace is always stripped; no ellipsis is appended.

    `hi` is the hard ceiling (max length). `lo` is the lower bound on where a
    boundary cut may land - it stops a boundary search from throwing away most
    of the text by cutting too early. Defaults to hi // 2 (so callers that only
    care about a ceiling can pass just `hi`, e.g. clip(s, 120)).
    """
    if lo is None:
        lo = hi // 2
    if hi < lo:
        lo = hi

    text = text.strip()
    if len(text) <= lo:
        return text
    window = text[:hi]
    # Prefer a sentence end (., !, ?) at or after lo.
    cut = max(window.rfind(c, lo) for c in ".!?")
    if cut != -1:
        return window[: cut + 1]
    # Else fall back to the last word boundary at or after lo.
    cut = window.rfind(" ", lo)
    if cut != -1:
        return window[:cut].rstrip()
    # No boundary in range: hard cut.
    return window.rstrip()
