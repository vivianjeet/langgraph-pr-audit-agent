"""WHAT THIS WHOLE FILE TESTS
=============================
The VERIFY step (the trust boundary) inside `cited_compliance_claims` - specifically the
substring check on src/citations.py:63:

    c.cited_text.strip() in by_source.get(c.source, "")

That one line decides whether a span Gemini quoted is KEPT (a verified citation) or DROPPED
(a hallucination). The product symptom this guards is: `compliance_hits > 0` but `citations = 0`
on a clearly regulated diff. Two very different causes produce that same symptom -

  (GOOD) the model fabricated a span -> verification correctly dropped it.
  (BAD)  the model quoted a REAL span, but a whitespace/normalisation/source mismatch made the
         substring check fail -> a legitimate citation was lost.

`test_citations.py` already covers the happy path, a fabricated drop, and fail-soft. THIS file
covers the EDGE cases that separate a good drop from a bad drop, so an empty `citations` list can
be trusted as "nothing real matched" rather than "the verifier is too brittle". Several tests PIN
current behaviour (including known brittleness) so a future change to the matching logic is caught.

No LLM, no DB: `call_gemini` is patched to return exactly the spans we want to verify, so each
scenario is deterministic and the test exercises ONLY the verify step.
"""
from unittest.mock import patch
import src.citations as cit
from src.citations import cited_compliance_claims, _CitedClaims, _Claim, _Citation

# Two passages from DIFFERENT sources, used across the tests. The double-space in "PII  (PAN)"
# is deliberate - it is the internal-whitespace case below.
PASSAGES = [
    {"source": "RBI-IT", "text": "Customer PII  (PAN) must be masked in application logs."},
    {"source": "HIPAA-164", "text": "PHI must never be written to logs in plaintext."},
]


def _run(*citations):
    """Patch call_gemini to return ONE claim carrying the given citation spans, then run the
    real verify step over PASSAGES. Returns the verified output list."""
    mock_out = _CitedClaims(claims=[_Claim(claim="logs sensitive data", citations=list(citations))])
    with patch.object(cit, "call_gemini", return_value=mock_out):
        return cited_compliance_claims("diff that logs user.pan", PASSAGES)


def test_verbatim_span_survives():
    # HAPPY PATH (the GOOD keep): an exact substring of the named passage passes verification.
    out = _run(_Citation(source="RBI-IT", cited_text="must be masked in application logs"))
    assert len(out) == 1
    assert out[0]["citations"][0]["cited_text"] == "must be masked in application logs"


def test_outer_whitespace_is_tolerated():
    # THE BAD-DROP GUARD: a legitimate quote with leading/trailing whitespace must NOT be lost -
    # .strip() handles outer whitespace. If this ever fails, the verifier is dropping real quotes.
    out = _run(_Citation(source="RBI-IT", cited_text="   must be masked   "))
    assert len(out) == 1


def test_wrong_source_is_dropped():
    # SOURCE KEYING: a span that IS real text but is attributed to the WRONG passage must drop -
    # "PHI must never" is HIPAA's text, not RBI-IT's, so claiming it under RBI-IT fails the check.
    out = _run(_Citation(source="RBI-IT", cited_text="PHI must never be written to logs"))
    assert out == []


def test_fabricated_span_is_dropped():
    # THE GOOD DROP: a span that appears in NO passage is a hallucination and must be dropped,
    # leaving the claim with zero surviving citations -> the claim is not emitted at all.
    out = _run(_Citation(source="RBI-IT", cited_text="PII must be encrypted at rest"))
    assert out == []


def test_internal_whitespace_is_normalised():
    # THE BAD-DROP FIX (regression guard): the passage has "PII  (PAN)" (two spaces); a model that
    # quotes "PII (PAN)" (one space) is a legitimate, near-verbatim quote. _norm_ws collapses any
    # whitespace run to a single space on BOTH sides, so this real quote now SURVIVES instead of
    # being dropped as a false hallucination. (Before the fix this returned [] - see git history.)
    out = _run(_Citation(source="RBI-IT", cited_text="PII (PAN)"))   # single space vs passage's double
    assert len(out) == 1
    # The rendered citation keeps the model's ORIGINAL text - normalisation is match-only.
    assert out[0]["citations"][0]["cited_text"] == "PII (PAN)"


def test_newline_in_quote_is_normalised():
    # A quote that wrapped a newline (common when a model copies across a line break) must still
    # match - _norm_ws turns the "\n" into a single space. Passage is one line; quote spans words.
    out = _run(_Citation(source="HIPAA-164", cited_text="written to logs\nin plaintext"))
    assert len(out) == 1


def test_fabrication_still_dropped_after_normalisation():
    # GUARD: normalisation must NOT make the check so loose it accepts invented text. A span that
    # is not in the passage (whitespace-normalised or not) is still dropped.
    out = _run(_Citation(source="RBI-IT", cited_text="PII must be encrypted at rest"))
    assert out == []


def test_partial_real_substring_survives():
    # A shorter exact slice of the passage still verifies - the check is substring, not full-line,
    # so the model may quote just the relevant fragment.
    out = _run(_Citation(source="HIPAA-164", cited_text="written to logs in plaintext"))
    assert len(out) == 1


def test_span_across_two_same_source_passages():
    # CROSS-PASSAGE BOUNDARY: when two passages share a source, by_source concatenates them with a
    # "\n". After _norm_ws that "\n" becomes a single space, so a span straddling the join verifies.
    # Pins that same-source multi-passage corpora don't silently lose boundary-spanning quotes.
    two_same_source = [
        {"source": "RBI-IT", "text": "Customer PII must be masked"},
        {"source": "RBI-IT", "text": "in all application logs and audit trails"},
    ]
    mock_out = _CitedClaims(claims=[_Claim(
        claim="logs PII",
        citations=[_Citation(source="RBI-IT", cited_text="must be masked in all application logs")])])
    with patch.object(cit, "call_gemini", return_value=mock_out):
        out = cited_compliance_claims("diff", two_same_source)
    assert len(out) == 1            # the span crosses the two-passage join and still verifies


def test_mixed_one_real_one_fake_keeps_only_the_real():
    # When a claim carries two spans - one verbatim, one hallucinated - only the real one survives,
    # and the claim IS emitted (it has a surviving citation). Proves filtering is per-CITATION,
    # not all-or-nothing per claim.
    out = _run(
        _Citation(source="RBI-IT", cited_text="must be masked in application logs"),   # real
        _Citation(source="RBI-IT", cited_text="and shredded weekly"),                  # fake
    )
    assert len(out) == 1
    assert len(out[0]["citations"]) == 1
    assert out[0]["citations"][0]["cited_text"] == "must be masked in application logs"
