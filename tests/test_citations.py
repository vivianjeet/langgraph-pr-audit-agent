from unittest.mock import patch
import src.citations as cit
from src.citations import _CitedClaims, _Claim, _Citation

def test_returns_empty_without_passages():
    assert cit.cited_compliance_claims("diff", []) == []

def test_keeps_a_span_that_is_a_real_substring():
    out = _CitedClaims(claims=[_Claim(
        claim="Logs PII unmasked.",
        citations=[_Citation(source="RBI", cited_text="PII must be masked")])])
    with patch.object(cit, "call_gemini", return_value=out):
        res = cit.cited_compliance_claims(
            "diff that logs user.pan",
            [{"text": "Sensitive customer data: PII must be masked in logs.", "source": "RBI"}])
    assert res == [{"claim": "Logs PII unmasked.",
                    "citations": [{"source": "RBI", "cited_text": "PII must be masked"}]}]


def test_drops_a_hallucinated_span_not_in_the_passage():
    # The model invented a quote that is NOT in the passage - it must be dropped, leaving no claim.
    out = _CitedClaims(claims=[_Claim(
        claim="Violates made-up rule.",
        citations=[_Citation(source="RBI", cited_text="customers must wear hats")])])
    with patch.object(cit, "call_gemini", return_value=out):
        res = cit.cited_compliance_claims(
            "diff", [{"text": "PII must be masked in logs.", "source": "RBI"}])
    assert res == []

def test_model_failure_is_fail_soft():
    with patch.object(cit, "call_gemini", side_effect=RuntimeError("down")):
        assert cit.cited_compliance_claims("diff", [{"text": "x", "source": "RBI"}]) == []