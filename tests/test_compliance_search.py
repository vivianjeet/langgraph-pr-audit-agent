# the vectorstore helper's threshold filter (DB mocked).
from unittest.mock import patch
import src.db.vectorstore as vs


def test_search_compliance_filters_below_threshold():
    # rows are (content, source, framework, similarity) - 0.40 < SIM_THRESHOLD is dropped.
    rows = [("masked PII rule", "GDPR Art. 32", "gdpr", 0.91), ("loose match", "RBI", "rbi", 0.40)]
    with patch.object(vs, "embed", return_value=[0.0] * 768), \
         patch.object(vs, "get_conn") as gc:
        cur = gc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        out = vs.search_compliance("PII logging", k=3)
    assert [h["source"] for h in out] == ["GDPR Art. 32"]   # only the >0.7 hit survives
    assert out[0]["text"] == "masked PII rule"
    assert out[0]["framework"] == "gdpr"                     # framework flows through the result


def test_search_compliance_framework_filter_adds_where_clause():
    rows = [("PHI rule", "HIPAA 164.312", "hipaa", 0.88)]
    with patch.object(vs, "embed", return_value=[0.0] * 768), \
         patch.object(vs, "get_conn") as gc:
        cur = gc.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
        cur.fetchall.return_value = rows
        out = vs.search_compliance("PHI in logs", k=3, framework="hipaa")
    sql = cur.execute.call_args_list[-1].args[0]            # last execute = the SELECT
    assert "WHERE framework = %s" in sql                    # the filter branch was taken
    assert out[0]["framework"] == "hipaa"