# the server's @tool functions are ordinary callables; test directly.
from unittest.mock import patch
import src.mcp.compliance_rag_server as srv


def test_search_compliance_docs_delegates_to_vectorstore():
    hit = [{"text": "mask PII", "source": "GDPR Art. 32", "framework": "gdpr"}]
    with patch.object(srv, "search_compliance", return_value=hit) as m:
        out = srv.search_compliance_docs("PII logging", k=2)
    assert out == hit
    m.assert_called_once_with("PII logging", k=2, framework=None)


def test_search_compliance_docs_passes_framework_filter():
    with patch.object(srv, "search_compliance", return_value=[]) as m:
        srv.search_compliance_docs("PHI in logs", k=3, framework="hipaa")
    m.assert_called_once_with("PHI in logs", k=3, framework="hipaa")


def test_get_pr_audit_history_delegates_to_semantic_recall():
    with patch.object(srv, "retrieve_similar_prs", return_value=[{"pr_summary": "x"}]) as m:
        assert srv.get_pr_audit_history("auth change", k=2) == [{"pr_summary": "x"}]
        m.assert_called_once_with("auth change", k=2)