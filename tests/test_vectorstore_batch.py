from unittest.mock import MagicMock, patch
import src.db.vectorstore as vs
import src.llm_retry as llm_retry


def _fake_resp(n):
    """Mimic an EmbedContentResponse with n embeddings of EMBED_DIM floats each."""
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=[0.0] * vs.EMBED_DIM) for _ in range(n)]
    return resp


def test_embed_batch_empty_returns_empty():
    assert vs.embed_batch([]) == []


def test_embed_batch_preserves_count_and_dim():
    texts = [f"chunk {i}" for i in range(5)]
    with patch.object(llm_retry, "_raw_embed", return_value=_fake_resp(5)) as m:
        out = vs.embed_batch(texts)
    assert len(out) == 5                       # one vector per input
    assert all(len(v) == vs.EMBED_DIM for v in out)
    m.assert_called_once()                     # 5 < EMBED_BATCH -> a SINGLE call


def test_embed_batch_splits_into_groups():
    # 250 texts with EMBED_BATCH=100 -> 3 calls (100 + 100 + 50).
    texts = [f"chunk {i}" for i in range(250)]
    with patch.object(vs, "EMBED_BATCH", 100), \
         patch.object(llm_retry, "_raw_embed",
                      side_effect=[_fake_resp(100), _fake_resp(100), _fake_resp(50)]) as m:
        out = vs.embed_batch(texts)
    assert len(out) == 250
    assert m.call_count == 3                    # proves batching, not one-at-a-time


def test_embed_batch_raises_on_count_mismatch():
    import pytest
    with patch.object(llm_retry, "_raw_embed", return_value=_fake_resp(2)):
        with pytest.raises(RuntimeError):
            vs.embed_batch(["a", "b", "c"])     # asked for 3, API "returned" 2

def test_store_pr_audit_embeds_embed_text_not_summary():
    # Regression: store must vectorise embed_text (the diff),
    # NOT the summary - the store/query representation mismatch that silently broke retrieval.
    from unittest.mock import patch, MagicMock
    with patch.object(vs, "embed", return_value=[0.0] * vs.EMBED_DIM) as m_embed, \
         patch.object(vs, "get_conn", return_value=MagicMock()):
        vs.store_pr_audit("THE SUMMARY", {"k": 1}, embed_text="THE DIFF")
    m_embed.assert_called_once_with("THE DIFF")     # embedded the diff, not the summary