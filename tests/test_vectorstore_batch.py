"""Tests in this file: the vectorstore embedding helpers (Gemini API mocked).

- test_embed_batch_empty_returns_empty          : [] in -> [] out, no API call.
- test_embed_batch_preserves_count_and_dim      : one vector per input, correct dimension.
- test_embed_batch_splits_into_groups           : large corpus is chunked into EMBED_BATCH groups.
- test_embed_batch_raises_on_count_mismatch     : a count mismatch from the API raises.
- test_store_pr_audit_embeds_embed_text_not_summary: storage embeds embed_text, not the summary label.
- test_embed_memoises_same_text_one_api_call    : repeated text hits the cache (one API call).
- test_embed_returns_fresh_list_so_callers_can_mutate: embed() returns a fresh list each call.
"""
from unittest.mock import MagicMock, patch
import src.db.vectorstore as vs
import src.llm_retry as llm_retry
import src.config as cfg


def _fake_resp(n):
    """Mimic an EmbedContentResponse with n embeddings of cfg.EMBED_OUTPUT_DIM floats each."""
    resp = MagicMock()
    resp.embeddings = [MagicMock(values=[0.0] * cfg.EMBED_OUTPUT_DIM) for _ in range(n)]
    return resp


def test_embed_batch_empty_returns_empty():
    assert vs.embed_batch([]) == []


def test_embed_batch_preserves_count_and_dim():
    texts = [f"chunk {i}" for i in range(5)]
    with patch.object(llm_retry, "_raw_embed", return_value=_fake_resp(5)) as m:
        out = vs.embed_batch(texts)
    assert len(out) == 5                       # one vector per input
    assert all(len(v) == cfg.EMBED_OUTPUT_DIM for v in out)
    m.assert_called_once()                     # 5 < EMBED_BATCH_SIZE -> a SINGLE call


def test_embed_batch_splits_into_groups():
    # 250 texts with EMBED_BATCH_SIZE=100 -> 3 calls (100 + 100 + 50).
    texts = [f"chunk {i}" for i in range(250)]
    with patch.object(cfg, "EMBED_BATCH_SIZE", 100), \
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
    with patch.object(vs, "embed", return_value=[0.0] * cfg.EMBED_OUTPUT_DIM) as m_embed, \
         patch.object(vs, "get_conn", return_value=MagicMock()):
        vs.store_pr_audit("THE SUMMARY", {"k": 1}, embed_text="THE DIFF")
    m_embed.assert_called_once_with("THE DIFF")     # embedded the diff, not the summary


def test_embed_memoises_same_text_one_api_call():
    # The same text embedded twice (e.g. retrieve THEN finalize on one diff) must hit
    # the cache: only ONE underlying API call, identical vectors returned.
    with patch.object(llm_retry, "_raw_embed", return_value=_fake_resp(1)) as m:
        v1 = vs.embed("identical diff")
        v2 = vs.embed("identical diff")
    assert v1 == v2
    m.assert_called_once()                          # second call served from cache


def test_embed_returns_fresh_list_so_callers_can_mutate():
    # Cache holds an immutable tuple; embed() hands back a NEW list each time, so a
    # caller mutating its result can't corrupt a later cache hit.
    with patch.object(llm_retry, "_raw_embed", return_value=_fake_resp(1)):
        a = vs.embed("x")
        a.append(999.0)            # mutate the returned list
        b = vs.embed("x")          # cache hit
    assert 999.0 not in b          # the cache was not corrupted