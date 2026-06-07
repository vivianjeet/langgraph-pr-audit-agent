from unittest.mock import patch, MagicMock
import src.db.vectorstore as vs


def test_cosine_basics():
    assert vs._cosine([1.0, 0.0], [1.0, 0.0]) == 1.0          # identical -> 1
    assert abs(vs._cosine([1.0, 0.0], [0.0, 1.0])) < 1e-9     # orthogonal -> 0
    assert vs._cosine([0.0, 0.0], [1.0, 1.0]) == 0.0          # zero vector -> 0, no div error


def _row(summary, emb, sim):
    # (pr_summary, report, embedding, similarity) - the 4-col SELECT shape.
    return (summary, {"r": summary}, emb, sim)


def test_retrieve_drops_near_duplicate():
    # rows best-first: A and A' are near-identical embeddings (the same PR's two runs); B differs.
    rows = [
        _row("A  needs-changes", [1.0, 0.0, 0.0], 0.99),
        _row("A' approve",       [0.999, 0.001, 0.0], 0.98),   # ~dup of A -> must be dropped
        _row("B  other PR",      [0.0, 1.0, 0.0], 0.97),
    ]
    cur = MagicMock()
    cur.fetchall.return_value = rows
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    with patch.object(vs, "embed", return_value=[1.0, 0.0, 0.0]), \
         patch.object(vs, "get_conn") as gc:
        gc.return_value.__enter__.return_value = conn
        out = vs.retrieve_similar_prs("query", k=3)
    summaries = [o["pr_summary"] for o in out]
    assert "A  needs-changes" in summaries       # first of the dup pair kept
    assert "A' approve" not in summaries          # near-dup dropped
    assert "B  other PR" in summaries             # the genuinely-different one survives
    assert len(out) == 2


def test_retrieve_respects_threshold():
    # A row at/below SIM_THRESHOLD is filtered out entirely (unchanged behaviour).
    rows = [_row("low", [1.0, 0.0], vs.SIM_THRESHOLD - 0.01)]
    cur = MagicMock(); cur.fetchall.return_value = rows
    conn = MagicMock(); conn.cursor.return_value.__enter__.return_value = cur
    with patch.object(vs, "embed", return_value=[1.0, 0.0]), \
         patch.object(vs, "get_conn") as gc:
        gc.return_value.__enter__.return_value = conn
        assert vs.retrieve_similar_prs("query") == []