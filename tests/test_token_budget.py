from src.token_budget import TokenBudgetManager, Segment, estimate_tokens


def test_estimate_tokens_rough():
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("") == 1


def test_keeps_high_priority_drops_low_when_over_budget():
    segs = [
        Segment(0, "system", "s" * 40),     # 10 tok, mandatory
        Segment(1, "query",  "q" * 40),     # 10 tok
        Segment(3, "history","h" * 400),    # 100 tok, lowest priority
    ]
    kept, log = TokenBudgetManager(budget_tokens=25).fit(segs)
    labels = [s.label for s in kept]
    assert "system" in labels and "query" in labels   # high priority survived
    assert "history" not in labels                     # low priority trimmed
    assert any("trimmed 'history'" in line for line in log)


def test_kept_segments_returned_in_original_order():
    segs = [Segment(0, "a", "x"*4), Segment(2, "b", "x"*4), Segment(1, "c", "x"*4)]
    kept, _ = TokenBudgetManager(budget_tokens=1000).fit(segs)
    assert [s.label for s in kept] == ["a", "b", "c"]   # original order, not priority order


def test_mandatory_kept_even_if_over_budget():
    segs = [Segment(0, "system", "s" * 4000)]           # 1000 tok, budget only 10
    kept, log = TokenBudgetManager(budget_tokens=10).fit(segs)
    assert [s.label for s in kept] == ["system"]
    assert any("mandatory" in line and "over budget" in line for line in log)


def test_synthetic_long_session_trims_oldest_history_first():
    # The 'demo on synthetic load' case: many history segments, only the NEWEST fit.
    # Encode age into priority so OLDEST is trimmed first (history oldest first).
    # history:0 is the OLDEST -> highest priority number (3+19) -> dropped first;
    # history:19 is the NEWEST -> lowest history priority (3+0) -> kept if room.
    segs = [Segment(0, "system", "s"*40), Segment(1, "query", "q"*40)]
    n = 20
    segs += [Segment(3 + (n - 1 - i), f"history:{i}", "h"*40) for i in range(n)]  # i=0 oldest
    kept, log = TokenBudgetManager(budget_tokens=45).fit(segs)        # room for system+query+~2 history
    hist_kept = sorted(int(s.label.split(":")[1]) for s in kept if s.label.startswith("history"))
    assert len(hist_kept) <= 3                                        # most history trimmed
    assert all(idx >= n - 3 for idx in hist_kept)                    # only the NEWEST survived
    assert len(log) >= 17                                            # trims logged (never silent)