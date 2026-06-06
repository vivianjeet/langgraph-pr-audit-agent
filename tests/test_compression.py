"""Tests in this file: history compression - the pure functions (should_compress,
compress_history), the live driver (run_compression_pass) and the graph node (compress_node).

HOW IT WORKS (no DB, no LLM)
- `call_gemini` is patched on src.compression so compress_history never hits the network;
  tests either return a fake summary object or raise to exercise the no-LLM fallback.
- compress_node is a thin wrapper: it reads `messages` (audit substate) + `force_compress`
  (top-level channel) and delegates to run_compression_pass, returning {} (pass-through) or
  {"compressed": [...]}. Tests pin both branches + that it reads force_compress off TOP level.
"""
from unittest.mock import patch
import src.compression as comp
from src.compression import should_compress, compress_history, run_compression_pass
from src.nodes.compress import compress_node


def _count(s):  # deterministic fake counter for tests
    return len(s)


def test_should_compress_threshold():
    msgs = ["x" * 50]
    assert should_compress(msgs, _count, budget=100, threshold=0.8) is False   # 50 < 80
    assert should_compress(["x" * 90], _count, budget=100, threshold=0.8) is True  # 90 >= 80


def test_no_compression_when_short():
    # 3 msgs, ratio 0.5 -> split=1, but keeping (3-1)=2 >= min_keep is fine... so to assert the
    # "too short to bother" path, use a list where compressing wouldn't shrink meaningfully:
    assert compress_history(["a"], compress_ratio=0.5) == ["a"]          # split=0 -> unchanged
    assert compress_history(["a", "b"], compress_ratio=0.5, min_keep=2) == ["a", "b"]  # keep<min


def test_compress_collapses_oldest_half():
    class _Sum:
        summary = "decision=reject, security=0.0, auth/login.py SQLi."
    msgs = [f"m{i}" for i in range(10)]
    with patch.object(comp, "call_gemini", return_value=_Sum()):
        out = compress_history(msgs, compress_ratio=0.5)             # oldest 50% (m0..m4)
    assert len(out) == 6                                  # 1 summary + 5 recent
    assert out[0].startswith("System: [compressed 5 earlier messages]")
    assert out[-5:] == ["m5", "m6", "m7", "m8", "m9"]    # newest half preserved verbatim


def test_compress_fallback_keeps_signal_lines_when_llm_fails():
    # 6 msgs, ratio 0.5 -> split=3: oldest 3 (the signal ones) get folded, newest 3 kept verbatim.
    msgs = ["System: Audit plan -> deep", "noise reasoning", "System: Human reject",
            "r1", "r2", "r3"]
    with patch.object(comp, "call_gemini", side_effect=RuntimeError("down")):
        out = compress_history(msgs, compress_ratio=0.5)
    assert "Audit plan" in out[0] and "Human reject" in out[0]   # signal kept from oldest half
    assert "noise reasoning" not in out[0]                       # noise dropped
    assert out[-3:] == ["r1", "r2", "r3"]                        # newest half verbatim


# --- run_compression_pass: the live driver (force vs auto vs not-triggered) ---

def test_driver_not_triggered_below_threshold_without_force():
    update, report = run_compression_pass(["short"], force=False, budget=10000)
    assert update == {}                                  # nothing written -> pass-through
    assert "not triggered" in report


def test_driver_forced_with_large_flag():
    msgs = [f"m{i}" for i in range(10)]
    class _Sum: summary = "decision=reject, security=0.0"
    with patch.object(comp, "call_gemini", return_value=_Sum()):
        update, report = run_compression_pass(msgs, force=True, budget=10000)  # below 80% but FORCED
    assert "compressed" in update and len(update["compressed"]) < len(msgs)
    assert "--large" in report


def test_driver_auto_fires_over_threshold():
    big = ["x" * 4000 for _ in range(3)]                 # ~3000 est tokens >= 80% of 3500 budget
    class _Sum: summary = "recap"
    with patch.object(comp, "call_gemini", return_value=_Sum()):
        update, report = run_compression_pass(big, force=False, budget=3500)
    assert "compressed" in update
    assert "AUTO" in report


# --- compress_node: the graph node wrapping the driver ---

def test_compress_node_passthrough_when_not_triggered():
    # Small session, not forced -> node returns {} so the graph flows on untouched.
    out = compress_node({"audit": {"messages": ["short"]}, "force_compress": False})
    assert out == {}


def test_compress_node_forced_writes_compressed_channel():
    msgs = [f"m{i}" for i in range(10)]
    class _Sum: summary = "recap"
    with patch.object(comp, "call_gemini", return_value=_Sum()):
        out = compress_node({"audit": {"messages": msgs}, "force_compress": True})
    assert "compressed" in out and len(out["compressed"]) < len(msgs)


def test_compress_node_reads_force_from_top_level_not_audit():
    # force_compress is a TOP-LEVEL channel, NOT inside the audit substate. A flag placed in
    # audit must be IGNORED (proves the node reads the right level).
    msgs = [f"m{i}" for i in range(10)]
    out = compress_node({"audit": {"messages": msgs, "force_compress": True},  # wrong level
                         "force_compress": False})                              # right level
    assert out == {}                                     # honoured top-level False -> pass-through
