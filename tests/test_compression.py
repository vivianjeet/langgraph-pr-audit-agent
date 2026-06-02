from unittest.mock import patch
import src.compression as comp
from src.compression import should_compress, compress_history


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
