"""Tests in this file: the LLM retry / API-key-rotation layer (src/llm_retry.py).

Retry-delay parsing:
- test_retry_delay_parses_structured     : delay read from a structured error field.
- test_retry_delay_parses_message        : delay parsed from the error message text.

Quota classification:
- test_is_daily_quota_true_on_perday      : per-day quota error classified as daily.
- test_is_daily_quota_false_on_perminute  : per-minute quota error is NOT daily.
- test_rotation_then_quota_exhausted      : keys rotate, then QuotaExhaustedError when all spent.

Blocked-key detection + rotation:
- test_is_key_blocked_detects_403_service_blocked : 403 service-blocked is a blocked key.
- test_is_key_blocked_detects_permission_denied   : permission-denied is a blocked key.
- test_is_key_blocked_false_on_503                : 503 is transient, NOT a blocked key.
- test_blocked_key_rotates_then_exhausts          : blocked keys rotate, then exhaust.

Thinking-budget call shape (Day 34):
- test_call_thinking_rotates_then_exhausts        : call_thinking inherits key-rotation + fail-closed.
- test_call_thinking_passes_through_result        : forwards thinking_budget + returns _raw_thinking's value.
"""
from unittest.mock import patch
import pytest
import src.llm_retry as r

def test_retry_delay_parses_structured():
    assert r._retry_delay_seconds(Exception("'retryDelay': '14s'")) == 14.0

def test_retry_delay_parses_message():
    assert r._retry_delay_seconds(Exception("Please retry in 14.9s.")) == 14.9

def test_is_daily_quota_true_on_perday():
    assert r._is_daily_quota(Exception("429 ... quotaId PerDay ...")) is True

def test_is_daily_quota_false_on_perminute():
    assert r._is_daily_quota(Exception("429 ... PerMinute ... retryDelay 15s")) is False

def test_rotation_then_quota_exhausted():
    # 2 keys, both per-day exhausted -> rotates once, then raises.
    with patch.object(r, "_KEYS", ["k1", "k2"]), patch.object(r, "_key_idx", 0):
        boom = Exception("429 RESOURCE_EXHAUSTED PerDay")
        with patch.object(r, "_raw_chat", side_effect=boom), \
             patch.object(r, "_refresh_clients"):
            with pytest.raises(r.QuotaExhaustedError):
                r.call_gemini("m", [], object, 100)


def test_is_key_blocked_detects_403_service_blocked():
    assert r._is_key_blocked(Exception("403 PERMISSION_DENIED ... API_KEY_SERVICE_BLOCKED")) is True

def test_is_key_blocked_detects_permission_denied():
    assert r._is_key_blocked(Exception("403 PERMISSION_DENIED on this key")) is True

def test_is_key_blocked_false_on_503():
    assert r._is_key_blocked(Exception("503 UNAVAILABLE")) is False

def test_blocked_key_rotates_then_exhausts():
    # 2 keys, every call raises a blocked-key 403 -> rotates once, then raises
    # QuotaExhaustedError. Proves a 403 now rotates like daily-quota (fast, no 5x retry).
    with patch.object(r, "_KEYS", ["k1", "k2"]), patch.object(r, "_key_idx", 0):
        boom = Exception("403 PERMISSION_DENIED API_KEY_SERVICE_BLOCKED")
        with patch.object(r, "_raw_chat", side_effect=boom) as mock_chat, \
             patch.object(r, "_refresh_clients"):
            with pytest.raises(r.QuotaExhaustedError):
                r.call_gemini("m", [], object, 100)
            assert mock_chat.call_count == 2   # key1, rotate, key2, then give up


def test_call_thinking_rotates_then_exhausts():
    # call_thinking must inherit the SAME _run_with_rotation contract as call_gemini: a per-day
    # error on every key rotates once (2 keys) then raises QuotaExhaustedError (fail-closed). This is
    # the whole point of routing thinking through the spine instead of a bare client.
    with patch.object(r, "_KEYS", ["k1", "k2"]), patch.object(r, "_key_idx", 0):
        boom = Exception("429 RESOURCE_EXHAUSTED PerDay")
        with patch.object(r, "_raw_thinking", side_effect=boom) as mock_think, \
             patch.object(r, "_refresh_clients"):
            with pytest.raises(r.QuotaExhaustedError):
                r.call_thinking("m", [], object, 100, 4000)
            assert mock_think.call_count == 2   # key1, rotate, key2, then give up


def test_raw_chat_uses_openai_style_max_tokens_key():
    # REGRESSION GUARD: instructor's GENAI_TOOLS mode translates generation_config via an OpenAI->Gemini
    # map keyed on "max_tokens" (NOT "max_output_tokens"). A literal max_output_tokens is silently dropped
    # and the output cap never applies. Pin that _raw_chat sends the key instructor actually honors.
    captured = {}
    class _Comp:
        def create(self, **kw): captured.update(kw); return "ok"
    class _Chat:
        completions = _Comp()
    class _Client:
        chat = _Chat()
    with patch.object(r, "_client", _Client()):
        r._raw_chat("m", [], object, 1234)
    assert captured["generation_config"] == {"max_tokens": 1234}   # the honored key, with the value


def test_raw_thinking_caps_via_max_tokens_and_budgets_via_config():
    # Same translation rule for the thinking shape: the OUTPUT cap rides generation_config={"max_tokens"},
    # the THINKING budget rides config=GenerateContentConfig(thinking_config=...) (which the mode extracts).
    captured = {}
    class _Comp:
        def create_with_completion(self, **kw): captured.update(kw); return ("p", "raw")
    class _Chat:
        completions = _Comp()
    class _Client:
        chat = _Chat()
    with patch.object(r, "_client", _Client()):
        r._raw_thinking("m", [], object, 1234, 555)
    assert captured["generation_config"] == {"max_tokens": 1234}        # cap via the honored key
    assert captured["config"].thinking_config.thinking_budget == 555    # budget via config=


def test_call_thinking_passes_through_result():
    # On success call_thinking returns _raw_thinking's value unchanged (the (parsed, raw) tuple) and
    # forwards the thinking_budget arg. No rotation needed -> _raw_thinking is called exactly once.
    sentinel = ("PARSED", "RAW")
    captured = {}
    def _raw(model, messages, response_model, max_output_tokens, thinking_budget):
        captured.update(budget=thinking_budget, model=model)
        return sentinel
    with patch.object(r, "_raw_thinking", side_effect=_raw) as mock_think:
        out = r.call_thinking("gemini-2.5-flash", [], object, 100, 4000)
    assert out is sentinel              # tuple passed straight through
    assert captured["budget"] == 4000   # thinking_budget forwarded
    assert mock_think.call_count == 1