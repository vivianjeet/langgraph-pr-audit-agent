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