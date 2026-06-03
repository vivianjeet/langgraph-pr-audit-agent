# Live end-to-end test (calls Gemini). Marked 'integration' so unit runs can skip it:
#   pytest -m "not integration"   # fast, $0
#   pytest -m integration         # full, costs a few cents
import asyncio
import uuid
import pytest
from src.graph import app

pytestmark = pytest.mark.integration

# A realistic diff: an auth change that introduces a raw-SQL f-string (SQL injection).
REAL_DIFF = """
diff --git a/banking/auth/login.py b/banking/auth/login.py
index 83db48f..f9a2498 100644
--- a/banking/auth/login.py
+++ b/banking/auth/login.py
@@ -10,6 +10,7 @@
 def authenticate_user(username, password):
-    query = "SELECT * FROM users WHERE username = %s AND password = %s"
-    cursor.execute(query, (username, password))
+    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
+    cursor.execute(query)
     return cursor.fetchone()
"""

def test_sql_injection_pr_is_escalated():
    from src.llm_retry import QuotaExhaustedError

    # The audit nodes are async, so drive the graph via the async API (astream/aget_state).
    # A fresh thread_id per attempt is REQUIRED: reusing one would resume the interrupted
    # checkpoint instead of running a clean pass.

    # Stream until the graph either finishes or pauses (interrupt_before human_review).
    # The audit nodes are async, so drive the graph via the async API (astream/aget_state).
    # If all API keys are daily-exhausted the graph fails closed - skip rather than error,
    # so a quota day doesn't look like a code failure.
    async def _drive():
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        initial = {"audit": {"messages": [REAL_DIFF]}}
        async for _ in app.astream(initial, config=config):
            pass
        snapshot = await app.aget_state(config)
        return await app.aget_state(config)

    
    try:
        snapshot = asyncio.run(_drive())
    except QuotaExhaustedError as e:
        pytest.skip(f"All Gemini keys daily-exhausted: {e}")

    final = snapshot.values.get("audit", {})

    print("\n=== FINAL STATE ===")
    print("paused before:", snapshot.next)
    print("security_score:", final.get("security_score"))
    print("security_findings:", final.get("security_findings"))
    print("files_changed:", final.get("files_changed"))
    for m in final.get("messages", []):
        print("-", str(m)[:200])

    # The contract: a malicious f-string SQL change on an auth file MUST NOT pass clean -
    # it must ESCALATE to human review. That's deterministic and fail-closed-safe (it holds
    # even if a transient 5xx zeroes one audit dimension's findings). We do NOT assert a
    # specific security_findings count: whether the LLM emits a structured finding on a given
    # call is non-deterministic, but escalation is the actual guarantee.
    assert "banking/auth/login.py" in final["files_changed"]   # deterministic regex parse
    assert final["security_score"] < 0.7                       # not a clean pass
    assert snapshot.next == ("human_review",)                  # graph paused for a human
