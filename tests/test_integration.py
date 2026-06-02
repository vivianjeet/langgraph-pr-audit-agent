# Live end-to-end test (calls Gemini). Marked 'integration' so unit runs can skip it:
#   pytest -m "not integration"   # fast, $0
#   pytest -m integration         # full, costs a few cents
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
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial = {"audit": {"messages": [REAL_DIFF]}}

    # Stream until the graph either finishes or pauses (interrupt_before human_review).
    # If all API keys are daily-exhausted the graph fails closed - skip rather than error,
    # so a quota day doesn't look like a code failure.
    try:
        for _ in app.stream(initial, config=config):
            pass
    except QuotaExhaustedError as e:
        pytest.skip(f"All Gemini keys daily-exhausted: {e}")

    final = app.get_state(config).values.get("audit", {})
    print("\n=== FINAL STATE ===")
    print("security_score:", final.get("security_score"))
    print("security_findings:", final.get("security_findings"))
    print("files_changed:", final.get("files_changed"))
    print("parsed_diff:", repr(final.get("parsed_diff"))[:300])
    print("=== MESSAGES ===")
    for m in final.get("messages", []):
        print("-", str(m)[:200])
    # An f-string SQL query on an auth file MUST be caught and escalated, not silently finalized.
    assert final["security_score"] < 0.7
    assert "banking/auth/login.py" in final["files_changed"]
    assert len(final["security_findings"]) >= 1