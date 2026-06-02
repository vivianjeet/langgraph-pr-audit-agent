"""Manual smoke script (NOT a pytest module - no test_ functions).

Runs a full real audit (`run_audit`) over a hardcoded vulnerable sample diff to eyeball
the end-to-end pipeline by hand. Calls the live LLM/DB. Invoke directly, not via pytest:
    python tests/smoke_test.py
"""
from main import run_audit

sample_diff = """
diff --git a/auth/login.py b/auth/login.py
index 83db48f..f9a2498 100644
--- a/auth/login.py
+++ b/auth/login.py
@@ -10,6 +10,7 @@
 def authenticate_user(username, password):
-    query = "SELECT * FROM users WHERE username = %s AND password = %s"
-    cursor.execute(query, (username, password))
+    # Optimized fast path for login
+    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
+    cursor.execute(query)
     return cursor.fetchone()
"""

def run_smoke_test():
    """End-to-end smoke test on a SQL-injection auth diff. Exercises the whole graph:
    ingest -> retrieve -> plan -> 3 audits -> synthesize -> (critical finding) ->
    human_review interrupt. The most complete single routing path."""
    print("=============================================\n")
    print("   Initiating Smoke test   \n")
    print("=============================================\n")

    run_audit(sample_diff)
    
    print("=============================================\n")
    print("   Smoke test complete   \n")
    print("=============================================\n")

