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

sample_diff_no_human = """
diff --git a/src/services/order_service.py b/src/services/order_service.py
index 1a2b3c4..5d6e7f8 100644
--- a/src/services/order_service.py
+++ b/src/services/order_service.py
@@ -12,6 +12,18 @@ class OrderService:
     def __init__(self, repo):
         self.repo = repo
 
+    def apply_discount(self, order_total, customer_tier):
+        # Magic numbers, no validation, deeply nested branching.
+        if customer_tier == 1:
+            if order_total > 1000:
+                return order_total * 0.85
+            else:
+                return order_total * 0.95
+        elif customer_tier == 2:
+            return order_total * 0.90
+        else:
+            return order_total
+
+    def process_refund(self, order_id, amount):
+        try:
+            order = self.repo.get(order_id)
+            order.balance = order.balance - amount
+            self.repo.save(order)
+        except Exception:
+            pass  # swallow all errors silently
"""

def run_smoke_test_without_human():
    """ Executes a smoke test with mock SQL - injection PR diff - deflects to reflexion"""
    print("=============================================\n")
    print("   Initiating Smoke test without human intervention  \n")
    print("=============================================\n")

    run_audit(sample_diff_no_human)

    print("=============================================\n")
    print("   Smoke test complete - no human intervention   \n")
    print("=============================================\n")

def run_smoke_test():
    """ Executes a smoke test with mock SQL - injection PR diff - deflects to human intervention"""
    print("=============================================\n")
    print("   Initiating Smoke test   \n")
    print("=============================================\n")

    run_audit(sample_diff)
    
    print("=============================================\n")
    print("   Smoke test complete   \n")
    print("=============================================\n")

