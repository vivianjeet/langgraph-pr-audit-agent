import pytest
from src.nodes.ingest import parse_github_diff, ingest_pr_node
from src.state import AuditState

sample_diff = """
diff --git a/payment_gateway.py b/payment_gateway.py
index 8329a4f..9b3c1d2 100644
--- a/payment_gateway.py
+++ b/payment_gateway.py
@@ -10,3 +10,4 @@
 def process_payment(account_id, amount):
-    query = "SELECT balance FROM accounts WHERE id = '" + account_id + "'"
+    query = "SELECT balance FROM accounts WHERE id = %s"
+    db.execute(query, (account_id,))
"""

def test_parse_githuib_diff_extracts_correct_lines():
    parsed, files_changed = parse_github_diff(sample_diff)

    assert "[FILE MODIFIED]: payment_gateway.py" in parsed
    assert "[REMOVED]:     query = \"SELECT balance FROM accounts WHERE id = '\" + account_id + \"'\"" in parsed
    assert "[ADDED]:     query = \"SELECT balance FROM accounts WHERE id = %s\"" in parsed
    assert "@@ -10,3 +10,4 @@" not in parsed
    assert "payment_gateway.py" in files_changed
    assert len(files_changed) == 1

def test_ingest_pr_node_state_update():
    mock_diff = """diff --git a/test.py b/test.py\n--- a/test.py\n+++ b/test.py\n+print('Hello Context')"""
    mock_state = {"messages": [mock_diff], 
                  "parsed_diff": "", 
                  "files_changed": ["payment_gateway.py"]
                  }

    result = ingest_pr_node(mock_state)

    assert "messages" in result
    assert len(result["messages"]) == 1
    assert "System: Ingested PR data. Extracted changes:" in result["messages"][0]
    assert "[ADDED]: print('Hello Context')" in result["messages"][0]
    assert "files_changed" in result
    assert len(result["files_changed"]) == 1
    assert "payment_gateway.py" not in result["files_changed"]
