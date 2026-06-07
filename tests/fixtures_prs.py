#  5 representative diffs for the integration pass + README numbers.
from tests.test_integration import REAL_DIFF

SQLI = REAL_DIFF
PII_LOGGING = "diff --git a/app/log.py b/app/log.py\n@@\n+    logger.info(f\"user {user.pan} logged in\")\n"
AUTH_CHANGE = "diff --git a/auth/login.py b/auth/login.py\n@@\n+    if user.role == 'admin':\n+        skip_mfa = True\n"
CLEAN_REFACTOR = "diff --git a/util.py b/util.py\n@@\n-def calc(x):\n+def compute(x):\n     return x * 2\n"
QUALITY_GODOBJECT = "diff --git a/svc.py b/svc.py\n@@\n+class Service:\n" + "".join(
    f"+    def m{i}(self): ...\n" for i in range(60))

ALL = {"sqli": SQLI, "pii": PII_LOGGING, "auth": AUTH_CHANGE,
       "clean": CLEAN_REFACTOR, "quality": QUALITY_GODOBJECT}