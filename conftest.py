# This file tells pytest to add the project root to the Python path
# which resolves the "ModuleNotFoundError: No module named 'src'" error.
import pytest

@pytest.fixture
def vuln_diff():
    return "[FILE]: auth/login.py\n+ query = 'SELECT * FROM u WHERE n=' + name"