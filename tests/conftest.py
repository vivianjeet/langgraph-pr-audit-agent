"""Shared pytest fixtures for the whole suite (no tests live here).

- _clear_embed_cache (autouse): clears the per-process embedding memo before each test, so a
  patched `embed` in one test can't be shadowed by a value cached in another.
"""
import pytest
from src.db import vectorstore as vs


@pytest.fixture(autouse=True)
def _clear_embed_cache():
    """vectorstore.embed is memoised per-process. Clear it around every test so a
    value cached under a real/patched embed in one test can't shadow another test's
    expectations (e.g. call-count asserts on a patched embed)."""
    vs.embed_cache_clear()
    yield
    vs.embed_cache_clear()
