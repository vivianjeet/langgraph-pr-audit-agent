"""Shared pytest fixtures for the whole suite (no tests live here).

- _clear_embed_cache (autouse): clears the per-process embedding memo before each test, so a
  patched `embed` in one test can't be shadowed by a value cached in another.
- _disable_langfuse (autouse): strips LANGFUSE_* env so tracing is a no-op and no test makes a
  real network call. run_audit now opens a trace + flushes on exit (live HTTP when keys are set);
  without this a test driving run_audit BLOCKS on shutdown()/flush and hangs the suite.
"""
import pytest
from src.db import vectorstore as vs


@pytest.fixture(autouse=True)
def _disable_langfuse(monkeypatch):
    """Observability is off the critical path, so disabling it changes no tested behaviour - but
    a live LANGFUSE_* in the developer's .env would make audit_trace / score_audit / flush_traces
    do real (blocking) HTTP. Remove the keys + reset the cached client so _langfuse() -> None."""
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "LANGFUSE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    import src.llm_client as llm
    monkeypatch.setattr(llm, "_LANGFUSE", None, raising=False)


@pytest.fixture(autouse=True)
def _clear_embed_cache():
    """vectorstore.embed is memoised per-process. Clear it around every test so a
    value cached under a real/patched embed in one test can't shadow another test's
    expectations (e.g. call-count asserts on a patched embed)."""
    vs.embed_cache_clear()
    yield
    vs.embed_cache_clear()
