# Pins the opt-in durable checkpointer (--durable): durable_app must yield a graph backed by
# AsyncSqliteSaver (the async saver, since the audit graph is async-driven), carrying the SAME
# allow-listed serde as the default in-RAM app, and its threads must survive a process restart
# (modelled here as closing the connection and reopening a FRESH one on the same file).
# Hermetic: a throwaway 1-node graph + a temp SQLite file. No LLM, no Postgres.
import asyncio
import os
import tempfile
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import src.graph as graph


def _tmp_db() -> str:
    # A real filesystem path (not /tmp, which the aiosqlite worker can't open on Windows).
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    os.remove(path)          # we want the path, not the empty file; the saver creates it.
    return path


def test_durable_app_uses_async_sqlite_saver_with_our_serde():
    async def _check():
        async with graph.durable_app(_tmp_db()) as app:
            cp = app.checkpointer
            # The async saver specifically: the sync SqliteSaver would raise on app.astream.
            assert isinstance(cp, AsyncSqliteSaver)
            # Same serializer object as the in-RAM default -> domain types round-trip identically
            # (Severity / *Finding survive the SQLite hop as real objects, not degraded dicts).
            assert cp.serde is graph.serde
    asyncio.run(_check())


def test_durable_app_preserves_human_review_interrupt():
    # Durability must not change topology: the human-review hard stop is still wired.
    async def _check():
        async with graph.durable_app(_tmp_db()) as app:
            assert "human_review" in app.interrupt_before_nodes
    asyncio.run(_check())


def test_thread_state_survives_a_fresh_reopen():
    # The whole point of --durable: state written by one connection is readable by a LATER one.
    class S(TypedDict):
        n: int

    builder = StateGraph(S)
    builder.add_node("inc", lambda s: {"n": s["n"] + 1})
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)

    db = _tmp_db()
    cfg = {"configurable": {"thread_id": "t1"}}

    async def _write():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            await builder.compile(checkpointer=saver).ainvoke({"n": 41}, cfg)
        # connection closed here -> the checkpoint must now live on disk, not in memory.

    async def _reopen():
        async with AsyncSqliteSaver.from_conn_string(db) as saver:
            return (await builder.compile(checkpointer=saver).aget_state(cfg)).values

    asyncio.run(_write())
    vals = asyncio.run(_reopen())        # fresh connection, fresh event loop
    assert vals == {"n": 42}, vals
    os.remove(db)
