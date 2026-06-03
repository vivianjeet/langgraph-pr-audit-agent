"""Benchmark full-audit end-to-end latency over N runs on a fixed diff.

Establishes a latency baseline so later cost/caching work (LiteLLM + prompt caching) can show a
before/after improvement. It MEASURES wall-clock only; it does not optimize. Token usage is read
from the LangSmith trace per run (each Gemini call logs prompt/completion tokens) - this script
does not thread token counts through call_gemini (that's a later, LiteLLM-standardized change).

The audit nodes are async, so the graph is driven via app.astream wrapped in asyncio.run (the sync
app.stream raises "No synchronous function provided" once nodes are async).

Run: python -m scripts.bench_audit
Makes real LLM calls (one full audit per run) - keep RUNS modest.
"""
import asyncio
import statistics
import time
import uuid
from src.graph import app
from tests.test_integration import REAL_DIFF

RUNS = 5


async def _once() -> float:
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.perf_counter()
    # Measure the first pass: the SQLi fixture escalates and pauses at human_review, so the stream
    # ends there. That's the full ingest -> retrieve -> plan -> 3 audits -> synthesize cost, which
    # is what we're baselining (the human-decision + finalize tail is not LLM-bound).
    async for _ in app.astream({"audit": {"messages": [REAL_DIFF]}}, config=cfg):
        pass
    return time.perf_counter() - t0


async def main() -> None:
    print(f"Benchmarking full audit over {RUNS} runs on the sample diff...\n")
    times = []
    for i in range(RUNS):
        dt = await _once()
        times.append(dt)
        print(f"  run {i + 1}: {dt:.2f}s")
    times.sort()
    print(f"\n  min={times[0]:.2f}s  median={statistics.median(times):.2f}s  max={times[-1]:.2f}s")
    print("  (token usage: read from the LangSmith trace per run - this script times wall-clock; "
          "LangSmith holds the token ledger.)")


if __name__ == "__main__":
    asyncio.run(main())
