"""One-shot latency breakdown: per-MCP-server startup, per-node, total. Times wall-clock only."""
import asyncio, time, uuid
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.mcp_client import _server_specs
from src.graph import app
from tests.test_integration import REAL_DIFF


async def time_mcp_servers() -> dict:
    specs = _server_specs()
    out = {}
    for name, spec in specs.items():
        t0 = time.perf_counter()
        try:
            tools = await MultiServerMCPClient({name: spec}).get_tools()
            out[name] = (time.perf_counter() - t0, len(tools))
        except Exception as e:
            out[name] = (time.perf_counter() - t0, f"FAILED: {type(e).__name__}: {e}")
    return out


async def time_nodes_and_total():
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    node_times = {}
    t_start = time.perf_counter()
    last = t_start
    async for event in app.astream({"audit": {"messages": [REAL_DIFF]}}, config=cfg):
        now = time.perf_counter()
        for node_name in event:
            node_times[node_name] = now - last
        last = now
    return node_times, time.perf_counter() - t_start


async def main():
    print("=== MCP server startup (each spawned alone) ===")
    for name, (dt, n) in (await time_mcp_servers()).items():
        print(f"  {name:14s} {dt:7.2f}s  ({n} tools)")

    print("\n=== Per-node delta + total (one audit run) ===")
    node_times, total = await time_nodes_and_total()
    for node, dt in node_times.items():
        print(f"  {node:18s} {dt:7.2f}s")
    print(f"  {'TOTAL':18s} {total:7.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
