# spawn compliance_rag_server over stdio, list + call its tools
# via the RAW mcp client SDK (not the langchain adapter). Proves the server is MCP-spec-compliant
# and callable without LangGraph.   python -m scripts.mcp_test_client
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import sys
import os

async def main():
    verbose = "--v" in sys.argv or "--verbose" in sys.argv
    print(f"[client] verbose={verbose}  (sys.argv={sys.argv})", file=sys.stderr)
    SERVER = StdioServerParameters(
        command="python", 
        args=["-m", "src.mcp.compliance_rag_server"],
        env={**os.environ, **({"MCP_DEBUG": "1"} if verbose else {})}
    )
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()                       # MCP handshake

            tools = await session.list_tools()
            print("Tools exposed by compliance-rag:")
            for t in tools.tools:
                print(f"  - {t.name}: {t.description.splitlines()[0]}")

            print("\nCalling search_compliance_docs('PII logging', k=3)  [all frameworks]:")
            result = await session.call_tool("search_compliance_docs", {"query": "PII logging", "k": 3})
            for block in result.content:
                print("  ", getattr(block, "text", block))

            print("\nCalling search_compliance_docs('PHI in logs', k=2, framework='hipaa')  [one pack]:")
            result = await session.call_tool(
                "search_compliance_docs", {"query": "PHI in logs", "k": 2, "framework": "hipaa"})
            for block in result.content:
                print("  ", getattr(block, "text", block))


if __name__ == "__main__":
    asyncio.run(main())