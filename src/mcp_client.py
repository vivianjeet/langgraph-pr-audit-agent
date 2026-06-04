# src/mcp_client.py - stand up external MCP servers and expose their tools to the agent.
# Servers (stdio transport = each is a subprocess we talk to over stdin/stdout):
#   - filesystem (read-only): repo file context, a server we did NOT write.
#   - compliance (OURS, src/mcp/compliance_rag_server.py): search_compliance_docs 
#   + get_pr_audit_history.
# Tools return as LangChain StructuredTools via langchain-mcp-adapters 
#   - directly awaitable.
import os
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

log = logging.getLogger(__name__)

# repo root = two levels up from this file (src/ -> repo/). The filesystem server is scoped to it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _server_specs() -> dict:
    """Launch specs for the MCP servers. stdio: each server is a child process."""
    return {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", REPO_ROOT],
            "transport": "stdio",
        },
        "compliance": {
            # run OUR server as a module so its `from src...` imports resolve.
            "command": "python",
            "args": ["-m", "src.mcp.compliance_rag_server"],
            "transport": "stdio",
        },
    }

async def load_mcp_tools() -> list:
    """Start the configured MCP servers and return their tools as LangChain tools.
    Fail-SOFT: if a server can't start (npx missing, DB down, our server errors), return the
    tools that DID load - the audit must never crash because an optional tool source is down.
    Returns [] only if nothing loaded at all."""
    try:
        client = MultiServerMCPClient(_server_specs())
        return await client.get_tools()
    except Exception as e:
        log.warning("MCP tool load failed (%s); audit continues tool-less.", e)
        return []
