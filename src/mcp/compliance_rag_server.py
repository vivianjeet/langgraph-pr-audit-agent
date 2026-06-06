# MCP server exposing the compliance RAG over stdio.
from mcp.server.fastmcp import FastMCP
from src.db.vectorstore import search_compliance, retrieve_similar_prs
import logging
import os
import sys

# Diagnostics go to STDERR, never stdout: stdout is the JSON-RPC channel the MCP client reads,
# so any print/stdout write corrupts the protocol. stderr is captured by the client (shown in
# your terminal when running mcp_test_client.py; written to Claude Desktop's MCP log files).
# Quiet by default; set MCP_DEBUG=1 for the per-call in/out lines.
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("MCP_DEBUG") else logging.WARNING,
    stream=sys.stderr,
    format="%(asctime)s %(levelname)s compliance-rag %(message)s",
)
log = logging.getLogger("compliance-rag")

mcp = FastMCP("compliance-rag")

@mcp.tool()
def search_compliance_docs(query: str, k: int = 3, framework: str | None = None) -> list[dict]:
    """Search the multi-framework compliance corpus for passages relevant to `query`.
    Covers many regulations - RBI (banking), HIPAA (healthcare), PCI-DSS (cards), OWASP
    (app security), GDPR/DPDP (privacy), and any others installed as rule packs. Pass
    `framework` (e.g. 'hipaa') to restrict to one; omit it to search all.
    Returns up to `k` passages, each {text, source, framework, similarity}, cosine-ranked,
    above a similarity threshold. Use this to ground a security or compliance finding in an
    actual regulatory clause - e.g. before flagging unmasked PII logging, find the rule it breaks."""
    log.debug("search_compliance_docs q=%r k=%d framework=%s", query, k, framework)
    hits = search_compliance(query, k=k, framework=framework)
    log.debug("search_compliance_docs -> %d hits (frameworks=%s)",
              len(hits), sorted({h.get("framework") for h in hits}))
    return hits


@mcp.tool()
def get_pr_audit_history(query: str, k: int = 3) -> list[dict]:
    """Return up to `k` past PR audits similar to `query` (the agent's semantic memory).
    Lets an external MCP client reuse this agent's audit history without importing its code.
    Returns {pr_summary, report, similarity} per match."""
    log.debug("get_pr_audit_history q=%r k=%d", query, k)
    hits = retrieve_similar_prs(query, k=k)
    log.debug("get_pr_audit_history -> %d hits", len(hits))
    return hits


if __name__ == "__main__":
    mcp.run()        # stdio transport by default; the client spawns this as `python -m ...`