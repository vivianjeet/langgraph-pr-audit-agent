# MCP server exposing the compliance RAG over stdio.
from mcp.server.fastmcp import FastMCP
from src.db.vectorstore import search_compliance, retrieve_similar_prs

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
    return search_compliance(query, k=k, framework=framework)


@mcp.tool()
def get_pr_audit_history(query: str, k: int = 3) -> list[dict]:
    """Return up to `k` past PR audits similar to `query` (the agent's semantic memory).
    Lets an external MCP client reuse this agent's audit history without importing its code.
    Returns {pr_summary, report, similarity} per match."""
    return retrieve_similar_prs(query, k=k)


if __name__ == "__main__":
    mcp.run()        # stdio transport by default; the client spawns this as `python -m ...`