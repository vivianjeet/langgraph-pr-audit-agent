# run the 5 fixtures end-to-end, print a results table for the README.
import asyncio, sys, time, uuid
from src.graph import app
from tests.fixtures_prs import ALL


async def _run(name, diff):
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    t0 = time.perf_counter()
    async for _ in app.astream({"audit": {"messages": [diff]}}, config=cfg):
        pass
    dt = time.perf_counter() - t0
    snap = (await app.aget_state(cfg)).values
    audit = snap.get("audit", {})
    escalated = bool((await app.aget_state(cfg)).next)   # paused at human_review?

    # With --findings, print every finding's dimension + severity + title - handy for seeing
    # WHICH finding (and which dimension) drove an escalation. Off by default so the table is clean.
    if "--findings" in sys.argv:
        for dim in ("security_findings", "quality_findings", "test_findings"):
            for f in audit.get(dim, []):
                print(f"  [{name}] {dim}: severity={f.severity} title={getattr(f, 'title', '')!r}")

    return {
        "name": name, "secs": round(dt, 1),
        "sec_findings": len(audit.get("security_findings", [])),
        "compliance_hits": len(audit.get("compliance_context", [])),
        "citations": len(audit.get("compliance_citations", [])),
        "escalated": escalated,
        "sec_score": audit.get("security_score"),
    }


async def main():
    rows = [await _run(n, d) for n, d in ALL.items()]
    hdr = f"{'PR':8s}{'secs':>6s}{'secF':>6s}{'compl':>7s}{'cites':>7s}{'escal':>7s}{'secScore':>10s}"
    print(hdr)
    for r in rows:
        print(f"{r['name']:8s}{r['secs']:6.1f}{r['sec_findings']:6d}{r['compliance_hits']:7d}"
              f"{r['citations']:7d}{str(r['escalated']):>7s}{str(r['sec_score']):>10s}")

if __name__ == "__main__":
    asyncio.run(main())