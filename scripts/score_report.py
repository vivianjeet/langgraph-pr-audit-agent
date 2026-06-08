"""Per-dimension audit-score view over the pr_audits table - the authoritative store, NOT Langfuse.

Langfuse Scores only surface a cross-run AVERAGE (and averaging scores of DIFFERENT diffs is a
near-meaningless metric). The questions you actually ask - "what did the latest audit score?",
"how are the last N runs trending?", "how does this branch compare?" - are precise SQL over
pr_audits.report, which already holds security_score / quality_score / test_score per run.

Usage:
    python -m scripts.score_report            # latest + last-10 moving average
    python -m scripts.score_report --n 20     # window size for the moving average
    python -m scripts.score_report --by-branch  # average per branch
    python -m scripts.score_report --hist     # per-dimension histogram (0.2-wide buckets)
"""
import argparse

_DIMS = ("security_score", "quality_score", "test_score")

# Histogram bucket edges. width_bucket(score, 0, 1, 5) maps [0,1] into 5 equal bins; the top
# edge 1.0 lands in an overflow bucket (6) which we fold back into the last bin so a perfect
# 1.0 counts as [0.8-1.0].
_HIST_LABELS = ("[0.0-0.2)", "[0.2-0.4)", "[0.4-0.6)", "[0.6-0.8)", "[0.8-1.0]")


def _rows(n: int):
    """The newest N audits as (id, created_at, {dim: score}). Reads scores out of the JSONB
    report column so it works regardless of how the row was written."""
    from src.db.vectorstore import get_conn
    sel = ", ".join(f"(report->>'{d}')::float AS {d}" for d in _DIMS)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, created_at, {sel} FROM pr_audits ORDER BY id DESC LIMIT %s;",
            (n,),
        )
        out = []
        for r in cur.fetchall():
            out.append((r[0], r[1], dict(zip(_DIMS, r[2:]))))
        return out


def _fmt(v):
    return "  --  " if v is None else f"{v:>6.2f}"


def latest_and_moving(n: int):
    rows = _rows(n)
    if not rows:
        print("No audits in pr_audits yet - run an audit first.")
        return

    print(f"{'dimension':16s}{'latest':>8s}{f'avg(last {len(rows)})':>16s}")
    print("-" * 40)
    latest = rows[0][2]
    for d in _DIMS:
        vals = [r[2][d] for r in rows if r[2][d] is not None]
        avg = sum(vals) / len(vals) if vals else None
        print(f"{d:16s}{_fmt(latest[d])}{_fmt(avg):>16s}")
    print(f"\nlatest audit: id={rows[0][0]}  at {rows[0][1]:%Y-%m-%d %H:%M}")


def by_branch():
    """Average per dimension grouped by the branch label stored in report (if present)."""
    from src.db.vectorstore import get_conn
    sel = ", ".join(f"AVG((report->>'{d}')::float) AS {d}" for d in _DIMS)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT COALESCE(report->>'branch', '(unlabelled)') AS branch, COUNT(*), {sel} "
            f"FROM pr_audits GROUP BY branch ORDER BY COUNT(*) DESC;"
        )
        rows = cur.fetchall()
    if not rows:
        print("No audits in pr_audits yet.")
        return
    print(f"{'branch':24s}{'runs':>6s}" + "".join(f"{d.split('_')[0]:>10s}" for d in _DIMS))
    print("-" * 60)
    for branch, cnt, *avgs in rows:
        print(f"{branch[:24]:24s}{cnt:>6d}" + "".join(_fmt(a).rjust(10) for a in avgs))


def histogram():
    """Per-dimension distribution of scores into 0.2-wide buckets, as ASCII bars. The buckets
    Langfuse's Scores table can't give for a NUMERIC score - computed here over pr_audits with
    Postgres width_bucket."""
    from src.db.vectorstore import get_conn
    # For each dimension: count rows per bucket. LEAST folds the 1.0 overflow bucket (6) into 5.
    with get_conn() as conn, conn.cursor() as cur:
        per_dim = {}
        for d in _DIMS:
            cur.execute(
                f"SELECT LEAST(width_bucket((report->>'{d}')::float, 0, 1, 5), 5) AS b, COUNT(*) "
                f"FROM pr_audits WHERE report ? '{d}' GROUP BY b ORDER BY b;"
            )
            counts = {int(b): c for b, c in cur.fetchall()}
            per_dim[d] = [counts.get(i, 0) for i in range(1, 6)]   # buckets 1..5

    total = sum(sum(v) for v in per_dim.values())
    if total == 0:
        print("No audits in pr_audits yet - run an audit first.")
        return

    peak = max((max(v) for v in per_dim.values()), default=0) or 1
    for d in _DIMS:
        n = sum(per_dim[d])
        print(f"\n{d}  (n={n})")
        for label, cnt in zip(_HIST_LABELS, per_dim[d]):
            bar = "#" * round(cnt / peak * 30)
            print(f"  {label}  {bar:<30s} {cnt}")


def main():
    ap = argparse.ArgumentParser(description="Per-dimension audit-score report over pr_audits.")
    ap.add_argument("--n", type=int, default=10, help="window size for the moving average")
    ap.add_argument("--by-branch", action="store_true", help="average per branch instead")
    ap.add_argument("--hist", action="store_true", help="per-dimension histogram (0.2-wide buckets)")
    args = ap.parse_args()
    if args.hist:
        histogram()
    elif args.by_branch:
        by_branch()
    else:
        latest_and_moving(args.n)


if __name__ == "__main__":
    main()
