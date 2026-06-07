"""Pre-flight check for the compliance corpus: is it seeded, and is every pack present?
Catches the two integration-pass gotchas in one glance:
  - forgot seed_compliance -> empty table -> compliance_hits=0 across the board
  - a pack silently failed to load -> one framework missing / uneven counts
Run: python -m scripts.check_corpus
"""
import os
from dotenv import load_dotenv
import psycopg

load_dotenv()


def main():
    with psycopg.connect(os.environ["DATABASE_URL"]) as c:
        rows = c.execute(
            "SELECT framework, count(*) FROM compliance_docs GROUP BY framework ORDER BY 1"
        ).fetchall()
    total = sum(n for _, n in rows)
    print(f"compliance_docs: {total} docs across {len(rows)} frameworks")
    for fw, n in rows:
        print(f"  {fw:10s} {n}")
    if not rows:
        print("EMPTY - run `python -m scripts.seed_compliance`")


if __name__ == "__main__":
    main()
