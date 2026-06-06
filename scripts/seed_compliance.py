# Loads EVERY packs/*.yaml file. Adding a framework = dropping a YAML file here. No code change.
import glob
import os
import yaml
from src.db.vectorstore import embed_batch, get_conn

PACKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packs")


def _load_packs() -> list[tuple[str, str, str]]:
    """Read every packs/*.yaml -> list of (content, source, framework). Fail LOUD on a malformed
    pack (missing framework / docs) - a silently-skipped pack is a silently-missing regulation."""
    rows = []
    for path in sorted(glob.glob(os.path.join(PACKS_DIR, "*.yaml"))):
        with open(path, encoding="utf-8") as f:
            pack = yaml.safe_load(f)
        framework = pack["framework"]                    # KeyError = malformed pack, intended
        for doc in pack["docs"]:
            rows.append((doc["content"].strip(), doc["source"], framework))
    return rows


def main():
    docs = _load_packs()
    frameworks = sorted({f for _, _, f in docs})
    with get_conn() as conn, conn.cursor() as cur:
        # Dedup FIRST (idempotent), then embed only the new docs in ONE batched pass.
        # Per-doc embed() here floods the embedding endpoint's per-minute quota on a large
        # corpus -> RESOURCE_EXHAUSTED; embed_batch chunks at EMBED_BATCH per round-trip.
        new_docs = []
        for content, source, framework in docs:
            cur.execute("SELECT 1 FROM compliance_docs WHERE content = %s;", (content,))
            if cur.fetchone():
                continue                                 # idempotent: skip already-seeded
            new_docs.append((content, source, framework))

        vectors = embed_batch([content for content, _, _ in new_docs])
        for (content, source, framework), embedding in zip(new_docs, vectors):
            cur.execute(
                "INSERT INTO compliance_docs (content, source, framework, embedding) "
                "VALUES (%s, %s, %s, %s);",
                (content, source, framework, embedding),
            )
        added = len(new_docs)
        conn.commit()
    print(f"Seeded compliance corpus: {added} new doc(s), {len(docs)} total across "
          f"{len(frameworks)} frameworks ({', '.join(frameworks)}).")

if __name__ == "__main__":
    main()