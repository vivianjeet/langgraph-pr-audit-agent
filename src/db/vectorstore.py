# pgvector access layer: ember text, init schema (HNSW), 
# store + retrieve similar PR audits.

import os
import json
import psycopg
from pgvector.psycopg import register_vector
from src.llm_retry import call_embed, QuotaExhaustedError
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
EMBED_MODEL = "gemini-embedding-001" # 768-dim by default
EMBED_DIM = 768

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100
SIM_THRESHOLD = 0.7 # cosine similarity; distance = 1 - sim, so we keep distance < 0.3

def embed(text: str) -> list[float]:
    """
    Return a 768-dim embedding for `test` using Gemini's embedding model
    """
    resp = call_embed(
        model=EMBED_MODEL,
        contents=[text],
        output_dim=EMBED_DIM
    )
    return list(resp.embeddings[0].values)

# Gemini's embed endpoint caps how many texts ( and total tokens) one call accepts,
# so we send the corpus in groups of EMBED_BATCH instead of one giant request
EMBED_BATCH = 100

def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed many texts with the fewest network round-trips
    
    Single-text `embed()` does one HTTP call per text wasteful for large corpus
    This batches `contents` (one call per EMBED_BATCH texts) and preserves input order
    result[i] is the vector for texts[i]. Returns [] for every empty input
    """

    if not texts:
        return []
    
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        resp = call_embed(
            model=EMBED_MODEL,
            contents=batch,
            output_dim=EMBED_DIM
        )

        # Defensive : the API must return one embedding per input, in order.
        if len(resp.embeddings) != len(batch):
            raise RuntimeError(
                f"embed_batch: sent {len(batch)} texts, got {len(resp.embeddings)} vectors"
            )
        vectors.extend(list(e.values) for e  in resp.embeddings)
    return vectors

def init_schema():
    """
    Create the extension, table, and HNSW index. Idempotent - safe to re run.
    """
    with psycopg.connect(DATABASE_URL, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS pr_audits (
                id          BIGSERIAL PRIMARY KEY,
                pr_summary  TEXT NOT NULL,
                report      JSONB NOT NULL,
                embedding   VECTOR({EMBED_DIM}) NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS pr_audits_embedding_hnsw
            ON pr_audits USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
            """
        )
        conn.commit()

def get_conn():
    """
    Open a pgvector-registered connection. Caller is responsible for closing.
    """
    conn = psycopg.connect(DATABASE_URL, connect_timeout=5)
    register_vector(conn)
    return conn

def store_pr_audit(pr_summary: str, report: dict, embed_text: str | None = None) -> None:
    """
    Persists a finished audit so future PRs can retrieve it as precedent
    `embed_text` is what gets vectorised (default: the diff content that the
    retrieve node also embeds, so storage and query use the SAME representation).
    `pr_summary` stays the human-readable label in its own column
    """
    vec = embed(embed_text if embed_text is not None else pr_summary)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pr_audits (pr_summary, report, embedding) VALUES (%s, %s, %s);",
            (pr_summary, json.dumps(report), vec),
        )
        conn.commit()

def retrieve_similar_prs(query_text: str, k: int = 3) -> list[dict]:
    """
    Return upto `k` past audits with cosine similarity > SIM_THRESHOLD.
    """
    vec = embed(query_text)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
        cur.execute(
            """
            SELECT pr_summary, report, 1 - (embedding <=> %s::vector) AS similarity
            FROM pr_audits
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vec, vec, k),
        )
        rows = cur.fetchall()
    
    return [
        {"pr_summary": r[0], "report": r[1], "similarity": float(r[2])}
        for r in rows
        if float(r[2]) > SIM_THRESHOLD
    ]

if __name__ == "__main__":
    # `python -m src.db.vectorstore` initialises the schema once
    init_schema()
    print("pgvector schema initialised.")
