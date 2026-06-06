# pgvector access layer: ember text, init schema (HNSW), 
# store + retrieve similar PR audits.

import os
import json
from functools import lru_cache
import psycopg
from pgvector.psycopg import register_vector
from src.llm_retry import call_embed, QuotaExhaustedError
from dotenv import load_dotenv
from src.state import RuleStatus, RuleCategory

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
EMBED_MODEL = "gemini-embedding-001" # 768-dim by default
EMBED_DIM = 768

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 100
SIM_THRESHOLD = 0.7 # cosine similarity; distance = 1 - sim, so we keep distance < 0.3

# Build the SQL CHECK list from the enum so the column constraint and the
# Python type can never drift apart.
_RULE_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in RuleStatus)
_ACTIVE_RULE_STATUSES = ", ".join(f"'{s.value}'" for s in (RuleStatus.SEEDED, RuleStatus.LEARNED_APPROVED))
_RULE_CATEGORY_VALUES = ", ".join(f"'{c.value}'" for c in RuleCategory)

# --- Destructive: drop all memory tables ---
_MEMORY_TABLES = ("pr_audits", "session_episodes", "procedural_rules", "compliance_docs")

# ANSI colour codes for the terminal danger banner.
_RED = "\033[91m"
_BOLD = "\033[1m"
_RESET = "\033[0m"

@lru_cache(maxsize=256)
def _embed_cached(text: str) -> tuple[float, ...]:
    """Memoised single-text embed. Bounded (256 * ~768 floats) so it can't grow
    without limit. Returns an immutable tuple so a cached vector can't be mutated
    by one caller and corrupt another's hit.

    Why this matters: within ONE audit run the SAME diff is embedded by both the
    retrieve node (similarity query) and the finalize node (storage) - and the
    episodic recall embeds it again. Those are separate AMS instances (AMS is
    per-node), so the dedupe has to live HERE, at the embedding layer, not on AMS.
    """
    resp = call_embed(
        model=EMBED_MODEL,
        contents=[text],
        output_dim=EMBED_DIM
    )
    return tuple(resp.embeddings[0].values)


def embed(text: str) -> list[float]:
    """
    Return a 768-dim embedding for `text` using Gemini's embedding model.
    Cached per-process on `text` (see _embed_cached); returns a fresh list each
    call so callers are free to mutate it without touching the cache.
    """
    return list(_embed_cached(text))


def embed_cache_clear() -> None:
    """Drop the memoised embeddings (used by tests so a patched embed in one test
    can't be shadowed by a value cached in another)."""
    _embed_cached.cache_clear()

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
    Create the extension, table and HNSW index. Idempotent - safe to re run.
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

        # Episodic memory: compressed session summaries, 
        # retrievable by semantic similarity.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS session_episodes (
                id           SERIAL PRIMARY KEY,
                summary      TEXT NOT NULL,
                metadata     JSONB,
                embedding    VECTOR({EMBED_DIM}) NOT NULL,
                created_at   TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS session_episodes_embedding_hnsw
            ON session_episodes USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
            """
        )

        # Procedural memory: org rules / audit templates, 
        # fetched by category
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS procedural_rules (
                id           SERIAL PRIMARY KEY,
                category     TEXT NOT NULL
                    CHECK (category IN ({_RULE_CATEGORY_VALUES})),
                content      TEXT NOT NULL,
                status       TEXT NOT NULL
                    CHECK (status IN ({_RULE_STATUS_VALUES})),
                embedding    VECTOR({EMBED_DIM}) NOT NULL,
                source_decision TEXT,
                created_at   TIMESTAMPTZ DEFAULT now(),
                updated_at   TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS procedural_rules_category_idx
            ON procedural_rules (category);
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX procedural_rules_content_uniq
            ON procedural_rules (category, lower(content))
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS procedural_rules_embedding_hnsw
            ON procedural_rules USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
            """
        )
        # Compliance corpus: multi-framework regulatory passages (RBI, HIPAA, PCI-DSS, 
        # OWASP, GDPR, ...), retrieved by semantic similarity for the MCP 
        # search_compliance_docs tool. `framework` is a FREE-FORM tag (no CHECK) so a 
        # contributor's rule pack adds a new framework with zero schema change - 
        # the open axis of the horizontal product. Same embed+HNSW shape as pr_audits.
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS compliance_docs (
                id          SERIAL PRIMARY KEY,
                content     TEXT NOT NULL,
                source      TEXT NOT NULL,
                framework   TEXT NOT NULL,
                embedding   VECTOR({EMBED_DIM}) NOT NULL,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS compliance_docs_embedding_hnsw
            ON compliance_docs USING hnsw (embedding vector_cosine_ops)
            WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
            """
        )
        # Plain btree on framework: cheap WHERE framework = ... filtering for single-pack searches.
        cur.execute(
            "CREATE INDEX IF NOT EXISTS compliance_docs_framework ON compliance_docs (framework);"
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

def search_compliance(query_text: str, k: int = 3, framework: str | None = None) -> list[dict]:
    """Return up to `k` compliance passages with cosine similarity > SIM_THRESHOLD.
    Mirrors retrieve_similar_prs: embed the query, HNSW cosine search, strict threshold.
    `framework` is an optional filter (e.g. 'hipaa') - None searches every pack."""
    vec = embed(query_text)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
        if framework:
            cur.execute(
                """
                SELECT content, source, framework, 1 - (embedding <=> %s::vector) AS similarity
                FROM compliance_docs
                WHERE framework = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vec, framework, vec, k),
            )
        else:
            cur.execute(
                """
                SELECT content, source, framework, 1 - (embedding <=> %s::vector) AS similarity
                FROM compliance_docs
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vec, vec, k),
            )
        rows = cur.fetchall()
    return [
        {"text": r[0], "source": r[1], "framework": r[2], "similarity": float(r[3])}
        for r in rows
        if float(r[3]) > SIM_THRESHOLD
    ]

# --- Episodic memory ( session summaries ) ---
def store_episode(summary: str, metadata: dict | None = None) -> None:
    """
    Persist a compressed session summary, embedded for later semantic recall
    """
    vec = embed(summary)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO session_episodes (summary, metadata, embedding) VALUES (%s, %s, %s);",
            (summary, json.dumps(metadata or {}), vec),
        )
        conn.commit()

def retrieve_episodes(query_text: str, k: int = 3) -> list[dict]:
    """
    Return upto `k` past session summaries with cosine similarity > SIM_THRESHOLD.
    """
    vec = embed(query_text)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT summary, metadata, 1 - (embedding <=> %s::vector) AS similarity
            FROM session_episodes
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (vec, vec, k),
        )
        rows = cur.fetchall()
        return [
            {"summary": r[0], "metadata": r[1], "similarity": float(r[2])}
            for r in rows
            if float(r[2]) > SIM_THRESHOLD
        ]

# --- Procedural memory (org rules / templates) ---
def add_rule(category: RuleCategory, rule: str, status: RuleStatus,
             source_decision: str | None = None) -> None:
    """
    Store an organisational audit rule under a category (security/quality/coverage).
    The content is embedded so the review CLI can flag near-duplicate pending rules
    by cosine similarity (exact-text dedup can't catch reworded LLM phrasings).

    `source_decision` records the human's verdict on the PR this rule was learned from
    (approve/reject/needs-changes), so a reviewer sees a pending rule's provenance. None
    for seeded rules (no source audit) and for runs where no human review fired.
    """
    vec = embed(rule)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """INSERT INTO procedural_rules (category, content, status, embedding, source_decision)
            VALUES (%s, %s, %s, %s, %s);
            """,
            (category.value, rule, status.value, vec, source_decision),
        )
        conn.commit()

def get_rules(category: RuleCategory) -> list[str]:
    """Fetch ACTIVE rules for a category (most recent first)"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM procedural_rules "
            "WHERE category = %s "
            f"AND status in ({_ACTIVE_RULE_STATUSES}) "
            "ORDER BY id DESC;",
            (category.value,),
        )
        return [r[0] for r in cur.fetchall()]

def get_all_rule_contents(category: RuleCategory) -> list[str]:
    """Fetch all rules for a category (most recent first)"""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM procedural_rules "
            "WHERE category = %s "
            "ORDER BY id DESC;",
            (category.value,),
        )
        return [r[0] for r in cur.fetchall()]

# --- Procedural rule governance (offline review CLI) ---
def list_pending_rules() -> list[dict]:
    """All rules awaiting human review (learned_pending). Returns id (needed to target
    approve/reject), category, content and source_decision (the PR verdict it was learned
    from). Ordered by id so the CLI lists them stably."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, category, content, source_decision FROM procedural_rules "
            "WHERE status = %s ORDER BY id;",
            (RuleStatus.LEARNED_PENDING.value,),
        )
        return [{"id": r[0], "category": r[1], "content": r[2], "source_decision": r[3]}
                for r in cur.fetchall()]


def list_active_rules() -> list[dict]:
    """Active rules (seeded + learned_approved) so the CLI can offer 
    retire/delete on them."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"SELECT id, category, status, content FROM procedural_rules "
            f"WHERE status IN ({_ACTIVE_RULE_STATUSES}) ORDER BY id;"
        )
        return [{"id": r[0], "category": r[1], "status": r[2], "content": r[3]}
                for r in cur.fetchall()]


def set_rule_status(rule_id: int, status: RuleStatus) -> None:
    """Transition a rule by id (approve -> learned_approved, reject -> rejected,
    retire -> retired). Bumps updated_at so the row's last-touched time is honest."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE procedural_rules SET status = %s, updated_at = now() WHERE id = %s;",
            (status.value, rule_id),
        )
        conn.commit()


def delete_rule(rule_id: int) -> None:
    """Hard delete - removes the row entirely. Footgun for learned rules (they re-learn
    as pending next run because get_all_rule_contents no longer sees them); safe for seeded."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM procedural_rules WHERE id = %s;", (rule_id,))
        conn.commit()


def similar_rules(rule_id: int, k: int = 3) -> list[dict]:
    """For the review CLI's near-duplicate hint: given a pending rule's id, return up to k
    OTHER rules (any status, excluding the rule itself) most similar by cosine over the
    stored embedding. Reuses the pending rule's already-stored vector - no re-embed.

    ADVISORY ONLY. The SIM_THRESHOLD cutoff is unvalidated: it can miss reworded duplicates
    (false negative) or surface distinct-but-related rules (false positive). Tuning it properly
    needs a labeled dataset of dup/not-dup rule pairs, which is RAG-evaluation work deferred to
    Repo 2 (RAGAS). The CLI never acts on this score - the human approval gate is the real dedup,
    so a wrong threshold only costs a human a second glance, never a wrong deletion."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH};")
        cur.execute(
            """
            SELECT id, status, content,
                   1 - (embedding <=> (SELECT embedding FROM procedural_rules WHERE id = %s)) AS similarity
            FROM procedural_rules
            WHERE id != %s
            ORDER BY embedding <=> (SELECT embedding FROM procedural_rules WHERE id = %s)
            LIMIT %s;
            """,
            (rule_id, rule_id, rule_id, k),
        )
        rows = cur.fetchall()
    return [
        {"id": r[0], "status": r[1], "content": r[2], "similarity": float(r[3])}
        for r in rows
        if float(r[3]) > SIM_THRESHOLD
    ]

    
# ----- DANGER !! MURDER THE DB ---------------
def drop_schema() -> None:
    """
    Drop ALL memory tables (pr_audits, session_episodes, procedural_rules).

    DESTRUCTIVE and irreversible: permanently deletes every stored audit,
    session episode and procedural rule.

    Requires interactive confirmation: the caller must type 'yes' at the prompt.
    """

    print(
        f"{_RED}{_BOLD}"
        "============================================================\n"
        "  ⚠  DANGER: this will PERMANENTLY DROP these tables:\n"
        f"      {', '.join(_MEMORY_TABLES)}\n"
        "  All stored audits, episodes and rules will be lost.\n"
        "  This cannot be undone.\n"
        "============================================================"
        f"{_RESET}"
    )
    answer = input(f"{_RED}Type 'yes' to confirm: {_RESET}").strip().lower()
    if answer not in ("yes" , "y"):
        print("Aborted - no tables dropped.")
        return
    
    print(f"{_RED}{_BOLD}⚠ Dropping all tables ... !!!{_RESET}")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {', '.join(_MEMORY_TABLES)};")
        conn.commit()
        
    
if __name__ == "__main__":
    # `python -m src.db.vectorstore` initialises the schema once
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "drop":
        drop_schema()
    else:
        init_schema()
        print("pgvector schema initialised.")
