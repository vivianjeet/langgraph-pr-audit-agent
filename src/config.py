# src/config.py - the single place every tunable value lives. Change a knob here and the whole
# system follows. Near-leaf: imports the stdlib + src.state (for the Severity-keyed penalty table
# ONLY). state.py imports NOTHING from config, so config -> state is one-directional - no cycle.
# Everyone else imports `src.config as cfg`. Domain TYPES (enums, message-prefix sentinels) stay in
# src/state.py. The tier->model POLICY (fallback + _Tier) stays in src/llm_client.py; it composes
# the model strings FROM here.
#
# Constants are grouped by WHERE they are used. A group used by a single file names that file; a
# group shared across several files lists them all. The comment above each constant says the same.
import os
from dotenv import load_dotenv
from src.state import Severity

load_dotenv()

# ════════════════════════════ Gemini model identities (single provider) ════════════════════════════
# The literal model names Google ships. flash-lite < flash < pro on both capability and price.
# Used by: src/citations.py, src/compression.py, src/db/vectorstore.py, and every audit/plan node
# (security_audit / quality_audit / coverage_audit / plan / compliance / reflexion). Day 30+ the
# router's TIER_TABLE (src/llm_client.py) will compose its tiers from these same names.
GEMINI_FLASH_LITE_MODEL = "gemini-2.5-flash-lite"   # cheapest triage/extraction tier (router, Day 30+)
GEMINI_FLASH_MODEL      = "gemini-2.5-flash"         # default audit/plan/compliance reasoning
GEMINI_PRO_MODEL        = "gemini-2.5-pro"           # reflexion's smarter critique pass
GEMINI_EMBED_MODEL      = "gemini-embedding-001"     # embeddings (vectorstore); 768-dim by default

# The citation/cite-tier model, named as its OWN knob so it can diverge from the audit model
# without dragging the audit nodes along (a Flash-class extraction task today). BOTH consumers read
# this one constant: src/citations.py (CITATION_MODEL) AND the router's TIER_TABLE["cite"]
# (src/llm_client.py, Day 30+). To put citations on a different model, change just this line.
CITE_MODEL = GEMINI_FLASH_MODEL

# ════════════════════════════ LLM output-token budgets (max_output_tokens) ════════════════════════════
# Per-call output ceilings. Each is used by the file named on its line.
AUDIT_MAX_OUTPUT_TOKENS      = 4000    # security_audit / quality_audit / coverage_audit / plan nodes
REFLEXION_MAX_OUTPUT_TOKENS  = 6000    # src/nodes/reflexion.py (the critique pass)
COMPLIANCE_MAX_OUTPUT_TOKENS = 2000    # src/nodes/compliance.py (regulatory triage)
CITATION_MAX_OUTPUT_TOKENS   = 1024    # src/citations.py (grounded-citation extraction)
SUMMARY_MAX_OUTPUT_TOKENS    = 1024    # src/compression.py (history-summary output)

# ════════════════════════════ History compression (src/compression.py) ════════════════════════════
# All used only in src/compression.py: the trigger threshold, the fold ratio, the recent-message
# floor, and the demo context budget the standalone pass runs against.
COMPRESSION_DEMO_BUDGET_TOKENS = 10000   # context budget the demo compression pass measures against
COMPRESSION_TRIGGER_RATIO      = 0.8     # fire compression when usage >= 80% of the budget
COMPRESSION_FOLD_RATIO         = 0.5     # collapse the oldest 50% of the message list
COMPRESSION_MIN_RECENT_KEEP    = 2       # always keep at least this many newest messages verbatim

# ════════════════════════════ Vector store / pgvector (src/db/vectorstore.py) ════════════════════════════
# All used only in src/db/vectorstore.py (DATABASE_URL also read by scripts/check_corpus.py via env;
# EMBED_OUTPUT_DIM is additionally imported by scripts/bench_embed.py and a couple of tests).
DATABASE_URL                = os.environ.get("DATABASE_URL")
EMBED_OUTPUT_DIM            = 768   # embedding dimension; MUST match GEMINI_EMBED_MODEL's output size
EMBED_BATCH_SIZE            = 100   # texts per embed round-trip (keeps under the per-minute quota)
EMBED_CACHE_SIZE            = 256   # lru_cache size on embed()
DB_CONNECT_TIMEOUT_SECONDS  = 5     # psycopg connect timeout
HNSW_M                      = 16    # pgvector HNSW graph degree
HNSW_EF_CONSTRUCTION        = 64    # HNSW build-time candidate list size
HNSW_EF_SEARCH              = 100   # HNSW query-time candidate list size
SIMILARITY_THRESHOLD        = 0.7   # cosine-similarity cutoff; distance = 1 - sim, so keep dist < 0.3

# ════════════════════════════ Recall / search result counts (k) ════════════════════════════
# How many results each recall/search returns.
RECALL_SIMILAR_PRS_K = 3   # src/nodes/retrieve.py - similar past PR audits
RECALL_EPISODES_K    = 2   # src/nodes/retrieve.py - similar past sessions
SEARCH_DEFAULT_K     = 3   # default k for src/db/vectorstore.py, src/memory.py, src/nodes/compliance.py

# ════════════════════════════ LLM retry / key-rotation spine (src/llm_retry.py) ════════════════════════════
# All used only in src/llm_retry.py - the tenacity retry policy + Instructor's own retry count.
MAX_SERVER_RETRY_WAIT_SECONDS = 90.0   # never honour a server Retry-After longer than this
RETRY_MAX_ATTEMPTS            = 5       # tenacity stop_after_attempt
RETRY_BACKOFF_MIN_SECONDS     = 1       # wait_exponential min
RETRY_BACKOFF_MAX_SECONDS     = 30      # wait_exponential max
INSTRUCTOR_MAX_RETRIES        = 2       # Instructor's structured-output validation retries

# ════════════════════════════ Graph routing thresholds (src/graph.py) ════════════════════════════
# All used in src/graph.py (LOW_SCORE_THRESHOLD is ALSO used by src/evaluators.py).
MAX_REFLECTION_PASSES = 2                          # hard cap on self-critique re-audits
REFLECT_SCORE_LOW     = 0.5                        # borderline-score band: low edge (reflect inside it)
REFLECT_SCORE_HIGH    = 0.7                        # borderline-score band: high edge
LOW_SCORE_THRESHOLD   = 0.5                        # escalate / inconsistency line (graph.py + evaluators.py)
SCORE_KEYS            = ("security_score", "quality_score", "test_score")
AUTH_FILE_HINTS       = ("auth", "login", "session", "credential", "token", "password", "permission")

# ════════════════════════════ Presentation / report shape ════════════════════════════
# Text-clip widths and how many findings/queries to surface.
CLIP_WIDTH_DEFAULT    = 140   # src/nodes/retrieve.py (episodes), src/nodes/finalize.py
CLIP_WIDTH_SHORT      = 120   # src/nodes/retrieve.py (PR summaries)
CLIP_WIDTH_LONG       = 160   # src/nodes/plan.py, src/nodes/compliance.py
TOP_FINDINGS_PER_DIM  = 3     # src/nodes/finalize.py - N worst findings surfaced per dimension
MAX_COMPLIANCE_QUERIES = 3    # src/nodes/compliance.py - triage queries actually executed

# ════════════════════════════ Risk scoring (src/nodes/synthesize_report.py) ════════════════════════════
# Multiplicative score penalty per finding severity. Used only in synthesize_report.py.
SEVERITY_PENALTY = {
    Severity.CRITICAL: 0.6,
    Severity.HIGH:     0.3,
    Severity.MEDIUM:   0.15,
    Severity.LOW:      0.07,
    Severity.INFO:     0.02,
    Severity.NONE:     0.0,
}
