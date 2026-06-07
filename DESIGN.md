# Design notes

The [README](README.md) describes *what* the system does and how to run it. This document collects
the *why* - the deliberate trade-offs behind the design, and the one capability (large-diff handling)
that is built but not wired into the live audit.

## Why these design decisions

Each choice is a deliberate trade-off, not an accident of how it grew.

**Why four memory types instead of one vector store?** Different recall needs different mechanisms.
Precedent ("have we seen a PR like this?") and past sessions need *similarity* search; org rules need
*exact* category lookup, not a fuzzy match. Collapsing them into one store would force similarity
semantics onto rules that must apply verbatim. So semantic and episodic use pgvector; procedural is a
plain keyed table. See [Agent Memory](README.md#agent-memory-four-types-one-system).

**Why is learned-rule activation gated by a human?** Rules derived from the agent's own findings are a
feedback loop - one false-positive CRITICAL would otherwise become a permanent rule injected into
every future audit, with nothing to un-learn it. Learned rules land as `learned_pending` and never
inject until a human approves them. The agent proposes; a human decides. A verdict of `needs-changes`
or `reject` on the PR suppresses learning entirely - findings on code that is being revised or
abandoned should not become standing rules.

**Why does compression write to its own channel instead of editing the transcript?** The working
message list is append-only by reducer design - that reducer is what protects the parallel audit
fan-out from clobbering itself - so compression *cannot* overwrite it. The compacted history goes to a
separate channel, which is also conceptually right: a compressed session is proto-episodic memory, so
`finalize` promotes it into the episodic store. See
[History compression](README.md#history-compression-the-compress-node).

**Why is the budget manager not wired into the live audit?** A single PR diff is far below the model's
window, so an in-band budget check would trim nothing and only add latency. It is built and tested
under synthetic load, ready for genuinely large inputs - wiring it now would be cost with no benefit.
See [handling very large diffs](#handling-very-large-1m-token-pr-diffs) below.

**Why fail closed?** A transport or auth failure that left scores at a default 1.0 would let a broken
audit look like a clean PR. Instead, when an audit node errors the scores are forced to 0.0 and the
run escalates - a failure is never a false pass.

**Why multiplicative scoring?** A linear penalty sum let several moderate findings drive a score to
exactly 0.0 purely by count - a messy class scoring the same as a catastrophe. Each finding instead
multiplicatively erodes the remaining score, so several moderate findings trend low with diminishing
returns while a single CRITICAL still bites hard. Severity drives the risk, not raw finding count.

**Why async audits but synchronous everything else?** Only the three audits run on the same parallel
step, so only they benefit from concurrency. The sequential nodes (compliance triage, plan, reflexion)
call the router's sync `llm.call`; the parallel fan-out calls `llm.acall`. Same tier table and
fallback chain either way - the split is purely about whether a node needs the event loop, not two
different routers. The blocking Gemini call runs on a worker thread, reusing the existing retry /
key-rotation stack rather than reimplementing it against an async client.

**Why cache the diff for most nodes but the prefix for security?** These are two different reuse
patterns, and a Gemini `CachedContent` is bound to one model, so they can't be the same cache.
*Within* one audit the same diff is sent by every Flash node (compliance, plan, quality, coverage)
while their instructions differ - so the diff is the high-reuse part, cached once and reused across
them. *Across* audits the security prompt's prefix (instructions + rules + compliance) is identical
for different PRs of the same corpus - so for security the prefix is the cross-PR reuse part. Security
takes this Pro-tier cache path only on a regulated diff (one with compliance context); an unregulated
diff runs the security audit on plain Flash like the other nodes. On its Pro path it could not share
the Flash diff handle regardless, since a `CachedContent` is model-bound. **Why does compliance prime
the diff cache?** It runs first and sequentially, so it creates the handle before the parallel
quality/coverage calls reach it - they become pure reusers and the concurrent create-race never
happens. We did not need a lock; ordering solved it. Both caches honour Gemini's 2048-token floor:
under it the call is rejected and the node falls back to a plain (uncached) call, so a small diff or a
small prefix costs nothing extra. The diff clears the floor on large PRs (where the saving matters);
the prefix usually does not yet, so security's cache is a documented forward-looking path for batch
runs rather than a live saving today.

**Why does the tier router re-raise `QuotaExhaustedError` instead of a generic error?** The router
walks a fallback chain and, when every tier fails, raises. But the nodes' fail-closed contract keys off
the *type*: `except QuotaExhaustedError` aborts the run rather than degrading to a clean score. If the
router masked a total quota exhaustion as a plain `RuntimeError`, that except would miss and a node
could record a false-clean result. So the router re-raises `QuotaExhaustedError` as itself when the
cause is quota; only a genuinely different failure becomes the generic "all tiers failed" error.

**How is key rotation concurrency-safe?** Because the three audits run at once, they can all hit a
dead key in the same instant and each try to rotate - which would skip past good keys
(`KEY1 → KEY2 → KEY3 → KEY4` for a single exhaustion). A `threading.Lock` plus a *double-checked*
rotation fixes this: under the lock a thread re-checks whether the key it saw fail is still current;
if another thread already rotated, it rides along instead of rotating again, and clients are rebound
inside the lock so a key and its client never desync. The lock lives in the retry layer (the one door
every Gemini call passes through), not on the nodes.

**Why structured outputs everywhere (Instructor) and no manual JSON parsing?** Every LLM node returns
a Pydantic `response_model` through one call path (`call_gemini` / `call_gemini_async`), so there is
no hand-rolled `json.loads` of model output anywhere in `src/` - the only `json` use is JSONB column
serialization in the DB layer. Each output field carries a `Field(description=...)`, which Instructor
puts into the schema sent to the model; that is what actually moves output quality, not "using
Instructor" as a label.

**Why keep Instructor instead of switching to raw tool-choice (`tool_config`) everywhere?** Gemini's
`FunctionCallingConfig` can force a call (`mode=ANY` pinned to one function) - the benchmark in
`scripts/tool_choice_bench.py` shows that forcing is the cheapest mode because it drops the "should I
call?" reasoning. But the audit's forced calls are all *structured extraction* into a pydantic model,
and Instructor already forces that on the Gemini spine AND retries on a schema-validation failure -
strictly more than raw `tool_config`, which forces the call but does nothing about a malformed result.
So Instructor stays the path for the schemas; the tool-choice benchmark earns its place as the measured
token story and for the one thing it adds that Instructor does not - `parallel` mode, where a diff
needing two MCP lookups returns both calls in one round-trip instead of two sequential ones.

**Why one central `src/config.py` instead of constants next to where they're used?** Model names,
token ceilings, retry limits, score thresholds and similarity cut-offs were scattered across nodes,
and the same value (a `0.5` score floor, a `768` embedding dimension, a model name) appeared in more
than one file - so a change had to be made in several places and could silently drift out of sync.
Pulling every tunable into one module, grouped by where it's used and imported as `cfg`, makes each
value defined once: the tier table reads its models from `cfg`, the retry spine reads its limits from
`cfg`, the graph reads its thresholds from `cfg`. The constraint that keeps this clean is direction:
`config` imports only stdlib (and `Severity` from `state.py`), and `state.py` never imports `config` -
a one-way edge with no cycle, so config stays a leaf the whole tree can depend on.

## Handling very large (1M+ token) PR diffs

This repo does **not** route the live audit through the budget manager, and that is deliberate: a
single PR diff is far smaller than Gemini's ~1M-token window, so an in-band budget check here would
always keep everything, trim nothing and only add latency. The class and its tests are the artifact;
it is demonstrated under synthetic oversized load in `tests/test_token_budget.py`.

If you are cloning this to handle genuinely huge diffs, the budget manager is the **last** mile, not
the whole fix. A diff larger than the model window breaks in three places, in this order:

1. **Embedding (breaks first).** `retrieve` and `finalize` embed the diff for similarity search and
   storage, and embedding models have a much smaller input limit than the chat window. You cannot fix
   this by trimming the text before embedding - a trimmed embedding represents a *different* text than
   the real diff, so similarity search returns wrong results. The fix is **chunked ingestion +
   retrieval** (split the diff per file/hunk, embed each chunk, retrieve the relevant ones).
2. **Parsing.** `parse_github_diff` (`src/nodes/ingest.py`) keeps every added/removed line, so a huge
   diff stays huge after parsing. You need a pre-reduction step (per-file summaries or changed-hunk
   headers).
3. **Prompt assembly (the budget manager's job).** Once the pieces are reasonably sized, route the
   prompt assembly in `plan.py` (and the audit nodes) through `TokenBudgetManager.fit(...)` instead of
   concatenating the diff and precedent directly, then set `budget_tokens` to your model's real window
   and (optionally) pass a real `counter`.

In short: handling a 1M+ diff is primarily a chunking/RAG problem (steps 1-2); the budget manager
cleanly handles the final fit (step 3) and is built to drop into that pipeline unchanged.
