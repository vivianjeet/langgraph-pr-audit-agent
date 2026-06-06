from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from src.memory import AgentMemorySystem as AMS, AMSState
from src.nodes.ingest import ingest_pr_node
from src.nodes.security_audit import security_audit_node
from src.nodes.quality_audit import quality_audit_node
from src.nodes.coverage_audit import coverage_audit_node
from src.nodes.synthesize_report import synthesize_report_node
from src.nodes.plan import plan_audit_node
from src.nodes.reflexion import reflexion_node
from src.nodes.retrieve import retrieve_context_node
from src.nodes.finalize import finalize_report_node
from src.nodes.compress import compress_node
from src.nodes.compliance import compliance_node
from src.state import Severity

#=================================================================
# Node Stubs
#=================================================================

def human_review_node(state: AMSState):
    return {}


#=================================================================
# Routing Logic
#=================================================================

AUTH_HINTS = ("auth", "login", "session", "credential", "token", "password", "permission")
REFLECT_LO, REFLECT_HI = 0.5, 0.7
MAX_REFLECTIONS = 2
SCORE_KEYS = ("security_score", "quality_score", "test_score")

def should_reflect(state: AMSState) -> bool:
    """
    Decides if the AI needs to critique its own work or move forward
    Reflect (self-critique then re-audit) when the result is uncertain:
        - ANY of borderline security_score in [0.5,0.7], OR
        - an auth-related file changed but security found nothing (suspicious silence)
    Hard cap: never reflect more than twice (iteration_count guard).
    """

    ams = AMS(state)
    if ams.read("iteration_count",0) >= MAX_REFLECTIONS:
        return False

    # Borderline on any dimension -> worth a sharper second pass
    for key in SCORE_KEYS:
        if REFLECT_LO <= ams.read(key,1.0) <= REFLECT_HI:
            return True

    # Suspicious silence: auth-related change but no security findings -> reflect to see
    # if we missed something
    files = ams.read("files_changed",[])
    touched_auth = any(any(h in f.lower() for h in AUTH_HINTS) for f in files)
    if touched_auth and len(ams.read("security_findings",[])) == 0:
        return True
    return False

def needs_human_review(state: AMSState) -> bool:
    """
    Escalate to a human on any CRITICAL finding or a low score (<0.5).
    """
    ams = AMS(state)
    if any(ams.read(key,1.0) < 0.5 for key in SCORE_KEYS):
        return True
    all_findings = (
        ams.read("security_findings",[])
        + ams.read("quality_findings",[])
        + ams.read("test_findings",[])
    )
    return any(f.severity == Severity.CRITICAL for f in all_findings)

def route_after_synthesis(state: AMSState) -> str:
    """
    Combine the two predicates into one routing decision LangGraph can map to modes
    Decide whether to reflect or continue to human review after seeing the synthesized 
    report
    """

    if needs_human_review(state):
        return "human_review"
    if should_reflect(state):
        return "reflect"
    return "finalize"

#=================================================================
# Wiring the graph toloplogy
#=================================================================

builder = StateGraph(AMSState)

# Add all nodes to the graph
builder.add_node("ingest", ingest_pr_node)
builder.add_node("retrieve", retrieve_context_node)
builder.add_node("plan", plan_audit_node)
builder.add_node("security_audit", security_audit_node)
builder.add_node("quality_audit", quality_audit_node)
builder.add_node("coverage_audit", coverage_audit_node)
builder.add_node("synthesize", synthesize_report_node)
builder.add_node("reflexion", reflexion_node)
builder.add_node("human_review", human_review_node)
builder.add_node("compress", compress_node)
builder.add_node("finalize", finalize_report_node)
builder.add_node("compliance", compliance_node)

# Linear flow at the start
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "retrieve")
builder.add_edge("retrieve", "compliance")
builder.add_edge("compliance", "plan")

# Fan out: plan sent to three audit nodes in parallel
builder.add_edge("plan","security_audit")
builder.add_edge("plan","quality_audit")
builder.add_edge("plan","coverage_audit")

# Fan in : Wait for all three audits to finish, then synthesize
builder.add_edge("security_audit","synthesize")
builder.add_edge("quality_audit","synthesize")
builder.add_edge("coverage_audit","synthesize")

# Conditional routing : Do we reflect or continue to human review ?
builder.add_conditional_edges(
    "synthesize",
    route_after_synthesis,
    {
        "reflect":"reflexion",
        "human_review":"human_review",
        "finalize":"compress"        # go through compress before finalize
    }
)

# If we reflect, we go back to plan stage
builder.add_edge("reflexion","plan")

# Both finalize paths (clean audit + post human review) funnel through compress, so finalize
# always sees a fresh `compressed` channel (compress is a pass-through when not triggered).
builder.add_edge("human_review","compress")
builder.add_edge("compress","finalize")
builder.add_edge("finalize",END)

#=================================================================
# Compile the graph
#=================================================================

# We use memory saver to give the graph "threads" (checkpointing).
# Allow-list our own domain types so the checkpointer can (de)serialize them on resume
# (Severity, *Finding, AuditDepth, etc.). langgraph-checkpoint >=4 requires explicit
# (module, qualname) tuples - a bare "src.state" module string is silently ignored and the
# types come back as plain dicts (then `finding.severity` blows up on the human-review resume).
# DERIVE the list from src.state's own enum/model classes so adding a type can't drift it.
import enum as _enum, inspect as _inspect
from pydantic import BaseModel as _BaseModel
import src.state as _state
_ALLOWED_STATE_TYPES = [
    (_state.__name__, name)
    for name, obj in vars(_state).items()
    if _inspect.isclass(obj) and obj.__module__ == _state.__name__
    and issubclass(obj, (_enum.Enum, _BaseModel))
]
serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_STATE_TYPES)


def build_app(checkpointer):
    """Compile the graph against ANY checkpointer (MemorySaver, AsyncSqliteSaver, ...).
    The topology and the human-review interrupt are identical regardless of where threads are
    persisted - only durability changes. interrupt_before is the hard stop for human approval."""
    return builder.compile(checkpointer=checkpointer, interrupt_before=["human_review"])


# Default checkpointer: in-process, in-RAM threads. Compiled at import so callers can do
# `from src.graph import app`. Ephemeral - threads vanish when the process exits, which is
# fine for one-shot runs and CI gating (CI never resumes; it gates on the exit code).
app = build_app(MemorySaver(serde=serde))


@asynccontextmanager
async def durable_app(db_path: str = "checkpoints.sqlite"):
    """Opt-in durable checkpointing: yield an `app` whose threads persist to a SQLite file, so a
    run that PAUSED for human review can resume in a LATER process. Uses AsyncSqliteSaver (not the
    sync SqliteSaver) because the audit graph runs on app.astream / aget_state - the async driver
    calls the async checkpoint methods, which only the aio saver implements.

    The SQLite dependency is imported HERE, not at module top, so the default in-RAM path never
    requires the optional package. Reuses the same `serde` as MemorySaver, so domain types
    (Severity, *Finding, AuditDepth, ...) round-trip identically through either backend.

    Usage:  async with durable_app() as app: await run_audit(diff, app=app)
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        saver.serde = serde          # from_conn_string takes no serde= kwarg; set it on the instance
        yield build_app(saver)
