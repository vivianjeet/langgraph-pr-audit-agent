from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AuditState
from src.nodes.ingest import ingest_pr_node
from src.nodes.security_audit import security_audit_node
from src.nodes.quality_audit import quality_audit_node
from src.nodes.coverage_audit import coverage_audit_node
from src.nodes.synthesize_report import synthesize_report_node
from src.nodes.plan import plan_audit_node
from src.nodes.reflexion import reflexion_node
from src.nodes.retrieve import retrieve_context_node
from src.nodes.finalize import finalize_report_node
from src.state import Severity

#=================================================================
# Node Stubs
#=================================================================

def human_review_node(state: AuditState):
    return {"messages" : ["System: Human Approved Report"]}


#=================================================================
# Routing Logic
#=================================================================

AUTH_HINTS = ("auth", "login", "session", "credential", "token", "password", "permission")
REFLECT_LO, REFLECT_HI = 0.5, 0.7
MAX_REFLECTIONS = 2
SCORE_KEYS = ("security_score", "quality_score", "test_score")

def should_reflect(state: AuditState) -> bool:
    """
    Decides if the AI needs to critique its own work or move forward
    Reflect (self-critique then re-audit) when the result is uncertain:
        - ANY of borderline security_score in [0.5,0.7], OR
        - an auth-related file changed but security found nothing (suspicious silence)
    Hard cap: never reflect more than twice (iteration_count guard).
    """

    if state.get("iteration_count",0) >= MAX_REFLECTIONS:
        return False
    
    # Borderline on any dimension -> worth a sharper second pass
    for key in SCORE_KEYS:
        if REFLECT_LO <= state.get(key,1.0) <= REFLECT_HI:
            return True
    
    # Suspicious silence: auth-related change but no security findings -> reflect to see 
    # if we missed something
    files = state.get("files_changed",[])
    touched_auth = any(any(h in f.lower() for h in AUTH_HINTS) for f in files)
    if touched_auth and len(state.get("security_findings",[])) == 0:
        return True
    return False

def needs_human_review(state: AuditState) -> bool:
    """
    Escalate to a human on any CRITICAL finding, or a low score (<0.5).
    """
    if any(state.get(key,1.0) < 0.5 for key in SCORE_KEYS):
        return True
    all_findings = (
        state.get("security_findings",[])
        + state.get("quality_findings",[])
        + state.get("test_findings",[])
    )
    return any(f.severity == Severity.CRITICAL for f in all_findings)

def route_after_synthesis(state: AuditState) -> str:
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

builder = StateGraph(AuditState)

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
builder.add_node("finalize", finalize_report_node)

# Linear flow at the start
builder.add_edge(START, "ingest")
builder.add_edge("ingest", "retrieve")
builder.add_edge("retrieve", "plan")

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
        "finalize":"finalize"
    }
)

# If we reflect, we go back to plan stage
builder.add_edge("reflexion","plan")

# FInalize the workflow
builder.add_edge("human_review","finalize")
builder.add_edge("finalize",END)

#=================================================================
# Compile the graph
#=================================================================

# We use memory saver to give the graph "threads" (checkpointing)
memory = MemorySaver()

# interrupt_before acts as hard stop for human approval
app = builder.compile(checkpointer=memory, interrupt_before=["human_review"])
