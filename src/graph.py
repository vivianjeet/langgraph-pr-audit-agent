from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AuditState
from src.nodes.ingest import ingest_pr_node

#=================================================================
# Node Stubs
#=================================================================

def retrieve_context_node(state: AuditState):
    return {"messages" : ["System: Retrieved similar PR history"]}

def plan_audit_node(state: AuditState):
    return {"messages" : ["System: Created target audit plan"]}

def security_audit_node(state: AuditState):
    return {"messages" : ["System: Security checks complete"]}

def quality_audit_node(state: AuditState):
    return {"messages" : ["System: Quality checks complete"]}

def test_audit_node(state: AuditState):
    return {"messages" : ["System: Testing checks complete"]}

def synthesize_report_node(state: AuditState):
    return {"messages" : ["System: Synthesized findings into draft report"]}

def reflexion_node(state: AuditState):
    #Increpemt loop counter so as to not to get stuck for ever
    current_count = state.get("iteration_count",0)

    return {
        "messages" : ["System: Critiques report, Found gaps."],
        "iteration_count" : current_count + 1
    }

def human_review_node(state: AuditState):
    return {"messages" : ["System: Human Approved Report"]}

def finalize_report_node(state: AuditState):
    return {"messages" : ["System: Finalized and published report"]}

#=================================================================
# Routing Logic
#=================================================================

def should_reflect(state: AuditState) -> str:
    """Decides if the AI needs to critique its own work or move forward"""
    iterations = state.get("iteration_count",0)
    # stop reflecting if  we already looped twice
    if iterations >=2:
        return "continue"
    
    return "reflect"

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
builder.add_node("test_audit", test_audit_node)
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
builder.add_edge("plan","test_audit")

# Fan in : Wait for all three audits to finish, then synthesize
builder.add_edge("security_audit","synthesize")
builder.add_edge("quality_audit","synthesize")
builder.add_edge("test_audit","synthesize")

# Conditional routing : Do we reflect or continue to human review ?
builder.add_conditional_edges(
    "synthesize",
    should_reflect,
    {
        "reflect":"reflexion",
        "continue":"human_review"
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
