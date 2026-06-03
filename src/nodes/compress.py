# Compression node: compacts the oldest slice of the session's message history into a single
# summary, written to the `compressed` channel (proto-episodic). Runs just before finalize.
# Pass-through when not triggered: returns {} so the graph flows on with state untouched.
from src.memory import AgentMemorySystem as AMS, AMSState
from src.compression import run_compression_pass

def compress_node(state: AMSState):
    """Force via the entry `force_compress` flag (--large), else auto-fire at 80% of budget.
    Writes the `compressed` channel; finalize then stores it as the episode. No-op otherwise."""
    ams = AMS(state)
    messages = ams.read("messages", [])
    force = bool(state.get("force_compress", False))   # top-level channel, not in audit substate
    update, report = run_compression_pass(messages, force=force)
    print(report)
    return update                                      # {"compressed":[...]} or {} (pass-through)
