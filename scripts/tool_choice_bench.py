# measure the token cost of each tool-choice mode on one fixed diff.
# Uses Gemini function-calling modes (FunctionCallingConfig.mode): AUTO / ANY / NONE.
# auto:  model decides whether to call (baseline, includes "should I call a tool?" reasoning)
# any:   model MUST call at least one tool (skips the decision; with allowed_function_names=[one]
#        it is FORCED to a specific tool - the "tool" mode below)
# tool:  ANY mode pinned to one allowed function (forced extraction - skips decision AND selection)
# parallel: two tools offered in AUTO so the model may emit both calls in one turn
from google import genai
from google.genai import types
from src.llm_retry import genai_client            # the spine's configured + key-rotated client

DIFF = "diff --git a/log.py b/log.py\n@@\n+    logger.info(f'user {user.pan} logged in')\n"

search_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="search_compliance_docs",
    description="Search the multi-framework compliance corpus.",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"]})])
history_tool = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="get_pr_audit_history",
    description="Return similar past PR audits.",
    parameters={"type": "object",
                "properties": {"query": {"type": "string"}}, "required": ["query"]})])


def _cfg(tools, mode, allowed=None):
    fc = types.FunctionCallingConfig(mode=mode, allowed_function_names=allowed)
    return types.GenerateContentConfig(
        tools=tools, max_output_tokens=500,
        tool_config=types.ToolConfig(function_calling_config=fc),
        system_instruction="Decide which compliance lookups this diff needs.")

MODES = {
    "auto":     ([search_tool], "AUTO", None),
    "any":      ([search_tool], "ANY", None),                                  # >=1 call
    "tool":     ([search_tool], "ANY", ["search_compliance_docs"]),            # forced to one
    "parallel": ([search_tool, history_tool], "AUTO", None),
}


def _run(mode, spec):
    tools, fc_mode, allowed = spec
    client = genai_client()
    resp = client.models.generate_content(
        model="gemini-2.5-flash", contents=DIFF, config=_cfg(tools, fc_mode, allowed))
    um = resp.usage_metadata
    calls = [p for c in (resp.candidates or []) for p in (c.content.parts or [])
             if getattr(p, "function_call", None)]
    return mode, um.prompt_token_count, um.candidates_token_count, len(calls)


def main():
    print(f"{'mode':10s}{'inTok':>7s}{'outTok':>7s}{'#calls':>7s}")
    for mode, spec in MODES.items():
        m, i, o, n = _run(mode, spec)
        print(f"{m:10s}{i:7d}{o:7d}{n:7d}")

if __name__ == "__main__":
    main()