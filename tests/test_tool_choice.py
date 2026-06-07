# the mode spec is well-formed (we don't unit-test live token counts;
# the benchmark script is the live evidence). Guards against a malformed tool_config spec.
import scripts.tool_choice_bench as b


def test_specific_tool_mode_pins_one_function():
    tools, mode, allowed = b.MODES["tool"]
    assert mode == "ANY" and allowed == ["search_compliance_docs"]   # forced to ONE tool


def test_any_mode_requires_a_call():
    _, mode, allowed = b.MODES["any"]
    assert mode == "ANY" and allowed is None                         # >=1 call, model picks which


def test_parallel_mode_offers_both_tools():
    tools, mode, _ = b.MODES["parallel"]
    assert len(tools) == 2 and mode == "AUTO"                        # two tools in one turn possible