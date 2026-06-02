"""Tests in this file: the compiled graph's structure (no LLM/DB).

- test_graph_compiles_and_has_expected_nodes: the StateGraph builds and contains all expected nodes.
"""
from src.graph import builder


def test_graph_compiles_and_has_expected_nodes():
    compiled = builder.compile()
    nodes = compiled.get_graph().nodes
    for n in ["ingest", "retrieve", "plan", "security_audit", "quality_audit",
              "coverage_audit", "synthesize", "reflexion", "human_review", "finalize"]:
        assert n in nodes