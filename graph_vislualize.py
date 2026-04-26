"""
Graph Visualization — Renders the pipeline architecture as Mermaid/PNG.

Uses a lightweight mirror of the orchestrator's graph structure
for clean visualization without runtime state issues.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.graph import StateGraph, END
from typing import Annotated
from typing_extensions import TypedDict
import operator


# Minimal typed state just for visualization
class VizState(TypedDict):
    query: str


def _noop(state):
    return state


def build_viz_graph():
    """Build a visualization-only mirror of the orchestrator graph."""
    graph = StateGraph(VizState)

    nodes = [
        "session_load",
        "intent_plan",
        "direct_end",
        "parallel_dispatch",
        "data_validate",
        "crag_grade",
        "credibility_score",
        "bias_detect",
        "source_compare",
        "synthesize",
        "hallucination_check",
        "critique",
        "format_output",
    ]
    for n in nodes:
        graph.add_node(n, _noop)

    graph.set_entry_point("session_load")
    graph.add_edge("session_load", "intent_plan")
    graph.add_conditional_edges(
        "intent_plan",
        lambda s: "direct_end",
        {"direct_end": "direct_end", "parallel_dispatch": "parallel_dispatch"},
    )
    graph.add_edge("direct_end", END)
    graph.add_edge("parallel_dispatch", "data_validate")
    graph.add_edge("data_validate", "crag_grade")
    graph.add_edge("crag_grade", "credibility_score")
    graph.add_edge("credibility_score", "bias_detect")
    graph.add_edge("bias_detect", "source_compare")
    graph.add_edge("source_compare", "synthesize")
    graph.add_edge("synthesize", "hallucination_check")
    graph.add_edge("hallucination_check", "critique")
    graph.add_conditional_edges(
        "critique",
        lambda s: "format_output",
        {"synthesize": "synthesize", "format_output": "format_output"},
    )
    graph.add_edge("format_output", END)

    return graph.compile()


def generate_graph_image():
    print("Generating Graph Visualization...")
    compiled = build_viz_graph()

    # Try PNG first
    try:
        graph_image = compiled.get_graph().draw_mermaid_png()
        with open("multi_agent_architecture.png", "wb") as f:
            f.write(graph_image)
        print("Success! Image saved as 'multi_agent_architecture.png'")
        return
    except Exception as e:
        print(f"PNG render failed ({e}), falling back to Mermaid text...")

    # Fallback: Mermaid text
    try:
        mermaid_text = compiled.get_graph().draw_mermaid()
        with open("multi_agent_architecture.mmd", "w", encoding="utf-8") as f:
            f.write(mermaid_text)
        print(f"Mermaid diagram saved as 'multi_agent_architecture.mmd'")
        print("\nMermaid source:\n")
        print(mermaid_text)
        print("\nPaste this into https://mermaid.live to render it.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    generate_graph_image()
